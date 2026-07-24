#!/usr/bin/env python3
"""Evaluate validation-selected high-precision TIER intervention policies."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.evidence.tier_model import TransactionEvidenceEncoder
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.tier_phase1_evidence_qualification import (
    TASK,
    audit_protocol,
    best_f1_threshold,
    binary_f1,
    configure_fraudgt,
    load_a2_model,
    loader_audit,
    query_batch,
    seed_process,
    select_values_by_edge_id,
    verify_repository,
)


DIRECTIONS = {"both", "evidence_positive", "evidence_negative"}


def load_parent_manifest(root, task, expected_commit):
    matches = []
    for path in root.glob("*/experiment_manifest.json"):
        item = json.loads(path.read_text())
        if (
            item.get("dataset") == task["dataset"]
            and item.get("evidence_family") == task["family"]
            and item.get("evidence_selection") == task["selection"]
        ):
            matches.append((path, item))
    if len(matches) != 1:
        raise RuntimeError(
            "expected one completed Phase 1b manifest for "
            f"{task['dataset']}/{task['family']}/{task['selection']}, "
            f"found {len(matches)}"
        )
    path, manifest = matches[0]
    if manifest.get("sampling_protocol") != "dynamic_random":
        raise RuntimeError("parent manifest is not dynamic_random")
    if not str(manifest.get("git_commit", "")).startswith(expected_commit):
        raise RuntimeError("parent evidence commit differs from registration")
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"parent checkpoint is missing: {checkpoint}")
    return path, manifest, checkpoint


@torch.no_grad()
def paired_features(
    evidence_model,
    a2_model,
    loader,
    index,
    spec,
    task,
    device,
    split,
):
    evidence_model.eval()
    a2_model.eval()
    rows = {
        "target_edge_ids": [],
        "labels": [],
        "evidence_scores": [],
        "a2_scores": [],
        "support_count": [],
    }
    requested = 0
    paired = 0
    for batch in loader:
        (
            target_ids,
            labels,
            evidence,
            target_raw,
            _,
            evidence_cpu,
        ) = query_batch(
            index,
            batch,
            spec,
            task["selection"],
            device,
        )
        evidence_logits, _ = evidence_model(evidence, target_raw)
        batch.split = split
        batch.to(device)
        a2_logits, a2_labels = a2_model(batch)
        a2_logits = a2_logits.squeeze(-1).detach().cpu()
        a2_labels = a2_labels.detach().cpu().long().view(-1)
        target_ids_device = target_ids.to(device)
        target_mask = torch.isin(batch[TASK].e_id, target_ids_device)
        a2_target_ids = batch[TASK].e_id[target_mask].detach().cpu()
        if (
            a2_target_ids.numel() != a2_logits.numel()
            or a2_target_ids.numel() != a2_labels.numel()
        ):
            raise AssertionError("A2 output is not edge aligned")
        evidence_logits, labels, support = select_values_by_edge_id(
            a2_target_ids,
            target_ids,
            evidence_logits.detach().cpu(),
            labels,
            evidence_cpu.support,
        )
        if not torch.equal(labels, a2_labels):
            raise AssertionError("A2 and evidence labels differ")
        rows["target_edge_ids"].append(a2_target_ids)
        rows["labels"].append(labels)
        rows["evidence_scores"].append(torch.sigmoid(evidence_logits))
        rows["a2_scores"].append(torch.sigmoid(a2_logits))
        rows["support_count"].append(support[:, 0].float())
        requested += int(target_ids.numel())
        paired += int(a2_target_ids.numel())
    output = {key: torch.cat(parts) for key, parts in rows.items()}
    output["requested_samples"] = requested
    output["paired_samples"] = paired
    output["paired_retention_rate"] = paired / max(requested, 1)
    return output


def policy_mask(rows, policy, a2_threshold, evidence_threshold):
    a2_scores = rows["a2_scores"]
    evidence_scores = rows["evidence_scores"]
    a2_predictions = a2_scores >= a2_threshold
    evidence_predictions = evidence_scores >= evidence_threshold
    mask = a2_predictions != evidence_predictions
    if policy["direction"] == "evidence_positive":
        mask &= evidence_predictions
    elif policy["direction"] == "evidence_negative":
        mask &= ~evidence_predictions
    elif policy["direction"] != "both":
        raise ValueError(f"unknown direction: {policy['direction']}")
    mask &= (a2_scores - a2_threshold).abs() <= policy["a2_max_margin"]
    mask &= (
        (evidence_scores - evidence_threshold).abs()
        >= policy["evidence_min_margin"]
    )
    mask &= rows["support_count"] >= policy["support_min"]
    return mask


def intervention_statistics(
    rows,
    intervention,
    a2_threshold,
    evidence_threshold,
):
    labels = rows["labels"].bool()
    a2_predictions = rows["a2_scores"] >= a2_threshold
    evidence_predictions = rows["evidence_scores"] >= evidence_threshold
    a2_correct = a2_predictions == labels
    evidence_correct = evidence_predictions == labels
    corrected = intervention & (~a2_correct) & evidence_correct
    broken = intervention & a2_correct & (~evidence_correct)
    corrected_count = int(corrected.sum())
    broken_count = int(broken.sum())
    changed = int(intervention.sum())
    a2_errors = int((~a2_correct).sum())
    routed = torch.where(
        intervention, evidence_predictions, a2_predictions)
    ratio = (
        corrected_count / broken_count
        if broken_count
        else None
    )
    return {
        "samples": int(labels.numel()),
        "positives": int(labels.sum()),
        "a2_errors": a2_errors,
        "candidate_disagreements": int(
            (a2_predictions != evidence_predictions).sum()
        ),
        "changed_predictions": changed,
        "corrected_predictions": corrected_count,
        "broken_predictions": broken_count,
        "corrected_minus_broken": corrected_count - broken_count,
        "corrected_to_broken_ratio": ratio,
        "corrected_without_breaks": (
            corrected_count > 0 and broken_count == 0
        ),
        "correction_rate_on_a2_errors": (
            corrected_count / a2_errors if a2_errors else 0.0
        ),
        "a2_same_batch_f1": binary_f1(labels, a2_predictions),
        "routed_same_batch_f1": binary_f1(labels, routed),
        "delta_paired_f1": (
            binary_f1(labels, routed)
            - binary_f1(labels, a2_predictions)
        ),
    }


def passes_gate(stats, gate, require_f1_gain=False):
    required_changed = max(
        int(gate["min_changed_predictions"]),
        math.ceil(
            float(gate["min_changed_fraction_of_a2_errors"])
            * stats["a2_errors"]
        ),
    )
    ratio = stats["corrected_to_broken_ratio"]
    ratio_passed = (
        stats["corrected_without_breaks"]
        or (
            ratio is not None
            and ratio >= float(gate["min_corrected_to_broken_ratio"])
        )
    )
    passed = (
        stats["correction_rate_on_a2_errors"]
        >= float(gate["min_correction_rate_on_a2_errors"])
        and ratio_passed
        and stats["corrected_minus_broken"] > 0
        and stats["changed_predictions"] >= required_changed
    )
    if require_f1_gain:
        passed &= stats["delta_paired_f1"] > 0
    return passed, required_changed


def quantile_values(values, quantiles):
    if not values.numel():
        return [0.0]
    return sorted({
        float(torch.quantile(values.float(), float(q)))
        for q in quantiles
    })


def candidate_policies(rows, a2_threshold, evidence_threshold, grid):
    a2_predictions = rows["a2_scores"] >= a2_threshold
    evidence_predictions = rows["evidence_scores"] >= evidence_threshold
    disagreements = a2_predictions != evidence_predictions
    a2_margins = (
        rows["a2_scores"] - a2_threshold).abs()[disagreements]
    evidence_margins = (
        rows["evidence_scores"] - evidence_threshold
    ).abs()[disagreements]
    supports = rows["support_count"][disagreements]
    a2_cutoffs = quantile_values(
        a2_margins, grid["a2_max_margin_quantiles"])
    evidence_cutoffs = quantile_values(
        evidence_margins, grid["evidence_min_margin_quantiles"])
    support_cutoffs = quantile_values(
        supports, grid["support_min_quantiles"])
    policies = []
    for direction in grid["directions"]:
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown registered direction: {direction}")
        for a2_cutoff in a2_cutoffs:
            for evidence_cutoff in evidence_cutoffs:
                for support_cutoff in support_cutoffs:
                    policies.append({
                        "direction": direction,
                        "a2_max_margin": a2_cutoff,
                        "evidence_min_margin": evidence_cutoff,
                        "support_min": support_cutoff,
                    })
    return policies


def select_policy(rows, a2_threshold, evidence_threshold, grid, gate):
    evaluated = []
    for policy in candidate_policies(
        rows, a2_threshold, evidence_threshold, grid
    ):
        mask = policy_mask(
            rows, policy, a2_threshold, evidence_threshold)
        stats = intervention_statistics(
            rows, mask, a2_threshold, evidence_threshold)
        eligible, required_changed = passes_gate(stats, gate)
        evaluated.append({
            "policy": policy,
            "statistics": stats,
            "eligible": eligible,
            "required_changed_predictions": required_changed,
        })
    if not evaluated:
        raise RuntimeError("registered policy grid is empty")
    eligible = [row for row in evaluated if row["eligible"]]
    pool = eligible or evaluated

    def rank(row):
        stats = row["statistics"]
        ratio = stats["corrected_to_broken_ratio"]
        ratio_rank = (
            ratio
            if ratio is not None
            else (float("inf") if stats["corrected_without_breaks"] else -1.0)
        )
        return (
            stats["corrected_minus_broken"],
            ratio_rank,
            stats["corrected_predictions"],
            stats["changed_predictions"],
        )

    selected = max(pool, key=rank)
    return selected, len(evaluated), len(eligible)


def task_output_dir(output_root, task, commit):
    variant = f"{task['family']}_{task['selection']}"
    return (
        output_root
        / f"{task['dataset']}_{variant}_seed{task['seed']}_{commit}"
    )


def run_task(spec, task, args):
    commit = verify_repository(spec)
    config_relative = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    parent_path, parent, evidence_checkpoint = load_parent_manifest(
        args.phase1b_root, task, spec["evidence_commit"])

    seed_process(int(task["seed"]))
    configure_fraudgt(
        config_path, int(task["seed"]), args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    _, val_loader, test_loader = loaders
    device = torch.device(args.device)
    store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=store.edge_index,
        timestamps=store.timestamps,
        raw_edge_attr=store.raw_edge_attr,
    )
    model = TransactionEvidenceEncoder(
        num_currencies=int(store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(store.raw_edge_attr[:, 3].max()) + 1,
        family=task["family"],
        hidden_dim=int(spec["hidden_dim"]),
        num_heads=int(spec["num_heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    saved = torch.load(evidence_checkpoint, map_location=device)
    model.load_state_dict(saved["model_state"], strict=True)
    a2_model = load_a2_model(
        dataset, task["a2_checkpoint"], device)

    started = time.monotonic()
    val = paired_features(
        model, a2_model, val_loader, index, spec, task, device, "val")
    evidence_threshold, _ = best_f1_threshold(
        val["labels"], val["evidence_scores"])
    a2_threshold, _ = best_f1_threshold(
        val["labels"], val["a2_scores"])
    selected, policies_evaluated, eligible_policies = select_policy(
        val,
        a2_threshold,
        evidence_threshold,
        spec["policy_grid"],
        spec["qualification_gate"],
    )
    test = paired_features(
        model, a2_model, test_loader, index, spec, task, device, "test")
    test_mask = policy_mask(
        test,
        selected["policy"],
        a2_threshold,
        evidence_threshold,
    )
    test_stats = intervention_statistics(
        test, test_mask, a2_threshold, evidence_threshold)
    test_passed, test_required_changed = passes_gate(
        test_stats, spec["qualification_gate"], require_f1_gain=True)
    decision = (
        "pass"
        if selected["eligible"] and test_passed
        else "fail"
    )
    baseline = spec["initial_a2"][task["dataset"]]
    output_dir = task_output_dir(args.output_dir, task, commit)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": task["dataset"],
        "model": spec["model"],
        "variant": f"{task['family']}_{task['selection']}",
        "evidence_family": task["family"],
        "evidence_selection": task["selection"],
        "seed": int(task["seed"]),
        "git_commit": commit,
        "config": config_relative,
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "parent_manifest": str(parent_path),
        "parent_evidence_commit": parent["git_commit"],
        "evidence_checkpoint": str(evidence_checkpoint),
        "a2_checkpoint": task["a2_checkpoint"],
        "evidence_val_threshold": evidence_threshold,
        "a2_val_threshold": a2_threshold,
        "policies_evaluated": policies_evaluated,
        "validation_eligible_policies": eligible_policies,
        "selected_policy": selected["policy"],
        "validation_statistics": selected["statistics"],
        "validation_policy_passed": selected["eligible"],
        "validation_required_changed_predictions": selected[
            "required_changed_predictions"
        ],
        "test_statistics": test_stats,
        "test_required_changed_predictions": test_required_changed,
        "qualification_decision": decision,
        "initial_a2_val_selected_test_f1": baseline[
            "val_selected_test_f1"
        ],
        "diagnostic_paired_test_f1": test_stats[
            "routed_same_batch_f1"
        ],
        "delta_vs_initial_a2_val_selected_f1": (
            test_stats["routed_same_batch_f1"]
            - baseline["val_selected_test_f1"]
        ),
        "initial_a2_raw_best_test_f1": baseline["raw_best_test_f1"],
        "raw_best_test_f1": None,
        "raw_best_note": (
            "not_applicable_single_locked_test_evaluation"
        ),
        "result_scope": (
            "paired_same_batch_mechanism_diagnostic_not_headline_result"
        ),
        "val_pairing": {
            key: val[key]
            for key in (
                "requested_samples",
                "paired_samples",
                "paired_retention_rate",
            )
        },
        "test_pairing": {
            key: test[key]
            for key in (
                "requested_samples",
                "paired_samples",
                "paired_retention_rate",
            )
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "phase1c_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "run/tier_phase1c_intervention_spec.json",
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--phase1b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    if not 0 <= args.task_index < len(spec["tasks"]):
        raise IndexError("task-index is outside the spec task list")
    manifest = run_task(spec, spec["tasks"][args.task_index], args)
    print(json.dumps({
        key: manifest[key]
        for key in (
            "dataset",
            "variant",
            "seed",
            "validation_policy_passed",
            "qualification_decision",
            "diagnostic_paired_test_f1",
            "delta_vs_initial_a2_val_selected_f1",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
