#!/usr/bin/env python3
"""Train-only directional utility feasibility probe for frozen TIER models."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.evidence.tier_model import TransactionEvidenceEncoder
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.tier_phase1_evidence_qualification import (
    TASK,
    audit_protocol,
    best_f1_threshold,
    configure_fraudgt,
    load_a2_model,
    loader_audit,
    seed_process,
    verify_repository,
)
from run.tier_phase1c_intervention_diagnostic import (
    intervention_statistics,
    load_parent_manifest,
    paired_features,
    passes_gate,
)


class DirectionalUtilityProbe(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def aggregate_unique_rows(rows):
    edge_ids = rows["target_edge_ids"].long()
    unique_ids, inverse = torch.unique(
        edge_ids, sorted=True, return_inverse=True)
    counts = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    counts.scatter_add_(0, inverse, torch.ones_like(edge_ids).float())
    output = {"target_edge_ids": unique_ids}
    label_sums = torch.zeros(unique_ids.numel(), dtype=torch.long)
    label_sums.scatter_add_(0, inverse, rows["labels"].long())
    if not torch.equal(
        label_sums, (label_sums > 0).long() * counts.long()
    ):
        raise AssertionError("one edge ID has inconsistent labels")
    output["labels"] = (label_sums > 0).long()
    for key in ("evidence_scores", "a2_scores", "support_count"):
        sums = torch.zeros(unique_ids.numel(), dtype=torch.float32)
        sums.scatter_add_(0, inverse, rows[key].float())
        output[key] = sums / counts
    return output


def router_features(rows, a2_threshold, evidence_threshold):
    a2 = rows["a2_scores"].float()
    evidence = rows["evidence_scores"].float()
    a2_margin = a2 - float(a2_threshold)
    evidence_margin = evidence - float(evidence_threshold)
    support = torch.log1p(rows["support_count"].float()) / 4.0
    return torch.stack(
        (
            a2,
            evidence,
            a2_margin.abs(),
            evidence_margin.abs(),
            evidence - a2,
            evidence * a2,
            support,
        ),
        dim=-1,
    )


def direction_candidates(rows, a2_threshold, evidence_threshold, direction):
    a2 = rows["a2_scores"] >= float(a2_threshold)
    evidence = rows["evidence_scores"] >= float(evidence_threshold)
    if direction == "add":
        return (~a2) & evidence
    if direction == "remove":
        return a2 & (~evidence)
    raise ValueError(f"unknown direction: {direction}")


def utility_targets(rows, candidate, evidence_threshold):
    evidence = rows["evidence_scores"] >= float(evidence_threshold)
    labels = rows["labels"].bool()
    return (evidence[candidate] == labels[candidate]).float()


def deterministic_calibration_mask(edge_ids, seed, modulus, bucket):
    hashed = (
        edge_ids.long() * 1103515247 + int(seed) * 12347
    ).remainder(int(modulus))
    return hashed == int(bucket)


def balanced_bce(logits, targets):
    positives = float(targets.sum())
    negatives = float(targets.numel() - targets.sum())
    pos_weight = (
        torch.tensor(
            negatives / max(positives, 1.0),
            device=logits.device,
        )
        if positives
        else torch.tensor(1.0, device=logits.device)
    )
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight)


def fit_direction_probe(
    rows,
    direction,
    a2_threshold,
    evidence_threshold,
    spec,
    seed,
    device,
):
    candidate = direction_candidates(
        rows, a2_threshold, evidence_threshold, direction)
    features = router_features(
        rows, a2_threshold, evidence_threshold)[candidate]
    targets = utility_targets(rows, candidate, evidence_threshold)
    edge_ids = rows["target_edge_ids"][candidate]
    calibration = deterministic_calibration_mask(
        edge_ids,
        seed,
        spec["calibration_hash_modulus"],
        spec["calibration_hash_bucket"],
    )
    fit = ~calibration
    diagnostics = {
        "direction": direction,
        "candidates": int(candidate.sum()),
        "fit_candidates": int(fit.sum()),
        "calibration_candidates": int(calibration.sum()),
        "fit_positive_utility": int(targets[fit].sum()),
        "calibration_positive_utility": int(targets[calibration].sum()),
        "trained": False,
        "best_calibration_loss": None,
        "epochs_completed": 0,
    }
    if (
        int(fit.sum()) < 2
        or torch.unique(targets[fit]).numel() < 2
    ):
        return None, diagnostics

    model = DirectionalUtilityProbe(
        features.size(1), int(spec["router_hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["router_learning_rate"]),
        weight_decay=float(spec["router_weight_decay"]),
    )
    fit_x = features[fit].to(device)
    fit_y = targets[fit].to(device)
    cal_x = features[calibration].to(device)
    cal_y = targets[calibration].to(device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(int(spec["router_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_bce(model(fit_x), fit_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if cal_y.numel():
                cal_loss = float(balanced_bce(model(cal_x), cal_y))
            else:
                cal_loss = float(loss.detach())
        if cal_loss < best_loss - 1e-5:
            best_loss = cal_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        diagnostics["epochs_completed"] = epoch + 1
        if stale >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    diagnostics["trained"] = True
    diagnostics["best_calibration_loss"] = best_loss
    return model, diagnostics


@torch.no_grad()
def direction_scores(
    model,
    rows,
    direction,
    a2_threshold,
    evidence_threshold,
    device,
):
    candidate = direction_candidates(
        rows, a2_threshold, evidence_threshold, direction)
    scores = torch.zeros(rows["labels"].numel(), dtype=torch.float32)
    if model is None:
        return scores, torch.zeros_like(candidate)
    if candidate.any():
        features = router_features(
            rows, a2_threshold, evidence_threshold)
        model.eval()
        scores[candidate] = torch.sigmoid(
            model(features[candidate].to(device))
        ).cpu()
    return scores, candidate


def threshold_values(scores, candidates, quantiles):
    values = scores[candidates]
    if not values.numel():
        return [1.1]
    return sorted({
        float(torch.quantile(values, float(quantile)))
        for quantile in quantiles
    })


def intervention_from_scores(
    add_scores,
    add_candidates,
    remove_scores,
    remove_candidates,
    add_threshold,
    remove_threshold,
):
    return (
        (add_candidates & (add_scores >= float(add_threshold)))
        | (
            remove_candidates
            & (remove_scores >= float(remove_threshold))
        )
    )


def select_calibration_policy(
    rows,
    add_scores,
    add_candidates,
    remove_scores,
    remove_candidates,
    a2_threshold,
    evidence_threshold,
    spec,
):
    add_thresholds = threshold_values(
        add_scores, add_candidates, spec["threshold_quantiles"])
    remove_thresholds = threshold_values(
        remove_scores, remove_candidates, spec["threshold_quantiles"])
    evaluated = []
    for add_threshold in add_thresholds:
        for remove_threshold in remove_thresholds:
            intervention = intervention_from_scores(
                add_scores,
                add_candidates,
                remove_scores,
                remove_candidates,
                add_threshold,
                remove_threshold,
            )
            stats = intervention_statistics(
                rows,
                intervention,
                a2_threshold,
                evidence_threshold,
            )
            passed, required = passes_gate(
                stats,
                spec["calibration_gate"],
                require_f1_gain=True,
            )
            evaluated.append({
                "add_threshold": add_threshold,
                "remove_threshold": remove_threshold,
                "statistics": stats,
                "eligible": passed,
                "required_changed_predictions": required,
            })
    eligible = [row for row in evaluated if row["eligible"]]
    pool = eligible or evaluated

    def rank(row):
        stats = row["statistics"]
        ratio = stats["corrected_to_broken_ratio"]
        ratio_rank = (
            ratio
            if ratio is not None
            else (
                float("inf")
                if stats["corrected_without_breaks"]
                else -1.0
            )
        )
        return (
            stats["delta_paired_f1"],
            stats["corrected_minus_broken"],
            ratio_rank,
            stats["changed_predictions"],
        )

    return max(pool, key=rank), len(evaluated), len(eligible)


def split_rows(rows, mask):
    return {
        key: value[mask]
        for key, value in rows.items()
        if isinstance(value, torch.Tensor)
    }


def evaluate_locked_policy(
    rows,
    add_model,
    remove_model,
    policy,
    a2_threshold,
    evidence_threshold,
    gate,
    device,
):
    add_scores, add_candidates = direction_scores(
        add_model,
        rows,
        "add",
        a2_threshold,
        evidence_threshold,
        device,
    )
    remove_scores, remove_candidates = direction_scores(
        remove_model,
        rows,
        "remove",
        a2_threshold,
        evidence_threshold,
        device,
    )
    intervention = intervention_from_scores(
        add_scores,
        add_candidates,
        remove_scores,
        remove_candidates,
        policy["add_threshold"],
        policy["remove_threshold"],
    )
    stats = intervention_statistics(
        rows, intervention, a2_threshold, evidence_threshold)
    passed, required = passes_gate(
        stats, gate, require_f1_gain=True)
    stats["add_interventions"] = int(
        (intervention & add_candidates).sum())
    stats["remove_interventions"] = int(
        (intervention & remove_candidates).sum())
    return stats, passed, required


def task_output_dir(output_root, task, commit):
    return output_root / (
        f"{task['dataset']}_{task['family']}_{task['selection']}_"
        f"seed{task['seed']}_{commit}"
    )


def run_task(spec, task, args):
    commit = verify_repository(spec)
    output_dir = task_output_dir(args.output_dir, task, commit)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_relative = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    parent_path, parent, evidence_checkpoint = load_parent_manifest(
        args.phase1b_root, task, spec["evidence_commit"])

    seed = int(task["seed"])
    seed_process(seed)
    configure_fraudgt(config_path, seed, args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)
    store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=store.edge_index,
        timestamps=store.timestamps,
        raw_edge_attr=store.raw_edge_attr,
    )
    evidence_model = TransactionEvidenceEncoder(
        num_currencies=int(store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(store.raw_edge_attr[:, 3].max()) + 1,
        family=task["family"],
        hidden_dim=int(spec["hidden_dim"]),
        num_heads=int(spec["num_heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    saved = torch.load(evidence_checkpoint, map_location=device)
    evidence_model.load_state_dict(saved["model_state"], strict=True)
    a2_model = load_a2_model(
        dataset, task["a2_checkpoint"], device)

    started = time.monotonic()
    train_sampled = paired_features(
        evidence_model,
        a2_model,
        train_loader,
        index,
        spec,
        task,
        device,
        "train",
    )
    train = aggregate_unique_rows(train_sampled)
    evidence_threshold, _ = best_f1_threshold(
        train["labels"], train["evidence_scores"])
    a2_threshold, _ = best_f1_threshold(
        train["labels"], train["a2_scores"])
    add_model, add_fit = fit_direction_probe(
        train,
        "add",
        a2_threshold,
        evidence_threshold,
        spec,
        seed,
        device,
    )
    remove_model, remove_fit = fit_direction_probe(
        train,
        "remove",
        a2_threshold,
        evidence_threshold,
        spec,
        seed,
        device,
    )
    calibration_mask = deterministic_calibration_mask(
        train["target_edge_ids"],
        seed,
        spec["calibration_hash_modulus"],
        spec["calibration_hash_bucket"],
    )
    calibration = split_rows(train, calibration_mask)
    add_scores, add_candidates = direction_scores(
        add_model,
        calibration,
        "add",
        a2_threshold,
        evidence_threshold,
        device,
    )
    remove_scores, remove_candidates = direction_scores(
        remove_model,
        calibration,
        "remove",
        a2_threshold,
        evidence_threshold,
        device,
    )
    policy, policies_evaluated, eligible_policies = (
        select_calibration_policy(
            calibration,
            add_scores,
            add_candidates,
            remove_scores,
            remove_candidates,
            a2_threshold,
            evidence_threshold,
            spec,
        )
    )

    val = paired_features(
        evidence_model,
        a2_model,
        val_loader,
        index,
        spec,
        task,
        device,
        "val",
    )
    val_stats, val_passed, val_required = evaluate_locked_policy(
        val,
        add_model,
        remove_model,
        policy,
        a2_threshold,
        evidence_threshold,
        spec["evaluation_gate"],
        device,
    )
    test = paired_features(
        evidence_model,
        a2_model,
        test_loader,
        index,
        spec,
        task,
        device,
        "test",
    )
    test_stats, test_passed, test_required = evaluate_locked_policy(
        test,
        add_model,
        remove_model,
        policy,
        a2_threshold,
        evidence_threshold,
        spec["evaluation_gate"],
        device,
    )
    decision = (
        "pass"
        if policy["eligible"] and val_passed and test_passed
        else "fail"
    )
    checkpoint_path = output_dir / "utility_probe.ckpt"
    torch.save(
        {
            "git_commit": commit,
            "sampling_protocol": "dynamic_random",
            "feature_names": [
                "a2_score",
                "evidence_score",
                "a2_abs_margin",
                "evidence_abs_margin",
                "score_difference",
                "score_interaction",
                "log_support_count",
            ],
            "add_state": (
                add_model.state_dict() if add_model is not None else None
            ),
            "remove_state": (
                remove_model.state_dict()
                if remove_model is not None
                else None
            ),
            "a2_threshold": a2_threshold,
            "evidence_threshold": evidence_threshold,
            "add_threshold": policy["add_threshold"],
            "remove_threshold": policy["remove_threshold"],
        },
        checkpoint_path,
    )
    baseline = spec["initial_a2"][task["dataset"]]
    manifest = {
        "dataset": task["dataset"],
        "model": spec["model"],
        "variant": f"{task['family']}_{task['selection']}",
        "evidence_family": task["family"],
        "evidence_selection": task["selection"],
        "seed": seed,
        "git_commit": commit,
        "config": config_relative,
        "checkpoint": str(checkpoint_path),
        "evidence_checkpoint": str(evidence_checkpoint),
        "a2_checkpoint": task["a2_checkpoint"],
        "parent_manifest": str(parent_path),
        "parent_evidence_commit": parent["git_commit"],
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "evidence_val_threshold_from_train": evidence_threshold,
        "a2_val_threshold_from_train": a2_threshold,
        "train_pairing": {
            key: train_sampled[key]
            for key in (
                "requested_samples",
                "paired_samples",
                "paired_retention_rate",
            )
        },
        "train_unique_edges": int(train["labels"].numel()),
        "add_probe_fit": add_fit,
        "remove_probe_fit": remove_fit,
        "policies_evaluated": policies_evaluated,
        "calibration_eligible_policies": eligible_policies,
        "selected_policy": {
            "add_threshold": policy["add_threshold"],
            "remove_threshold": policy["remove_threshold"],
        },
        "calibration_statistics": policy["statistics"],
        "calibration_policy_passed": policy["eligible"],
        "calibration_required_changed_predictions": policy[
            "required_changed_predictions"
        ],
        "validation_statistics": val_stats,
        "validation_policy_passed": val_passed,
        "validation_required_changed_predictions": val_required,
        "test_statistics": test_stats,
        "test_policy_passed": test_passed,
        "test_required_changed_predictions": test_required,
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
            "optimistic_in_sample_teacher_feasibility_probe_"
            "not_headline_result"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "phase2a_p0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "run/tier_phase2a_p0_utility_spec.json",
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
            "calibration_policy_passed",
            "validation_policy_passed",
            "test_policy_passed",
            "qualification_decision",
            "diagnostic_paired_test_f1",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
