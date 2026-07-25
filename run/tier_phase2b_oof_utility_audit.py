#!/usr/bin/env python3
"""Audit Phase 2b folds and qualify train-only OOF directional utility."""

import argparse
import copy
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from run.tier_phase1_evidence_qualification import (
    average_precision,
    binary_f1,
)
from run.tier_phase2b_crossfit_teachers import fold_ids


DATASETS = ("Small-LI", "Large-LI")
DIRECTIONS = ("add", "remove")
CONDITIONS = ("normal", "shuffled", "off")
SCORE_KEYS = {
    "normal": "evidence_scores",
    "shuffled": "shuffled_scores",
    "off": "off_scores",
}
FEATURE_NAMES = (
    "a2_score",
    "evidence_score",
    "a2_abs_margin",
    "evidence_abs_margin",
    "score_difference",
    "score_interaction",
    "log_support_count",
)
OOF_GATE = {
    "min_candidates": 50,
    "min_auprc_above_prevalence": 0.05,
    "min_normal_above_shuffled_auprc": 0.02,
    "min_corrected_to_broken": 1.5,
    "min_changed": 50,
}


class UtilityProbe(nn.Module):
    def __init__(self, feature_dim=7, hidden_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def git_output(*args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def verify_repository():
    if git_output("status", "--porcelain"):
        raise RuntimeError("refusing to audit from a dirty worktree")
    return git_output("rev-parse", "--short=8", "HEAD")


def audit_loader(rows):
    if [row["split"] for row in rows] != ["train", "val", "test"]:
        raise ValueError("loader audit split order differs")
    for row in rows:
        if row.get("shuffle") is not True:
            raise ValueError("one dynamic loader is not shuffled")
        if (
            row.get("loader_generator") is not None
            or row.get("sampler_generator") is not None
        ):
            raise ValueError("dedicated loader generator detected")


def read_trajectory(path, manifest):
    trajectory = path.parent / "training_trajectory.jsonl"
    if not trajectory.exists():
        raise FileNotFoundError(f"missing trajectory: {trajectory}")
    rows = [
        json.loads(line)
        for line in trajectory.read_text().splitlines()
        if line.strip()
    ]
    expected = (
        int(manifest["a2_epochs"]) + int(manifest["evidence_epochs"])
    )
    if len(rows) != expected:
        raise ValueError(f"{trajectory}: unexpected trajectory length")
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
        if int(row["supervised_samples"]) <= 0:
            raise ValueError(f"{trajectory}: empty supervised epoch")
        if int(row["held_out_outputs_seen"]) <= 0:
            raise ValueError(f"{trajectory}: held-out fold was not observed")
    if len(by_stage["a2"]) != int(manifest["a2_epochs"]):
        raise ValueError(f"{trajectory}: A2 epoch count differs")
    if len(by_stage["evidence"]) != int(manifest["evidence_epochs"]):
        raise ValueError(f"{trajectory}: evidence epoch count differs")
    return {
        stage: {
            "epochs": len(stage_rows),
            "last_loss": float(stage_rows[-1]["loss"]),
            "min_loss": min(float(row["loss"]) for row in stage_rows),
            "mean_supervised_samples": sum(
                int(row["supervised_samples"]) for row in stage_rows
            ) / len(stage_rows),
            "mean_held_out_outputs_seen": sum(
                int(row["held_out_outputs_seen"]) for row in stage_rows
            ) / len(stage_rows),
        }
        for stage, stage_rows in by_stage.items()
    }


def audit_fold(path, expected_input_commit):
    manifest = json.loads(path.read_text())
    if manifest.get("sampling_protocol") != "dynamic_random":
        raise ValueError(f"{path}: protocol differs")
    if not str(manifest.get("git_commit", "")).startswith(
        expected_input_commit
    ):
        raise ValueError(f"{path}: input commit differs")
    if manifest.get("supervision_scope") != (
        "train_non_held_out_target_edges_only"
    ):
        raise ValueError(f"{path}: supervision scope differs")
    if manifest.get("held_out_edges_remain_graph_context") is not True:
        raise ValueError(f"{path}: graph-context contract differs")
    if (
        int(manifest.get("validation_loader_iterations", -1)) != 0
        or int(manifest.get("test_loader_iterations", -1)) != 0
    ):
        raise ValueError(f"{path}: val/test loader was accessed")
    if (
        manifest.get("train_step_cap") is not None
        or manifest.get("collect_step_cap") is not None
    ):
        raise ValueError(f"{path}: formal run used a step cap")
    if (
        int(manifest["a2_epochs"]) != 80
        or int(manifest["evidence_epochs"]) != 60
    ):
        raise ValueError(f"{path}: teacher budget differs")
    audit_loader(manifest["loader_audit"])
    trajectory = read_trajectory(path, manifest)
    oof_path = Path(manifest["oof_scores"])
    if not oof_path.exists():
        local_oof = path.parent / "oof_scores.pt"
        if not local_oof.exists():
            raise FileNotFoundError(f"missing OOF table: {oof_path}")
        oof_path = local_oof
    payload = torch.load(oof_path, map_location="cpu")
    required = {
        "edge_ids",
        "labels",
        "a2_scores",
        "evidence_scores",
        "shuffled_scores",
        "off_scores",
        "support_count",
    }
    if not required.issubset(payload):
        raise ValueError(f"{oof_path}: OOF fields are incomplete")
    count = int(payload["edge_ids"].numel())
    for key in required:
        if int(payload[key].numel()) != count:
            raise ValueError(f"{oof_path}: {key} is not edge aligned")
    if torch.unique(payload["edge_ids"]).numel() != count:
        raise ValueError(f"{oof_path}: duplicate edge IDs within fold")
    fold = int(manifest["fold"])
    num_folds = int(manifest["num_folds"])
    assigned = fold_ids(
        payload["edge_ids"], int(manifest["seed"]), num_folds)
    if not torch.equal(
        assigned, torch.full_like(assigned, fold)
    ):
        raise ValueError(f"{oof_path}: edge assigned to wrong fold")
    if int(payload["fold"]) != fold:
        raise ValueError(f"{oof_path}: stored fold differs")
    if payload.get("sampling_protocol") != "dynamic_random":
        raise ValueError(f"{oof_path}: payload protocol differs")
    return manifest, payload, trajectory


def merge_dataset(folds):
    edge_sets = []
    for _, payload, _ in folds:
        edge_sets.append(set(payload["edge_ids"].tolist()))
    for left in range(len(edge_sets)):
        for right in range(left + 1, len(edge_sets)):
            if edge_sets[left].intersection(edge_sets[right]):
                raise ValueError("OOF edge appears in multiple folds")
    keys = (
        "edge_ids",
        "labels",
        "a2_scores",
        "evidence_scores",
        "shuffled_scores",
        "off_scores",
        "support_count",
    )
    output = {
        key: torch.cat([payload[key] for _, payload, _ in folds])
        for key in keys
    }
    output["a2_thresholds"] = torch.cat([
        torch.full_like(
            payload["a2_scores"], float(payload["a2_threshold"]))
        for _, payload, _ in folds
    ])
    output["evidence_thresholds"] = torch.cat([
        torch.full_like(
            payload["evidence_scores"],
            float(payload["evidence_threshold"]),
        )
        for _, payload, _ in folds
    ])
    order = torch.argsort(output["edge_ids"])
    return {
        key: values[order]
        for key, values in output.items()
    }


def split_assignments(edge_ids, seed):
    return (
        edge_ids.long() * 1103515247 + int(seed) * 12347
    ).remainder(5)


def condition_features(rows, condition):
    evidence = rows[SCORE_KEYS[condition]].float()
    a2 = rows["a2_scores"].float()
    a2_margin = a2 - rows["a2_thresholds"].float()
    evidence_margin = (
        evidence - rows["evidence_thresholds"].float()
    )
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


def direction_candidates(rows, condition, direction):
    a2 = rows["a2_scores"] >= rows["a2_thresholds"]
    evidence = (
        rows[SCORE_KEYS[condition]] >= rows["evidence_thresholds"]
    )
    if direction == "add":
        return (~a2) & evidence
    if direction == "remove":
        return a2 & (~evidence)
    raise ValueError(f"unknown direction: {direction}")


def normal_utility_targets(rows, direction):
    candidate = direction_candidates(rows, "normal", direction)
    evidence = (
        rows["evidence_scores"] >= rows["evidence_thresholds"]
    )
    labels = rows["labels"].bool()
    targets = torch.zeros(labels.numel(), dtype=torch.float32)
    targets[candidate] = (
        evidence[candidate] == labels[candidate]
    ).float()
    return targets


def balanced_bce(logits, targets):
    positives = float(targets.sum())
    negatives = float(targets.numel() - targets.sum())
    pos_weight = torch.tensor(
        negatives / max(positives, 1.0), device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight)


def train_probe(rows, direction, split, device, probe_seed, epochs=200):
    candidates = direction_candidates(rows, "normal", direction)
    targets = normal_utility_targets(rows, direction)
    train_mask = candidates & (split < 3)
    diagnostics = {
        "direction": direction,
        "total_candidates": int(candidates.sum()),
        "train_candidates": int(train_mask.sum()),
        "train_positive_utility": int(targets[train_mask].sum()),
        "train_negative_utility": int(
            train_mask.sum() - targets[train_mask].sum()
        ),
        "trained": False,
        "probe_seed": int(probe_seed),
        "epochs": 0,
        "final_loss": None,
    }
    if (
        int(train_mask.sum()) < 2
        or torch.unique(targets[train_mask]).numel() < 2
    ):
        return None, diagnostics
    torch.manual_seed(int(probe_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(probe_seed))
    model = UtilityProbe().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=1e-4)
    features = condition_features(rows, "normal")[train_mask].to(device)
    labels = targets[train_mask].to(device)
    final_loss = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_bce(model(features), labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    diagnostics.update({
        "trained": True,
        "epochs": epochs,
        "final_loss": final_loss,
    })
    return model, diagnostics


@torch.no_grad()
def probe_scores(model, rows, condition, device):
    if model is None:
        return torch.zeros(rows["labels"].numel())
    model.eval()
    return torch.sigmoid(
        model(condition_features(rows, condition).to(device))
    ).cpu()


def direction_metrics(
    model,
    rows,
    direction,
    split,
    device,
):
    base_candidates = direction_candidates(
        rows, "normal", direction)
    eval_mask = base_candidates & (split == 4)
    targets = normal_utility_targets(rows, direction)
    prevalence = (
        float(targets[eval_mask].mean())
        if eval_mask.any()
        else None
    )
    output = {
        "evaluation_candidates": int(eval_mask.sum()),
        "positive_utility": int(targets[eval_mask].sum()),
        "negative_utility": int(
            eval_mask.sum() - targets[eval_mask].sum()
        ),
        "utility_prevalence": prevalence,
    }
    for condition in CONDITIONS:
        scores = probe_scores(model, rows, condition, device)
        output[f"{condition}_auprc"] = (
            average_precision(
                targets[eval_mask].long(), scores[eval_mask])
            if eval_mask.any()
            and int(targets[eval_mask].sum()) > 0
            else None
        )
    normal = output["normal_auprc"]
    shuffled = output["shuffled_auprc"]
    output["normal_above_prevalence"] = (
        normal - prevalence
        if normal is not None and prevalence is not None
        else None
    )
    output["normal_above_shuffled"] = (
        normal - shuffled
        if normal is not None and shuffled is not None
        else None
    )
    output["qualified"] = (
        output["positive_utility"] > 0
        and output["negative_utility"] > 0
        and output["normal_above_prevalence"] is not None
        and output["normal_above_prevalence"]
        >= OOF_GATE["min_auprc_above_prevalence"]
        and output["normal_above_shuffled"] is not None
        and output["normal_above_shuffled"]
        >= OOF_GATE["min_normal_above_shuffled_auprc"]
    )
    return output


def threshold_values(scores, mask):
    values = scores[mask]
    if not values.numel():
        return [1.1]
    return sorted({
        float(torch.quantile(values, quantile))
        for quantile in torch.linspace(0.0, 1.0, 11)
    })


def intervention_mask(
    rows,
    condition,
    add_scores,
    remove_scores,
    add_threshold,
    remove_threshold,
):
    add = direction_candidates(rows, condition, "add")
    remove = direction_candidates(rows, condition, "remove")
    return (
        add & (add_scores >= float(add_threshold))
    ) | (
        remove & (remove_scores >= float(remove_threshold))
    )


def policy_statistics(rows, condition, intervention, split_mask):
    labels = rows["labels"].bool()
    a2_predictions = (
        rows["a2_scores"] >= rows["a2_thresholds"])
    evidence_predictions = (
        rows[SCORE_KEYS[condition]] >= rows["evidence_thresholds"])
    active = intervention & split_mask
    a2_correct = a2_predictions == labels
    evidence_correct = evidence_predictions == labels
    corrected = active & (~a2_correct) & evidence_correct
    broken = active & a2_correct & (~evidence_correct)
    corrected_count = int(corrected.sum())
    broken_count = int(broken.sum())
    routed = torch.where(
        active, evidence_predictions, a2_predictions)
    baseline_f1 = binary_f1(
        labels[split_mask], a2_predictions[split_mask])
    routed_f1 = binary_f1(
        labels[split_mask], routed[split_mask])
    return {
        "samples": int(split_mask.sum()),
        "a2_errors": int(((~a2_correct) & split_mask).sum()),
        "changed_predictions": int(active.sum()),
        "corrected_predictions": corrected_count,
        "broken_predictions": broken_count,
        "corrected_minus_broken": corrected_count - broken_count,
        "corrected_to_broken_ratio": (
            corrected_count / broken_count
            if broken_count
            else None
        ),
        "corrected_without_breaks": (
            corrected_count > 0 and broken_count == 0
        ),
        "a2_paired_f1": baseline_f1,
        "routed_paired_f1": routed_f1,
        "paired_f1_delta": routed_f1 - baseline_f1,
    }


def calibration_pass(stats):
    ratio = stats["corrected_to_broken_ratio"]
    ratio_ok = stats["corrected_without_breaks"] or (
        ratio is not None
        and ratio >= OOF_GATE["min_corrected_to_broken"]
    )
    return (
        stats["paired_f1_delta"] > 0
        and stats["corrected_minus_broken"] > 0
        and ratio_ok
        and stats["changed_predictions"] >= 20
    )


def evaluation_pass(stats):
    ratio = stats["corrected_to_broken_ratio"]
    ratio_ok = stats["corrected_without_breaks"] or (
        ratio is not None
        and ratio >= OOF_GATE["min_corrected_to_broken"]
    )
    return (
        stats["paired_f1_delta"] > 0
        and stats["corrected_minus_broken"] > 0
        and ratio_ok
        and stats["changed_predictions"] >= OOF_GATE["min_changed"]
    )


def select_policy(rows, probes, split, device):
    scores = {
        direction: probe_scores(
            probes[direction], rows, "normal", device)
        for direction in DIRECTIONS
    }
    calibration = split == 3
    add_thresholds = threshold_values(
        scores["add"],
        calibration & direction_candidates(
            rows, "normal", "add"),
    )
    remove_thresholds = threshold_values(
        scores["remove"],
        calibration & direction_candidates(
            rows, "normal", "remove"),
    )
    evaluated = []
    for add_threshold in add_thresholds:
        for remove_threshold in remove_thresholds:
            active = intervention_mask(
                rows,
                "normal",
                scores["add"],
                scores["remove"],
                add_threshold,
                remove_threshold,
            )
            stats = policy_statistics(
                rows, "normal", active, calibration)
            evaluated.append({
                "add_threshold": add_threshold,
                "remove_threshold": remove_threshold,
                "statistics": stats,
                "eligible": calibration_pass(stats),
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
                math.inf
                if stats["corrected_without_breaks"]
                else -1.0
            )
        )
        return (
            stats["paired_f1_delta"],
            stats["corrected_minus_broken"],
            ratio_rank,
            stats["changed_predictions"],
        )

    selected = max(pool, key=rank)
    return selected, scores, len(evaluated), len(eligible)


def evaluate_policy(
    rows,
    probes,
    policy,
    split,
    device,
    condition,
):
    scores = {
        direction: probe_scores(
            probes[direction], rows, condition, device)
        for direction in DIRECTIONS
    }
    active = intervention_mask(
        rows,
        condition,
        scores["add"],
        scores["remove"],
        policy["add_threshold"],
        policy["remove_threshold"],
    )
    return policy_statistics(
        rows, condition, active, split == 4)


def dataset_qualification(dataset, rows, seed, device, output_dir):
    split = split_assignments(rows["edge_ids"], seed)
    split_counts = {
        "probe_fit": int((split < 3).sum()),
        "threshold_calibration": int((split == 3).sum()),
        "locked_oof_evaluation": int((split == 4).sum()),
    }
    probes = {}
    fits = {}
    direction_rows = {}
    probe_seeds = {
        direction: int(seed) * 100 + direction_index
        for direction_index, direction in enumerate(DIRECTIONS, start=1)
    }
    for direction in DIRECTIONS:
        probes[direction], fits[direction] = train_probe(
            rows,
            direction,
            split,
            device,
            probe_seeds[direction],
        )
        direction_rows[direction] = direction_metrics(
            probes[direction], rows, direction, split, device)
    policy, _, policies_evaluated, eligible_policies = select_policy(
        rows, probes, split, device)
    condition_stats = {
        condition: evaluate_policy(
            rows, probes, policy, split, device, condition)
        for condition in CONDITIONS
    }
    total_candidates = sum(
        int(direction_candidates(
            rows, "normal", direction).sum())
        for direction in DIRECTIONS
    )
    evidence_qualified = any(
        row["qualified"] for row in direction_rows.values())
    normal_policy_passed = evaluation_pass(
        condition_stats["normal"])
    passed = (
        total_candidates >= OOF_GATE["min_candidates"]
        and evidence_qualified
        and bool(policy["eligible"])
        and normal_policy_passed
    )
    checkpoint = output_dir / f"{dataset}_utility_probes.ckpt"
    torch.save({
        "dataset": dataset,
        "feature_names": FEATURE_NAMES,
        "add_state": (
            probes["add"].state_dict()
            if probes["add"] is not None else None
        ),
        "remove_state": (
            probes["remove"].state_dict()
            if probes["remove"] is not None else None
        ),
        "add_threshold": policy["add_threshold"],
        "remove_threshold": policy["remove_threshold"],
        "probe_seeds": probe_seeds,
        "split_contract": "edge_hash_60_fit_20_calibration_20_eval",
    }, checkpoint)
    return {
        "dataset": dataset,
        "oof_edges": int(rows["labels"].numel()),
        "oof_positives": int(rows["labels"].sum()),
        "split_counts": split_counts,
        "total_normal_candidates": total_candidates,
        "probe_fit": fits,
        "direction_metrics": direction_rows,
        "policies_evaluated": policies_evaluated,
        "calibration_eligible_policies": eligible_policies,
        "selected_policy": {
            "add_threshold": policy["add_threshold"],
            "remove_threshold": policy["remove_threshold"],
        },
        "calibration_statistics": policy["statistics"],
        "condition_policy_statistics": condition_stats,
        "normal_policy_passed": normal_policy_passed,
        "evidence_qualified": evidence_qualified,
        "qualification_decision": "pass" if passed else "fail",
        "probe_checkpoint": str(checkpoint),
    }


def format_number(value):
    return "n/a" if value is None else f"{value:.5f}"


def render_markdown(payload):
    lines = [
        "# TIER Phase 2b OOF Utility Qualification Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: EXECUTED OOF tables audited",
        "- Sampling Protocol: `dynamic_random`",
        f"- Teacher Commit: `{payload['expected_input_commit']}`",
        f"- Audit Commit: `{payload['audit_commit']}`",
        "- Validation Loader Iterations: `0`",
        "- Test Loader Iterations: `0`",
        "",
        "## Fold Integrity",
        "",
        "| Dataset | Fold | OOF edges | Positives | Pair retention | "
        "A2 loss min/last | Evidence loss min/last |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["folds"]:
        lines.append(
            f"| {row['dataset']} | {row['fold']} | "
            f"{row['oof_unique_edges']} | {row['oof_positives']} | "
            f"{format_number(row['paired_retention_rate'])} | "
            f"{format_number(row['trajectory']['a2']['min_loss'])}/"
            f"{format_number(row['trajectory']['a2']['last_loss'])} | "
            f"{format_number(row['trajectory']['evidence']['min_loss'])}/"
            f"{format_number(row['trajectory']['evidence']['last_loss'])} |"
        )
    lines.extend([
        "",
        "## Directional Utility",
        "",
        "| Dataset | Direction | Eval candidates | Utility +/- | "
        "Prevalence | Normal AUPRC | Shuffled AUPRC | Off AUPRC | "
        "Normal-prev | Normal-shuffled | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for dataset in DATASETS:
        item = payload["datasets"][dataset]
        for direction in DIRECTIONS:
            row = item["direction_metrics"][direction]
            lines.append(
                f"| {dataset} | {direction} | "
                f"{row['evaluation_candidates']} | "
                f"{row['positive_utility']}/{row['negative_utility']} | "
                f"{format_number(row['utility_prevalence'])} | "
                f"{format_number(row['normal_auprc'])} | "
                f"{format_number(row['shuffled_auprc'])} | "
                f"{format_number(row['off_auprc'])} | "
                f"{format_number(row['normal_above_prevalence'])} | "
                f"{format_number(row['normal_above_shuffled'])} | "
                f"{'pass' if row['qualified'] else 'fail'} |"
            )
    lines.extend([
        "",
        "## Locked OOF Policy",
        "",
        "| Dataset | All candidates | Cal eligible | Eval changed | "
        "Corrected/broken | Ratio | A2 F1 | Routed F1 | Delta | "
        "Evidence gate | Policy gate | Final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for dataset in DATASETS:
        item = payload["datasets"][dataset]
        stats = item["condition_policy_statistics"]["normal"]
        lines.append(
            f"| {dataset} | {item['total_normal_candidates']} | "
            f"{item['calibration_eligible_policies']} | "
            f"{stats['changed_predictions']} | "
            f"{stats['corrected_predictions']}/"
            f"{stats['broken_predictions']} | "
            f"{format_number(stats['corrected_to_broken_ratio'])} | "
            f"{format_number(stats['a2_paired_f1'])} | "
            f"{format_number(stats['routed_paired_f1'])} | "
            f"{format_number(stats['paired_f1_delta'])} | "
            f"{'pass' if item['evidence_qualified'] else 'fail'} | "
            f"{'pass' if item['normal_policy_passed'] else 'fail'} | "
            f"{item['qualification_decision']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        payload["conclusion"],
        "",
        "The normal/shuffled/off AUPRC comparison uses the same normal "
        "direction-candidate population. No validation or test labels were "
        "loaded by this qualification stage.",
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-input-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    audit_commit = verify_repository()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.root.glob("*/phase2b_manifest.json"))
    if len(paths) != 6:
        raise RuntimeError(f"expected 6 manifests, found {len(paths)}")
    audited = [
        audit_fold(path, args.expected_input_commit) for path in paths]
    keys = {
        (manifest["dataset"], int(manifest["fold"]))
        for manifest, _, _ in audited
    }
    expected = {
        (dataset, fold)
        for dataset in DATASETS for fold in range(3)
    }
    if keys != expected:
        raise RuntimeError("fold matrix is incomplete or duplicated")
    grouped = defaultdict(list)
    fold_rows = []
    for manifest, oof, trajectory in audited:
        grouped[manifest["dataset"]].append(
            (manifest, oof, trajectory))
        fold_rows.append({
            "dataset": manifest["dataset"],
            "fold": int(manifest["fold"]),
            "oof_unique_edges": int(manifest["oof_unique_edges"]),
            "oof_positives": int(manifest["oof_positives"]),
            "paired_retention_rate": float(
                manifest["paired_retention_rate"]),
            "trajectory": trajectory,
        })
    device = torch.device(args.device)
    dataset_results = {}
    for dataset in DATASETS:
        folds = sorted(
            grouped[dataset], key=lambda row: int(row[0]["fold"]))
        merged = merge_dataset(folds)
        dataset_results[dataset] = dataset_qualification(
            dataset,
            merged,
            int(folds[0][0]["seed"]),
            device,
            args.output_dir,
        )
    passed = all(
        row["qualification_decision"] == "pass"
        for row in dataset_results.values()
    )
    conclusion = (
        "Both datasets passed the OOF utility gate. Validation-only "
        "directional probe evaluation is authorized; test remains closed."
        if passed
        else (
            "The OOF utility gate did not pass on both datasets. TIER stops: "
            "do not tune router thresholds, train CrossFusion, or open "
            "validation/test for this route. Proceed to the CET-FraudGT "
            "complementary temporal encoder mainline."
        )
    )
    payload = {
        "sampling_protocol": "dynamic_random",
        "expected_input_commit": args.expected_input_commit,
        "audit_commit": audit_commit,
        "split_contract": "edge_hash_60_fit_20_calibration_20_eval",
        "folds": sorted(
            fold_rows, key=lambda row: (row["dataset"], row["fold"])),
        "datasets": dataset_results,
        "cross_scale_pass": passed,
        "conclusion": conclusion,
    }
    json_path = args.output_dir / "oof_utility_results.json"
    md_path = args.output_dir / "oof_utility_results.md"
    manifest_path = args.output_dir / "oof_utility_manifest.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))
    manifest_path.write_text(json.dumps({
        "model": "TIER-OOF-Directional-Utility-Probes",
        "git_commit": audit_commit,
        "input_teacher_commit": args.expected_input_commit,
        "sampling_protocol": "dynamic_random",
        "checkpoint": {
            dataset: row["probe_checkpoint"]
            for dataset, row in dataset_results.items()
        },
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "cross_scale_pass": passed,
        "result_json": str(json_path),
        "result_markdown": str(md_path),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cross_scale_pass": passed,
        "datasets": {
            dataset: {
                "decision": row["qualification_decision"],
                "candidates": row["total_normal_candidates"],
                "normal_delta": row[
                    "condition_policy_statistics"
                ]["normal"]["paired_f1_delta"],
            }
            for dataset, row in dataset_results.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
