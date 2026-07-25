#!/usr/bin/env python3
"""Train-only leave-one-fold-out evidence risk-control feasibility audit."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import torch

from run.tier_phase1_evidence_qualification import binary_f1
from run.tier_phase2b_oof_utility_audit import (
    CONDITIONS,
    DIRECTIONS,
    SCORE_KEYS,
    UtilityProbe,
    balanced_bce,
    condition_features,
    direction_candidates,
    normal_utility_targets,
)


DATASETS = ("Small-LI", "Large-LI")
SUPPORT_MINIMA = (0, 1, 8, 24, 48)
A2_MARGIN_MAXIMA = (1.0, 0.50, 0.25, 0.10, 0.05)
ALIGNMENT_MINIMA = (-1.0, 0.0, 0.01)
RISK_BOUND = 0.40
CALIBRATION_MIN_CHANGED = 5
MIN_CORRECTED_TO_BROKEN = 1.5
MIN_HELDOUT_CHANGED = 50


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


def wilson_upper(successes, trials, z=1.6448536269514722):
    """One-sided Wilson upper bound for a binomial proportion."""
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)


def load_dataset_folds(oof_root, dataset):
    paths = sorted(oof_root.glob(f"{dataset}_fold*/oof_scores.pt"))
    if len(paths) != 3:
        raise ValueError(
            f"{dataset}: expected three OOF score files, found {len(paths)}")
    folds = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        fold = int(payload["fold"])
        if payload.get("sampling_protocol") != "dynamic_random":
            raise ValueError(f"{path}: protocol is not dynamic_random")
        if fold in folds:
            raise ValueError(f"{dataset}: duplicate fold {fold}")
        folds[fold] = payload
    if sorted(folds) != [0, 1, 2]:
        raise ValueError(f"{dataset}: fold IDs are not 0, 1, 2")
    edge_sets = [
        set(folds[fold]["edge_ids"].tolist())
        for fold in sorted(folds)
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            if edge_sets[left].intersection(edge_sets[right]):
                raise ValueError(
                    f"{dataset}: target edge appears in multiple folds")
    return folds, paths


def merge_payloads(payloads):
    keys = (
        "edge_ids",
        "labels",
        "a2_scores",
        "evidence_scores",
        "shuffled_scores",
        "off_scores",
        "support_count",
    )
    rows = {
        key: torch.cat([payload[key] for payload in payloads])
        for key in keys
    }
    rows["a2_thresholds"] = torch.cat([
        torch.full_like(
            payload["a2_scores"], float(payload["a2_threshold"]))
        for payload in payloads
    ])
    rows["evidence_thresholds"] = torch.cat([
        torch.full_like(
            payload["evidence_scores"],
            float(payload["evidence_threshold"]),
        )
        for payload in payloads
    ])
    rows["fold_ids"] = torch.cat([
        torch.full_like(
            payload["labels"].long(), int(payload["fold"]))
        for payload in payloads
    ])
    return rows


def train_probe_on_fold(rows, direction, train_fold, seed, epochs=200):
    candidates = (
        direction_candidates(rows, "normal", direction)
        & (rows["fold_ids"] == int(train_fold))
    )
    targets = normal_utility_targets(rows, direction)
    diagnostics = {
        "direction": direction,
        "train_fold": int(train_fold),
        "train_candidates": int(candidates.sum()),
        "train_positive_utility": int(targets[candidates].sum()),
        "train_negative_utility": int(
            candidates.sum() - targets[candidates].sum()),
        "trained": False,
    }
    if (
        int(candidates.sum()) < 2
        or torch.unique(targets[candidates]).numel() < 2
    ):
        return None, diagnostics
    torch.manual_seed(int(seed))
    model = UtilityProbe()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=1e-4)
    features = condition_features(rows, "normal")[candidates]
    labels = targets[candidates]
    final_loss = None
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_bce(model(features), labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    diagnostics.update({
        "trained": True,
        "epochs": int(epochs),
        "final_loss": final_loss,
    })
    return model, diagnostics


@torch.no_grad()
def utility_scores(model, rows, condition):
    if model is None:
        return torch.zeros(rows["labels"].numel())
    return torch.sigmoid(model(condition_features(rows, condition)))


def directional_alignment(rows, condition, direction):
    condition_score = rows[SCORE_KEYS[condition]].float()
    reference = rows["shuffled_scores"].float()
    if condition == "shuffled":
        reference = rows["evidence_scores"].float()
    if direction == "add":
        return condition_score - reference
    if direction == "remove":
        return reference - condition_score
    raise ValueError(f"unknown direction: {direction}")


def policy_mask(rows, condition, direction, scores, policy):
    if policy is None:
        return torch.zeros(rows["labels"].numel(), dtype=torch.bool)
    candidate = direction_candidates(rows, condition, direction)
    a2_margin = (
        rows["a2_scores"].float()
        - rows["a2_thresholds"].float()
    ).abs()
    return (
        candidate
        & (scores >= float(policy["score_threshold"]))
        & (rows["support_count"].float() >= float(policy["support_min"]))
        & (a2_margin <= float(policy["a2_margin_max"]))
        & (
            directional_alignment(rows, condition, direction)
            >= float(policy["alignment_min"])
        )
    )


def policy_statistics(rows, condition, active, fold):
    fold_mask = rows["fold_ids"] == int(fold)
    active = active & fold_mask
    labels = rows["labels"].bool()
    base = rows["a2_scores"] >= rows["a2_thresholds"]
    evidence = (
        rows[SCORE_KEYS[condition]] >= rows["evidence_thresholds"])
    corrected = active & (base != labels) & (evidence == labels)
    broken = active & (base == labels) & (evidence != labels)
    routed = torch.where(active, evidence, base)
    changed = int(active.sum())
    corrected_count = int(corrected.sum())
    broken_count = int(broken.sum())
    ratio = (
        corrected_count / broken_count
        if broken_count
        else None
    )
    base_f1 = binary_f1(labels[fold_mask], base[fold_mask])
    routed_f1 = binary_f1(labels[fold_mask], routed[fold_mask])
    return {
        "samples": int(fold_mask.sum()),
        "changed": changed,
        "corrected": corrected_count,
        "broken": broken_count,
        "corrected_minus_broken": corrected_count - broken_count,
        "corrected_to_broken_ratio": ratio,
        "corrected_without_breaks": (
            corrected_count > 0 and broken_count == 0),
        "break_rate": broken_count / max(changed, 1),
        "break_rate_wilson_upper_95": wilson_upper(
            broken_count, changed),
        "a2_f1": base_f1,
        "routed_f1": routed_f1,
        "paired_f1_delta": routed_f1 - base_f1,
    }


def threshold_grid(scores, mask):
    values = scores[mask]
    if not values.numel():
        return []
    return sorted({
        float(torch.quantile(values, quantile))
        for quantile in torch.linspace(0.0, 1.0, 11)
    })


def calibration_eligible(stats):
    ratio = stats["corrected_to_broken_ratio"]
    ratio_pass = stats["corrected_without_breaks"] or (
        ratio is not None and ratio >= MIN_CORRECTED_TO_BROKEN
    )
    return (
        stats["changed"] >= CALIBRATION_MIN_CHANGED
        and stats["corrected"] > stats["broken"]
        and ratio_pass
        and stats["paired_f1_delta"] > 0.0
        and stats["break_rate_wilson_upper_95"] <= RISK_BOUND
    )


def select_direction_policy(
    rows,
    direction,
    scores,
    calibration_fold,
):
    candidate = (
        direction_candidates(rows, "normal", direction)
        & (rows["fold_ids"] == int(calibration_fold))
    )
    evaluated = []
    for threshold in threshold_grid(scores, candidate):
        for support_min in SUPPORT_MINIMA:
            for margin_max in A2_MARGIN_MAXIMA:
                for alignment_min in ALIGNMENT_MINIMA:
                    policy = {
                        "score_threshold": threshold,
                        "support_min": support_min,
                        "a2_margin_max": margin_max,
                        "alignment_min": alignment_min,
                    }
                    active = policy_mask(
                        rows, "normal", direction, scores, policy)
                    stats = policy_statistics(
                        rows, "normal", active, calibration_fold)
                    if calibration_eligible(stats):
                        evaluated.append({
                            "policy": policy,
                            "calibration": stats,
                        })
    if not evaluated:
        return None, 0
    selected = max(
        evaluated,
        key=lambda row: (
            row["calibration"]["changed"],
            row["calibration"]["corrected_minus_broken"],
            row["calibration"]["paired_f1_delta"],
        ),
    )
    return selected, len(evaluated)


def audit_rotation(rows, heldout_fold, seed):
    remaining = [
        fold for fold in (0, 1, 2)
        if fold != int(heldout_fold)
    ]
    train_fold, calibration_fold = remaining
    probes = {}
    probe_diagnostics = {}
    selected = {}
    eligible_counts = {}
    for offset, direction in enumerate(DIRECTIONS):
        probe, diagnostics = train_probe_on_fold(
            rows,
            direction,
            train_fold,
            seed + offset,
        )
        probes[direction] = probe
        probe_diagnostics[direction] = diagnostics
        scores = utility_scores(probe, rows, "normal")
        selected[direction], eligible_counts[direction] = (
            select_direction_policy(
                rows,
                direction,
                scores,
                calibration_fold,
            )
        )
    condition_results = {}
    for condition in CONDITIONS:
        active = torch.zeros(
            rows["labels"].numel(), dtype=torch.bool)
        direction_rows = {}
        for direction in DIRECTIONS:
            scores = utility_scores(
                probes[direction], rows, condition)
            selected_policy = selected[direction]
            policy = (
                selected_policy["policy"]
                if selected_policy is not None
                else None
            )
            direction_active = policy_mask(
                rows, condition, direction, scores, policy)
            active |= direction_active
            direction_rows[direction] = {
                "changed": int(
                    (direction_active & (
                        rows["fold_ids"] == heldout_fold
                    )).sum()),
                "policy": policy,
            }
        condition_results[condition] = {
            "directions": direction_rows,
            "statistics": policy_statistics(
                rows, condition, active, heldout_fold),
        }
    return {
        "heldout_fold": int(heldout_fold),
        "train_fold": int(train_fold),
        "calibration_fold": int(calibration_fold),
        "probe_diagnostics": probe_diagnostics,
        "eligible_policy_counts": eligible_counts,
        "selected": selected,
        "conditions": condition_results,
    }


def sum_condition(rotations, condition):
    stats = [
        row["conditions"][condition]["statistics"]
        for row in rotations
    ]
    return {
        "changed": sum(row["changed"] for row in stats),
        "corrected": sum(row["corrected"] for row in stats),
        "broken": sum(row["broken"] for row in stats),
        "corrected_minus_broken": sum(
            row["corrected_minus_broken"] for row in stats),
        "positive_net_folds": sum(
            row["corrected_minus_broken"] > 0 for row in stats),
        "paired_f1_delta_sum": sum(
            row["paired_f1_delta"] for row in stats),
        "paired_f1_delta_mean": sum(
            row["paired_f1_delta"] for row in stats) / len(stats),
    }


def dataset_gate(summary):
    normal = summary["normal"]
    shuffled = summary["shuffled"]
    ratio = (
        normal["corrected"] / normal["broken"]
        if normal["broken"]
        else None
    )
    ratio_pass = (
        normal["corrected"] > 0 and normal["broken"] == 0
    ) or (
        ratio is not None and ratio >= MIN_CORRECTED_TO_BROKEN
    )
    mechanism_pass = (
        normal["corrected_minus_broken"]
        - shuffled["corrected_minus_broken"] >= 10
        or normal["paired_f1_delta_mean"]
        - shuffled["paired_f1_delta_mean"] >= 0.01
    )
    gate = {
        "min_changed": normal["changed"] >= MIN_HELDOUT_CHANGED,
        "positive_net": normal["corrected"] > normal["broken"],
        "ratio": ratio_pass,
        "positive_paired_f1": (
            normal["paired_f1_delta_sum"] > 0.0),
        "normal_above_shuffled": mechanism_pass,
        "positive_net_folds": normal["positive_net_folds"] >= 2,
    }
    gate["passed"] = all(gate.values())
    return gate


def run_dataset(oof_root, dataset, seed):
    folds, paths = load_dataset_folds(oof_root, dataset)
    rows = merge_payloads([folds[index] for index in (0, 1, 2)])
    rotations = [
        audit_rotation(rows, heldout_fold, seed + 100 * heldout_fold)
        for heldout_fold in (0, 1, 2)
    ]
    summary = {
        condition: sum_condition(rotations, condition)
        for condition in CONDITIONS
    }
    gate = dataset_gate(summary)
    return {
        "dataset": dataset,
        "source_files": [str(path) for path in paths],
        "samples": int(rows["labels"].numel()),
        "positives": int(rows["labels"].sum()),
        "rotations": rotations,
        "summary": summary,
        "gate": gate,
        "dependence_note": (
            "OOF artifacts do not contain entity or timestamp identifiers; "
            "Wilson bounds are row-level descriptive bounds and are not "
            "graph-dependence guarantees."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main():
    args = parse_args()
    commit = verify_repository()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        run_dataset(args.oof_root, dataset, args.seed + index * 1000)
        for index, dataset in enumerate(DATASETS)
    ]
    manifest = {
        "experiment": "dynamic_evidence_risk_control_feasibility",
        "git_commit": commit,
        "sampling_protocol": "dynamic_random",
        "data_scope": "train_only_oof",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "oof_root": str(args.oof_root),
        "seed": int(args.seed),
        "risk_bound": RISK_BOUND,
        "datasets": datasets,
        "passed": all(row["gate"]["passed"] for row in datasets),
    }
    output = args.output_dir / "risk_feasibility_manifest.json"
    output.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "commit": commit,
        "passed": manifest["passed"],
        "datasets": {
            row["dataset"]: {
                "normal": row["summary"]["normal"],
                "gate": row["gate"],
            }
            for row in datasets
        },
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
