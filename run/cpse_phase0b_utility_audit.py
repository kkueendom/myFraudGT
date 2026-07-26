#!/usr/bin/env python3
"""Cross-fold CPSE utility qualification with grouped harm control."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.tier_phase1_evidence_qualification import (
    average_precision,
    binary_f1,
)

CONDITIONS = ("normal", "shuffled", "off")
DIRECTIONS = ("add", "remove")
FEATURE_KEYS = {
    "normal": "normal_features",
    "shuffled": "shuffled_features",
    "off": "off_features",
}


class CPSEUtilityProbe(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def git_commit():
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    ).strip():
        raise RuntimeError("refusing to audit from a dirty worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "--short=8"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_dataset(root, dataset):
    manifests = sorted(root.glob(
        f"{dataset}_fold*/cpse_phase0b_manifest.json"))
    if len(manifests) != 3:
        raise ValueError(f"{dataset}: expected three CPSE manifests")
    rows = []
    seen = set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest["validation_loader_iterations"] != 0:
            raise ValueError("validation loader was opened")
        if manifest["test_loader_iterations"] != 0:
            raise ValueError("test loader was opened")
        if manifest["fraud_labels_used_for_cpse_training"]:
            raise ValueError("CPSE training used fraud labels")
        payload = torch.load(
            manifest["cpse_oof_features"],
            map_location="cpu",
            weights_only=False,
        )
        fold = int(payload["fold"])
        if fold in seen:
            raise ValueError("duplicate CPSE fold")
        seen.add(fold)
        payload["fold_ids"] = torch.full(
            (payload["edge_ids"].numel(),), fold, dtype=torch.long)
        rows.append(payload)
    if seen != {0, 1, 2}:
        raise ValueError("CPSE fold set is incomplete")
    edge_sets = [set(row["edge_ids"].tolist()) for row in rows]
    if any(
        edge_sets[left] & edge_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("edge appears in more than one CPSE fold")
    keys = (
        "edge_ids", "labels", "a2_scores", "normal_features",
        "shuffled_features", "off_features", "source_ids",
        "destination_ids", "timestamps", "fold_ids",
    )
    merged = {key: torch.cat([row[key] for row in rows]) for key in keys}
    merged["a2_thresholds"] = torch.cat([
        torch.full_like(row["a2_scores"], float(row["a2_threshold"]))
        for row in rows
    ])
    return merged, manifests


def graph_time_groups(source, destination, timestamps, block_count=32):
    """Time blocks with within-block shared-entity components."""
    count = int(source.numel())
    if not count:
        return torch.empty(0, dtype=torch.long)
    order = torch.argsort(timestamps, stable=True)
    block = torch.empty(count, dtype=torch.long)
    block[order] = (
        torch.arange(count) * int(block_count) // max(count, 1)
    ).clamp_max(int(block_count) - 1)
    groups = torch.empty(count, dtype=torch.long)
    next_group = 0
    for block_id in range(int(block_count)):
        indices = torch.where(block == block_id)[0]
        parent = list(range(indices.numel()))

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        last = {}
        for local, global_index in enumerate(indices.tolist()):
            for entity in (
                int(source[global_index]), int(destination[global_index])
            ):
                if entity in last:
                    union(local, last[entity])
                last[entity] = local
        mapping = {}
        for local, global_index in enumerate(indices.tolist()):
            root = find(local)
            if root not in mapping:
                mapping[root] = next_group
                next_group += 1
            groups[global_index] = mapping[root]
    return groups


def grouped_upper(active, broken, groups, family_size=64, delta=0.05):
    active_groups = torch.unique(groups[active])
    if not active_groups.numel():
        return 1.0
    rates = []
    for group in active_groups.tolist():
        selected = active & (groups == group)
        rates.append(float(broken[selected].float().mean()))
    mean = sum(rates) / len(rates)
    return min(
        1.0,
        mean + math.sqrt(
            math.log(family_size / delta) / (2.0 * len(rates))),
    )


def direction_candidates(rows, direction):
    base = rows["a2_scores"] >= rows["a2_thresholds"]
    return ~base if direction == "add" else base


def utility_targets(rows, direction):
    return (
        rows["labels"].bool()
        if direction == "add"
        else ~rows["labels"].bool()
    ).float()


def condition_features(rows, condition):
    a2 = rows["a2_scores"].float()
    margin = (a2 - rows["a2_thresholds"]).abs()
    return torch.cat(
        (a2[:, None], margin[:, None], rows[FEATURE_KEYS[condition]].float()),
        dim=1,
    )


def balanced_bce(logits, labels):
    positives = float(labels.sum())
    negatives = float(labels.numel() - labels.sum())
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(
            negatives / max(positives, 1.0), device=logits.device),
    )


def train_probe(rows, direction, train_fold, device, seed):
    mask = direction_candidates(rows, direction) & (
        rows["fold_ids"] == train_fold)
    labels = utility_targets(rows, direction)[mask]
    diagnostics = {
        "train_candidates": int(mask.sum()),
        "positive_utility": int(labels.sum()),
        "negative_utility": int(labels.numel() - labels.sum()),
        "trained": False,
    }
    if labels.numel() < 2 or torch.unique(labels).numel() < 2:
        return None, None, None, diagnostics
    features = condition_features(rows, "normal")[mask]
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-5)
    torch.manual_seed(seed)
    model = CPSEUtilityProbe(features.size(1)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-4)
    normalized = ((features - mean) / scale)
    batch_size = 8192
    for _ in range(30):
        permutation = torch.randperm(normalized.size(0))
        for start in range(0, normalized.size(0), batch_size):
            index = permutation[start:start + batch_size]
            logits = model(normalized[index].to(device))
            loss = balanced_bce(logits, labels[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    diagnostics["trained"] = True
    return model.eval(), mean, scale, diagnostics


@torch.no_grad()
def probe_scores(model, mean, scale, rows, condition, device):
    if model is None:
        return torch.zeros(rows["labels"].numel())
    features = (condition_features(rows, condition) - mean) / scale
    outputs = []
    for start in range(0, features.size(0), 16384):
        outputs.append(torch.sigmoid(
            model(features[start:start + 16384].to(device))).cpu())
    return torch.cat(outputs)


def select_policy(rows, direction, scores, calibration_fold, groups):
    candidate = direction_candidates(rows, direction)
    calibration = candidate & (rows["fold_ids"] == calibration_fold)
    values = scores[calibration]
    if not values.numel():
        return None
    thresholds = sorted({
        float(torch.quantile(values, quantile))
        for quantile in torch.linspace(0, 1, 64)
    })
    labels = rows["labels"].bool()
    base = rows["a2_scores"] >= rows["a2_thresholds"]
    evaluated = []
    for threshold in thresholds:
        active = calibration & (scores >= threshold)
        proposal = torch.ones_like(base) if direction == "add" else (
            torch.zeros_like(base))
        corrected = active & (base != labels) & (proposal == labels)
        broken = active & (base == labels) & (proposal != labels)
        upper = grouped_upper(
            active, broken, groups, family_size=len(thresholds))
        if (
            upper <= 0.40
            and int(corrected.sum()) > int(broken.sum())
        ):
            evaluated.append((
                int(active.sum()),
                int(corrected.sum() - broken.sum()),
                threshold,
                upper,
            ))
    if not evaluated:
        return None
    changed, net, threshold, upper = max(evaluated)
    return {
        "threshold": threshold,
        "calibration_changed": changed,
        "calibration_net": net,
        "harm_upper": upper,
    }


def f1_statistics(rows, active, direction_by_row, fold):
    evaluate = rows["fold_ids"] == fold
    labels = rows["labels"].bool()
    base = rows["a2_scores"] >= rows["a2_thresholds"]
    proposal = torch.where(direction_by_row > 0, True, False)
    corrected = active & (base != labels) & (proposal == labels)
    broken = active & (base == labels) & (proposal != labels)
    routed = torch.where(active, proposal, base)
    return {
        "changed": int((active & evaluate).sum()),
        "corrected": int((corrected & evaluate).sum()),
        "broken": int((broken & evaluate).sum()),
        "net": int((corrected & evaluate).sum() - (broken & evaluate).sum()),
        "a2_f1": binary_f1(labels[evaluate], base[evaluate]),
        "routed_f1": binary_f1(labels[evaluate], routed[evaluate]),
        "paired_f1_delta": (
            binary_f1(labels[evaluate], routed[evaluate])
            - binary_f1(labels[evaluate], base[evaluate])
        ),
    }


def audit_rotation(rows, eval_fold, device, seed):
    train_fold = (eval_fold + 1) % 3
    calibration_fold = (eval_fold + 2) % 3
    calibration_mask = rows["fold_ids"] == calibration_fold
    groups = torch.full_like(rows["fold_ids"], -1)
    groups[calibration_mask] = graph_time_groups(
        rows["source_ids"][calibration_mask],
        rows["destination_ids"][calibration_mask],
        rows["timestamps"][calibration_mask],
    )
    probes = {}
    policies = {}
    direction_metrics = {}
    for offset, direction in enumerate(DIRECTIONS):
        model, mean, scale, diagnostic = train_probe(
            rows, direction, train_fold, device, seed + offset)
        scores = {
            condition: probe_scores(
                model, mean, scale, rows, condition, device)
            for condition in CONDITIONS
        }
        policies[direction] = select_policy(
            rows, direction, scores["normal"], calibration_fold, groups)
        evaluation = direction_candidates(
            rows, direction) & (rows["fold_ids"] == eval_fold)
        targets = utility_targets(rows, direction)
        prevalence = (
            float(targets[evaluation].mean()) if evaluation.any() else None)
        metrics = {"diagnostic": diagnostic, "prevalence": prevalence}
        for condition in CONDITIONS:
            metrics[f"{condition}_auprc"] = (
                average_precision(
                    targets[evaluation].long(),
                    scores[condition][evaluation],
                )
                if evaluation.any() and targets[evaluation].sum() > 0
                else None
            )
        direction_metrics[direction] = metrics
        probes[direction] = scores

    condition_results = {}
    for condition in CONDITIONS:
        active = torch.zeros_like(rows["labels"], dtype=torch.bool)
        direction_by_row = torch.zeros_like(rows["labels"], dtype=torch.long)
        for direction in DIRECTIONS:
            policy = policies[direction]
            if policy is None:
                continue
            selected = (
                direction_candidates(rows, direction)
                & (probes[direction][condition] >= policy["threshold"])
            )
            active |= selected
            direction_by_row[selected] = 1 if direction == "add" else -1
        condition_results[condition] = f1_statistics(
            rows, active, direction_by_row, eval_fold)
    return {
        "evaluation_fold": eval_fold,
        "train_fold": train_fold,
        "calibration_fold": calibration_fold,
        "policies": policies,
        "direction_metrics": direction_metrics,
        "conditions": condition_results,
    }


def summarize_dataset(rows, rotations):
    summary = {}
    for condition in CONDITIONS:
        values = [row["conditions"][condition] for row in rotations]
        summary[condition] = {
            "changed": sum(row["changed"] for row in values),
            "corrected": sum(row["corrected"] for row in values),
            "broken": sum(row["broken"] for row in values),
            "net": sum(row["net"] for row in values),
            "positive_net_folds": sum(row["net"] > 0 for row in values),
            "paired_f1_delta_sum": sum(
                row["paired_f1_delta"] for row in values),
            "paired_f1_delta_mean": sum(
                row["paired_f1_delta"] for row in values) / 3,
        }
    metric_rows = [
        metrics
        for rotation in rotations
        for metrics in rotation["direction_metrics"].values()
        if metrics["normal_auprc"] is not None
        and metrics["shuffled_auprc"] is not None
        and metrics["prevalence"] is not None
    ]
    normal_auprc = sum(
        row["normal_auprc"] for row in metric_rows
    ) / max(len(metric_rows), 1)
    shuffled_auprc = sum(
        row["shuffled_auprc"] for row in metric_rows
    ) / max(len(metric_rows), 1)
    prevalence = sum(
        row["prevalence"] for row in metric_rows
    ) / max(len(metric_rows), 1)
    normal = summary["normal"]
    shuffled = summary["shuffled"]
    ratio = (
        normal["corrected"] / normal["broken"]
        if normal["broken"] else float("inf")
    )
    gate = {
        "min_changed": normal["changed"] >= 50,
        "positive_net": normal["corrected"] > normal["broken"],
        "ratio": ratio >= 1.5 and normal["corrected"] > 0,
        "positive_f1": normal["paired_f1_delta_sum"] > 0,
        "normal_above_shuffled": (
            normal["net"] - shuffled["net"] >= 10
            or normal["paired_f1_delta_mean"]
            - shuffled["paired_f1_delta_mean"] >= 0.01
        ),
        "positive_net_folds": normal["positive_net_folds"] >= 2,
        "auprc_above_prevalence": normal_auprc - prevalence >= 0.05,
        "auprc_above_shuffled": normal_auprc - shuffled_auprc >= 0.02,
        "nonempty_gtprc_policy": any(
            policy is not None
            for rotation in rotations
            for policy in rotation["policies"].values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "samples": int(rows["labels"].numel()),
        "positives": int(rows["labels"].sum()),
        "summary": summary,
        "normal_auprc": normal_auprc,
        "shuffled_auprc": shuffled_auprc,
        "utility_prevalence": prevalence,
        "corrected_to_broken_ratio": (
            ratio if math.isfinite(ratio) else None),
        "gate": gate,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main():
    args = parse_args()
    commit = git_commit()
    device = torch.device(args.device)
    datasets = {}
    for offset, dataset in enumerate(("Small-LI", "Large-LI")):
        rows, manifests = load_dataset(args.input_root, dataset)
        rotations = [
            audit_rotation(
                rows, fold, device, args.seed + offset * 1000 + fold * 100)
            for fold in range(3)
        ]
        datasets[dataset] = {
            **summarize_dataset(rows, rotations),
            "rotations": rotations,
            "source_manifests": [str(path) for path in manifests],
        }
    output = {
        "experiment": "CPSE_Phase0B_train_only_OOF_utility",
        "git_commit": commit,
        "sampling_protocol": "dynamic_random",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "datasets": datasets,
        "passed": all(row["gate"]["passed"] for row in datasets.values()),
        "decision": (
            "PROCEED_TO_TWO_SCALE_DYNAMIC_VALIDATION"
            if all(row["gate"]["passed"] for row in datasets.values())
            else "STOP_CPSE"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output_json),
        "decision": output["decision"],
        "gates": {
            dataset: row["gate"] for dataset, row in datasets.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()

