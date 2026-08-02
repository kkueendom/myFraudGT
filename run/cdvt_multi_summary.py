#!/usr/bin/env python3
"""Audit and summarize the preregistered matched Multi-CDVT screen."""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ("Small-LI", "Medium-LI", "Large-LI")
VARIANTS = {
    "multi_account_only": "account_only",
    "multi_cdvt": "dual_view",
}
SAMPLING_BAND = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def manifest_path(root, dataset, variant):
    return root / f"{dataset}_{variant}_seed42" / "manifest.json"


def validate_manifest(payload, path, dataset, variant):
    expected = {
        "phase": "CDVT_multi_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": VARIANTS[variant],
        "seed": 42,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: Multi screen must use lambda_cons=0")
    if int(payload.get("history_k", -1)) != 4:
        raise ValueError(f"{path}: Multi screen must use history_k=4")
    if payload.get("use_relation_types") is not True:
        raise ValueError(f"{path}: relation-aware event encoder is required")

    snapshot = payload.get("config_snapshot", {})
    dataset_cfg = snapshot.get("dataset", {})
    train = snapshot.get("train", {})
    val = snapshot.get("val", {})
    cdvt = snapshot.get("cdvt", {})
    protocol_ok = (
        dataset_cfg.get("reverse_mp") is True
        and dataset_cfg.get("add_ports") is True
        and dataset_cfg.get("tier_evidence") is True
        and train.get("add_ego_id") is True
        and train.get("sampler") == "link_neighbor"
        and int(train.get("iter_per_epoch", -1)) == 256
        and int(train.get("batch_size", -1)) == 2048
        and int(val.get("iter_per_epoch", -1)) == 256
        and val.get("fixed_target_panel") is False
        and int(cdvt.get("history_k", -1)) == 4
        and float(cdvt.get("lambda_cons", -1)) == 0.0
    )
    if not protocol_ok:
        raise ValueError(f"{path}: Multi config/protocol audit failed")

    loader_rows = payload.get("loader_audit", [])
    if {row.get("split") for row in loader_rows} != {"train", "val", "test"}:
        raise ValueError(f"{path}: loader audit is incomplete")
    if any(
        row.get("shuffle") is not True or row.get("generator") is not None
        for row in loader_rows
    ):
        raise ValueError(f"{path}: loader audit violates dynamic_random")

    relation_rows = payload.get("multi_dataset_audit", [])
    if {row.get("split") for row in relation_rows} != {"train", "val", "test"}:
        raise ValueError(f"{path}: reverse-relation audit is incomplete")
    if any(
        row.get("forward_edges") != row.get("reverse_edges")
        for row in relation_rows
    ):
        raise ValueError(f"{path}: forward/reverse edge counts differ")

    required = (
        "git_commit", "config", "checkpoint", "val_selected_epoch",
        "val_selected_test_f1", "raw_best_test_f1", "parameter_count",
        "peak_gpu_memory_bytes", "elapsed_seconds",
    )
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"{path}: missing manifest fields: {missing}")


def compact(payload, path):
    return {
        "val_selected_test_f1": float(payload["val_selected_test_f1"]),
        "raw_best_test_f1": float(payload["raw_best_test_f1"]),
        "val_selected_epoch": int(payload["val_selected_epoch"]),
        "raw_best_epoch": int(payload["raw_best_epoch"]),
        "git_commit": str(payload["git_commit"]),
        "config": str(payload["config"]),
        "checkpoint": str(payload["checkpoint"]),
        "parameter_count": int(payload["parameter_count"]),
        "peak_gpu_memory_bytes": int(payload["peak_gpu_memory_bytes"]),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "manifest": str(path.resolve()),
    }


def delta_status(delta):
    if abs(delta) < SAMPLING_BAND:
        return "possible_sampling_variation"
    return "clear_gain" if delta > 0 else "clear_loss"


def build_summary(multi_root, allow_incomplete=False):
    multi_root = Path(multi_root)
    rows = []
    missing = []
    for dataset in DATASETS:
        pair = {}
        for variant in VARIANTS:
            path = manifest_path(multi_root, dataset, variant)
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            validate_manifest(payload, path, dataset, variant)
            pair[variant] = compact(payload, path)
        if len(pair) != len(VARIANTS):
            continue
        val_delta = (
            pair["multi_cdvt"]["val_selected_test_f1"]
            - pair["multi_account_only"]["val_selected_test_f1"]
        )
        raw_delta = (
            pair["multi_cdvt"]["raw_best_test_f1"]
            - pair["multi_account_only"]["raw_best_test_f1"]
        )
        rows.append({
            "dataset": dataset,
            "seed": 42,
            "multi_fraudgt": pair["multi_account_only"],
            "multi_cdvt": pair["multi_cdvt"],
            "delta_val_selected": val_delta,
            "delta_raw_best": raw_delta,
            "val_selected_status": delta_status(val_delta),
        })

    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "Multi screen is incomplete: " + ", ".join(missing))
    deltas = [row["delta_val_selected"] for row in rows]
    complete = not missing and len(rows) == len(DATASETS)
    positive_count = sum(delta > 0 for delta in deltas)
    mean_delta = statistics.fmean(deltas) if deltas else None
    max_delta = max(deltas) if deltas else None
    min_delta = min(deltas) if deltas else None
    criteria = {
        "at_least_two_of_three_positive": (
            complete and positive_count >= 2),
        "mean_delta_gt_0_005": (
            complete and mean_delta is not None and mean_delta > 0.005),
        "at_least_one_delta_ge_0_01": (
            complete and max_delta is not None and max_delta >= 0.01),
        "no_delta_lt_minus_0_02": (
            complete and min_delta is not None and min_delta >= -0.02),
        "protocol_audit_complete": complete,
    }
    return {
        "sampling_protocol": "dynamic_random",
        "selection_metric": "Val-selected Test F1",
        "raw_best_role": "supplementary_only",
        "comparison": "matched Multi-CDVT minus Multi-FraudGT",
        "rows": rows,
        "positive_count": positive_count,
        "negative_count": sum(delta < 0 for delta in deltas),
        "clear_gain_count": sum(delta >= SAMPLING_BAND for delta in deltas),
        "clear_loss_count": sum(delta <= -SAMPLING_BAND for delta in deltas),
        "mean_delta_val_selected": mean_delta,
        "max_delta_val_selected": max_delta,
        "min_delta_val_selected": min_delta,
        "gate_criteria": criteria,
        "advance_to_six_dataset_screen": all(criteria.values()),
        "missing_manifests": sorted(missing),
        "complete": complete,
    }


def markdown(summary):
    lines = [
        "# Multi-CDVT Three-Scale Screen",
        "",
        "Primary metric: Val-selected Test F1. Raw-best is supplementary only.",
        "",
        "| Dataset | Seed | Multi-FraudGT | Multi-CDVT | Delta | Status | Raw delta |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['seed']} "
            f"| {row['multi_fraudgt']['val_selected_test_f1']:.5f} "
            f"| {row['multi_cdvt']['val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected']:+.5f} "
            f"| {row['val_selected_status']} "
            f"| {row['delta_raw_best']:+.5f} |")
    lines.extend(["", "## Preregistered Gate", ""])
    for name, passed in summary["gate_criteria"].items():
        lines.append(f"- `{name}`: {passed}")
    mean_delta = summary["mean_delta_val_selected"]
    if mean_delta is not None:
        lines.extend([
            "",
            f"Mean paired delta: {mean_delta:+.5f}.",
            f"Numeric wins/losses: {summary['positive_count']}/"
            f"{summary['negative_count']}.",
            "Advance to six-dataset screen: "
            f"{summary['advance_to_six_dataset_screen']}.",
        ])
    if summary["missing_manifests"]:
        lines.extend([
            "", "Missing final manifests:",
            *[f"- `{path}`" for path in summary["missing_manifests"]],
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(
        args.multi_root, allow_incomplete=args.allow_incomplete)
    if args.write:
        args.multi_root.mkdir(parents=True, exist_ok=True)
        (args.multi_root / "multi_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.multi_root / "multi_summary.md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
