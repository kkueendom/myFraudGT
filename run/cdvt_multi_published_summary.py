#!/usr/bin/env python3
"""Audit the six-dataset Multi-CDVT screen against the paper baseline."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.cdvt_protocol import MULTI_FRAUDGT_PAPER


DATASETS = (
    "Small-LI", "Small-HI", "Medium-LI",
    "Medium-HI", "Large-LI", "Large-HI",
)
SAMPLING_BAND = 0.005
MAX_EPOCHS = 500
EDGE_FF_CHUNK_SIZES = {
    "Large-LI": 65536,
    "Large-HI": 32768,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--large-hi-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def manifest_path(root, dataset, large_hi_root=None):
    if dataset == "Large-HI" and large_hi_root is not None:
        root = Path(large_hi_root)
    return root / f"{dataset}_multi_cdvt_seed42" / "manifest.json"


def require_equal(payload, path, expected):
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")


def require_finite(payload, path, fields):
    for field in fields:
        value = payload.get(field)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{path}: {field} must be finite")


def validate_manifest(payload, path, dataset):
    published = MULTI_FRAUDGT_PAPER[dataset]
    require_equal(payload, path, {
        "phase": "CDVT_multi_published_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": "multi_cdvt",
        "architecture_variant": "dual_view",
        "seed": 42,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
        "early_stopping_enabled": False,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": MAX_EPOCHS,
        "primary_baseline": "published_multi_fraudgt",
    })
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: lambda_cons must be 0")
    if int(payload.get("history_k", -1)) != 4:
        raise ValueError(f"{path}: history_k must be 4")
    if payload.get("use_relation_types") is not True:
        raise ValueError(f"{path}: relation-aware event encoder is required")

    expected_chunk = EDGE_FF_CHUNK_SIZES.get(dataset, 0)
    expected_checkpoint = dataset.startswith("Large-")
    if int(payload.get("edge_ff_chunk_size", -1)) != expected_chunk:
        raise ValueError(f"{path}: unexpected edge FF chunk size")
    if payload.get("edge_ff_checkpoint") is not expected_checkpoint:
        raise ValueError(f"{path}: unexpected edge FF checkpoint setting")

    recorded_published = float(payload.get(
        "published_multi_fraudgt_val_selected_test_f1", float("nan")))
    if not math.isclose(recorded_published, published, abs_tol=1e-12):
        raise ValueError(f"{path}: published baseline does not match Table 2")

    require_finite(payload, path, (
        "val_selected_test_f1", "raw_best_test_f1", "elapsed_seconds",
        "training_seconds", "inference_seconds", "parameter_count",
        "peak_gpu_memory_bytes",
    ))
    selected = float(payload["val_selected_test_f1"])
    recorded_delta = float(payload.get(
        "delta_val_selected_vs_published_multi_fraudgt", float("nan")))
    if not math.isclose(
        recorded_delta, selected - published, abs_tol=1e-12
    ):
        raise ValueError(f"{path}: published-baseline delta is inconsistent")
    for field in ("val_selected_epoch", "raw_best_epoch"):
        epoch = int(payload.get(field, -1))
        if not 0 <= epoch < MAX_EPOCHS:
            raise ValueError(f"{path}: {field} is outside the 500-epoch run")
    if int(payload["parameter_count"]) <= 0:
        raise ValueError(f"{path}: parameter_count must be positive")
    if int(payload["peak_gpu_memory_bytes"]) <= 0:
        raise ValueError(f"{path}: peak GPU memory must be positive")
    if float(payload["training_seconds"]) <= 0:
        raise ValueError(f"{path}: training time must be positive")
    if float(payload["inference_seconds"]) <= 0:
        raise ValueError(f"{path}: inference time must be positive")

    snapshot = payload.get("config_snapshot", {})
    dataset_cfg = snapshot.get("dataset", {})
    train = snapshot.get("train", {})
    val = snapshot.get("val", {})
    model = snapshot.get("model", {})
    cdvt = snapshot.get("cdvt", {})
    optim = snapshot.get("optim", {})
    gt = snapshot.get("gt", {})
    protocol_ok = (
        dataset_cfg.get("name") == dataset
        and dataset_cfg.get("reverse_mp") is True
        and dataset_cfg.get("add_ports") is True
        and dataset_cfg.get("tier_evidence") is True
        and train.get("sampler") == "link_neighbor"
        and train.get("add_ego_id") is True
        and int(train.get("iter_per_epoch", -1)) == 256
        and int(train.get("batch_size", -1)) == 2048
        and int(val.get("iter_per_epoch", -1)) == 256
        and val.get("fixed_target_panel") is False
        and model.get("type") == "CDVTModel"
        and cdvt.get("variant") == "dual_view"
        and int(cdvt.get("history_k", -1)) == 4
        and float(cdvt.get("lambda_cons", -1)) == 0.0
        and cdvt.get("use_relation_types") is True
        and int(optim.get("max_epoch", -1)) == MAX_EPOCHS
        and int(gt.get("edge_ff_chunk_size", -1)) == expected_chunk
        and gt.get("edge_ff_checkpoint") is expected_checkpoint
    )
    if not protocol_ok:
        raise ValueError(f"{path}: frozen Multi-CDVT protocol audit failed")

    loader_rows = payload.get("loader_audit", [])
    if {row.get("split") for row in loader_rows} != {
        "train", "val", "test"
    }:
        raise ValueError(f"{path}: loader audit is incomplete")
    if any(
        row.get("shuffle") is not True or row.get("generator") is not None
        for row in loader_rows
    ):
        raise ValueError(f"{path}: loader audit violates dynamic_random")

    relation_rows = payload.get("multi_dataset_audit", [])
    if {row.get("split") for row in relation_rows} != {
        "train", "val", "test"
    }:
        raise ValueError(f"{path}: reverse-relation audit is incomplete")
    if any(
        row.get("forward_edges") != row.get("reverse_edges")
        for row in relation_rows
    ):
        raise ValueError(f"{path}: forward/reverse edge counts differ")

    for field in ("git_commit", "config", "checkpoint"):
        if not payload.get(field):
            raise ValueError(f"{path}: missing {field}")


def delta_status(delta):
    if abs(delta) < SAMPLING_BAND:
        return "possible_sampling_variation"
    return "clear_gain" if delta > 0 else "clear_loss"


def gate_verdict(complete, positive_count, mean_delta):
    if not complete:
        return "incomplete"
    if positive_count < 3 or mean_delta <= 0:
        return "failed"
    if positive_count >= 4 and mean_delta > SAMPLING_BAND:
        return "clear_pass"
    return "boundary"


def compact(payload, path, dataset):
    published = MULTI_FRAUDGT_PAPER[dataset]
    selected = float(payload["val_selected_test_f1"])
    delta = selected - published
    return {
        "dataset": dataset,
        "seed": 42,
        "published_multi_fraudgt_val_selected_test_f1": published,
        "multi_cdvt_val_selected_test_f1": selected,
        "delta_val_selected": delta,
        "status": delta_status(delta),
        "val_selected_epoch": int(payload["val_selected_epoch"]),
        "raw_best_test_f1": float(payload["raw_best_test_f1"]),
        "raw_best_epoch": int(payload["raw_best_epoch"]),
        "git_commit": str(payload["git_commit"]),
        "config": str(payload["config"]),
        "checkpoint": str(payload["checkpoint"]),
        "parameter_count": int(payload["parameter_count"]),
        "peak_gpu_memory_bytes": int(payload["peak_gpu_memory_bytes"]),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "training_seconds": float(payload["training_seconds"]),
        "inference_seconds": float(payload["inference_seconds"]),
        "manifest": str(path.resolve()),
    }


def build_summary(
    result_root, allow_incomplete=False, large_hi_root=None
):
    result_root = Path(result_root)
    large_hi_root = (
        Path(large_hi_root) if large_hi_root is not None else None)
    rows = []
    missing = []
    for dataset in DATASETS:
        path = manifest_path(result_root, dataset, large_hi_root)
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        validate_manifest(payload, path, dataset)
        rows.append(compact(payload, path, dataset))

    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "Multi-CDVT published screen is incomplete: "
            + ", ".join(missing))
    commits = {row["git_commit"] for row in rows}
    if len(commits) > 1:
        raise ValueError("formal tasks were run from different Git commits")

    complete = not missing and len(rows) == len(DATASETS)
    deltas = [row["delta_val_selected"] for row in rows]
    positive_count = sum(delta > 0 for delta in deltas)
    mean_delta = statistics.fmean(deltas) if deltas else None
    verdict = gate_verdict(complete, positive_count, mean_delta)
    return {
        "sampling_protocol": "dynamic_random",
        "selection_metric": "Val-selected Test F1",
        "raw_best_role": "supplementary_only",
        "comparison": "Multi-CDVT minus published Multi-FraudGT Table 2",
        "result_root": str(result_root.resolve()),
        "large_hi_root": (
            str(large_hi_root.resolve())
            if large_hi_root is not None else None),
        "seed": 42,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_enabled": False,
        "rows": rows,
        "mean_multi_cdvt_val_selected_test_f1": (
            statistics.fmean(
                row["multi_cdvt_val_selected_test_f1"] for row in rows)
            if rows else None
        ),
        "mean_published_multi_fraudgt_val_selected_test_f1": (
            statistics.fmean(
                row["published_multi_fraudgt_val_selected_test_f1"]
                for row in rows)
            if rows else None
        ),
        "mean_delta_val_selected": mean_delta,
        "positive_count": positive_count,
        "negative_count": sum(delta < 0 for delta in deltas),
        "sampling_band_count": sum(
            abs(delta) < SAMPLING_BAND for delta in deltas),
        "gate_criteria": {
            "at_least_four_of_six_positive": (
                complete and positive_count >= 4),
            "mean_delta_positive": (
                complete and mean_delta is not None and mean_delta > 0),
            "mean_delta_gt_0_005": (
                complete and mean_delta is not None
                and mean_delta > SAMPLING_BAND),
            "protocol_audit_complete": complete,
        },
        "gate_verdict": verdict,
        "next_action": {
            "clear_pass": (
                "consider matched Multi-FraudGT reproduction and selected "
                "multi-seed confirmation"),
            "boundary": (
                "report the boundary result; do not auto-tune or expand"),
            "failed": (
                "stop Multi-CDVT expansion and retain PE-CDVT as the mainline"),
            "incomplete": "wait for all six final manifests",
        }[verdict],
        "git_commit": next(iter(commits)) if len(commits) == 1 else None,
        "missing_manifests": sorted(missing),
        "complete": complete,
    }


def markdown(summary):
    lines = [
        "# Multi-CDVT Published-Baseline Screen",
        "",
        "Primary metric: Val-selected Test F1. Raw-best is supplementary only.",
        "",
        "| Dataset | Published Multi-FraudGT | Multi-CDVT | Delta | Status | Val-selected epoch | Raw-best |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['published_multi_fraudgt_val_selected_test_f1']:.5f} "
            f"| {row['multi_cdvt_val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected']:+.5f} "
            f"| {row['status']} "
            f"| {row['val_selected_epoch']} "
            f"| {row['raw_best_test_f1']:.5f} |")
    lines.extend(["", "## Gate", ""])
    for name, passed in summary["gate_criteria"].items():
        lines.append(f"- `{name}`: {passed}")
    if summary["mean_delta_val_selected"] is not None:
        lines.extend([
            "",
            "Mean Multi-CDVT F1: "
            f"{summary['mean_multi_cdvt_val_selected_test_f1']:.5f}.",
            "Mean published Multi-FraudGT F1: "
            f"{summary['mean_published_multi_fraudgt_val_selected_test_f1']:.5f}.",
            f"Mean delta: {summary['mean_delta_val_selected']:+.5f}.",
            f"Wins/losses: {summary['positive_count']}/"
            f"{summary['negative_count']}.",
            f"Verdict: `{summary['gate_verdict']}`.",
            f"Next action: {summary['next_action']}.",
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
        args.result_root,
        allow_incomplete=args.allow_incomplete,
        large_hi_root=args.large_hi_root,
    )
    if args.write:
        output_root = args.output_root or args.result_root
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "published_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (output_root / "published_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
