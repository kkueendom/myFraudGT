#!/usr/bin/env python3
"""Audit the pre-registered three-seed Small-LI stability diagnostic."""

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


DATASET = "Small-LI"
SEEDS = (42, 43, 44)
MAX_EPOCHS = 500
SAMPLING_BAND = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-root", type=Path, required=True)
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def manifest_path(seed42_root, multiseed_root, seed):
    root = seed42_root if seed == 42 else multiseed_root
    return root / f"{DATASET}_multi_cdvt_seed{seed}" / "manifest.json"


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


def validate_manifest(payload, path, seed):
    published = MULTI_FRAUDGT_PAPER[DATASET]
    require_equal(payload, path, {
        "phase": "CDVT_multi_published_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": DATASET,
        "variant": "multi_cdvt",
        "architecture_variant": "dual_view",
        "seed": seed,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
        "early_stopping_enabled": False,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": MAX_EPOCHS,
        "primary_baseline": "published_multi_fraudgt",
        "edge_ff_chunk_size": 0,
        "edge_ff_checkpoint": False,
    })
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: lambda_cons must be 0")
    if int(payload.get("history_k", -1)) != 4:
        raise ValueError(f"{path}: history_k must be 4")
    if payload.get("use_relation_types") is not True:
        raise ValueError(f"{path}: relation-aware encoder is required")

    require_finite(payload, path, (
        "val_selected_test_f1", "raw_best_test_f1", "elapsed_seconds",
        "training_seconds", "inference_seconds", "parameter_count",
        "peak_gpu_memory_bytes",
    ))
    recorded_published = float(payload.get(
        "published_multi_fraudgt_val_selected_test_f1", float("nan")))
    if not math.isclose(recorded_published, published, abs_tol=1e-12):
        raise ValueError(f"{path}: published baseline is inconsistent")
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
    for field in ("parameter_count", "peak_gpu_memory_bytes"):
        if int(payload[field]) <= 0:
            raise ValueError(f"{path}: {field} must be positive")
    for field in ("elapsed_seconds", "training_seconds", "inference_seconds"):
        if float(payload[field]) <= 0:
            raise ValueError(f"{path}: {field} must be positive")

    snapshot = payload.get("config_snapshot", {})
    cdvt = snapshot.get("cdvt", {})
    dataset = snapshot.get("dataset", {})
    gt = snapshot.get("gt", {})
    model = snapshot.get("model", {})
    optim = snapshot.get("optim", {})
    train = snapshot.get("train", {})
    val = snapshot.get("val", {})
    protocol_ok = (
        int(snapshot.get("seed", -1)) == seed
        and dataset.get("name") == DATASET
        and dataset.get("reverse_mp") is True
        and dataset.get("add_ports") is True
        and dataset.get("tier_evidence") is True
        and model.get("type") == "CDVTModel"
        and cdvt.get("variant") == "dual_view"
        and int(cdvt.get("history_k", -1)) == 4
        and float(cdvt.get("lambda_cons", -1)) == 0.0
        and cdvt.get("use_relation_types") is True
        and int(optim.get("max_epoch", -1)) == MAX_EPOCHS
        and train.get("sampler") == "link_neighbor"
        and train.get("add_ego_id") is True
        and int(train.get("iter_per_epoch", -1)) == 256
        and int(train.get("batch_size", -1)) == 2048
        and val.get("sampler") == "link_neighbor"
        and int(val.get("iter_per_epoch", -1)) == 256
        and val.get("fixed_target_panel") is False
        and int(gt.get("edge_ff_chunk_size", -1)) == 0
        and gt.get("edge_ff_checkpoint") is False
    )
    if not protocol_ok:
        raise ValueError(f"{path}: frozen Small-LI protocol audit failed")

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
    for field in ("git_commit", "config", "checkpoint"):
        if not payload.get(field):
            raise ValueError(f"{path}: missing {field}")


def build_summary(seed42_root, multiseed_root):
    rows = []
    published = MULTI_FRAUDGT_PAPER[DATASET]
    for seed in SEEDS:
        path = manifest_path(seed42_root, multiseed_root, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing final manifest: {path}")
        payload = json.loads(path.read_text())
        validate_manifest(payload, path, seed)
        selected = float(payload["val_selected_test_f1"])
        rows.append({
            "seed": seed,
            "val_selected_test_f1": selected,
            "delta_vs_published": selected - published,
            "val_selected_epoch": int(payload["val_selected_epoch"]),
            "raw_best_test_f1": float(payload["raw_best_test_f1"]),
            "raw_best_epoch": int(payload["raw_best_epoch"]),
            "git_commit": str(payload["git_commit"]),
            "config": str(payload["config"]),
            "checkpoint": str(payload["checkpoint"]),
            "manifest": str(path.resolve()),
        })

    values = [row["val_selected_test_f1"] for row in rows]
    seed42 = values[0]
    alternate_mean = statistics.fmean(values[1:])
    spread = max(values) - min(values)
    return {
        "dataset": DATASET,
        "variant": "multi_cdvt",
        "sampling_protocol": "dynamic_random",
        "selection_metric": "Val-selected Test F1",
        "raw_best_role": "supplementary_only",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_enabled": False,
        "seeds": list(SEEDS),
        "rows": rows,
        "published_multi_fraudgt_val_selected_test_f1": published,
        "mean_val_selected_test_f1": statistics.fmean(values),
        "sample_std_val_selected_test_f1": statistics.stdev(values),
        "min_val_selected_test_f1": min(values),
        "max_val_selected_test_f1": max(values),
        "range_val_selected_test_f1": spread,
        "mean_delta_vs_published": statistics.fmean(values) - published,
        "positive_seed_count_vs_published": sum(
            value > published for value in values),
        "sampling_band_seed_count": sum(
            abs(value - published) < SAMPLING_BAND for value in values),
        "seed42_minus_alternate_seed_mean": seed42 - alternate_mean,
        "seed42_is_lowest": seed42 == min(values),
        "observed_range_gt_0_005": spread > SAMPLING_BAND,
        "interpretation_scope": (
            "descriptive three-seed stability diagnostic; not a powered "
            "significance test"),
        "complete": True,
    }


def markdown(summary):
    lines = [
        "# Multi-CDVT Small-LI Seed Stability",
        "",
        "Primary metric: Val-selected Test F1. Raw-best is supplementary only.",
        "",
        "| Seed | Val-selected Test F1 | Delta vs published | Epoch | Raw-best |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['seed']} | {row['val_selected_test_f1']:.5f} "
            f"| {row['delta_vs_published']:+.5f} "
            f"| {row['val_selected_epoch']} "
            f"| {row['raw_best_test_f1']:.5f} |")
    lines.extend([
        "",
        f"Mean: {summary['mean_val_selected_test_f1']:.5f}.",
        f"Sample standard deviation: "
        f"{summary['sample_std_val_selected_test_f1']:.5f}.",
        f"Range: {summary['range_val_selected_test_f1']:.5f}.",
        f"Mean delta vs published Multi-FraudGT: "
        f"{summary['mean_delta_vs_published']:+.5f}.",
        f"Seeds above published baseline: "
        f"{summary['positive_seed_count_vs_published']}/3.",
        f"Seed 42 minus seeds 43/44 mean: "
        f"{summary['seed42_minus_alternate_seed_mean']:+.5f}.",
        "",
        "This is a descriptive three-seed diagnostic, not a powered "
        "significance test.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(args.seed42_root, args.multiseed_root)
    if args.write:
        output_root = args.output_root or args.multiseed_root
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "small_li_multiseed_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (output_root / "small_li_multiseed_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
