#!/usr/bin/env python3
"""Summarize three independent CDVT seeds on representative datasets."""

import argparse
import json
import statistics
from pathlib import Path

from run.cdvt_protocol import (
    FRAUDGT_PAPER_REFERENCE,
    INITIAL_A2,
    MULTI_FRAUDGT_PAPER,
    PE_FRAUDGT_PAPER,
)


DATASETS = ("Small-LI", "Medium-LI", "Large-LI")
SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def result_path(dataset, seed, phase1_root, phase2_root, phase3_root):
    if seed != 42:
        return phase3_root / f"{dataset}_dual_view_seed{seed}" / "manifest.json"
    if dataset in {"Small-LI", "Large-LI"}:
        root = phase1_root
    else:
        root = phase2_root
    return root / f"{dataset}_dual_view_seed42" / "manifest.json"


def validate_manifest(payload, path, dataset, seed):
    expected = {
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": "dual_view",
        "architecture_variant": "dual_view",
        "seed": seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: final CDVT must use lambda_cons=0")
    if seed in {43, 44} and payload.get("phase") != "CDVT_phase3":
        raise ValueError(f"{path}: follow-up seed must be CDVT_phase3")
    snapshot = payload.get("config_snapshot", {})
    train = snapshot.get("train", {})
    val = snapshot.get("val", {})
    cdvt = snapshot.get("cdvt", {})
    protocol = (
        snapshot.get("dataset", {}).get("tier_evidence") is True
        and train.get("sampler") == "link_neighbor"
        and int(train.get("iter_per_epoch", -1)) == 256
        and int(train.get("batch_size", -1)) == 2048
        and int(train.get("eval_period", -1)) == 4
        and int(val.get("iter_per_epoch", -1)) == 256
        and val.get("fixed_target_panel") is False
        and int(payload.get("history_k", cdvt.get("history_k", -1))) == 4
        and payload.get(
            "use_relation_types", cdvt.get("use_relation_types", True)
        ) is True
    )
    if not protocol:
        raise ValueError(f"{path}: manifest violates the frozen protocol")


def load_rows(
    phase1_root, phase2_root, phase3_root, allow_incomplete=False
):
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = result_path(
                dataset, seed, phase1_root, phase2_root, phase3_root)
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            validate_manifest(payload, path, dataset, seed)
            a2 = INITIAL_A2[dataset]
            val_f1 = float(payload["val_selected_test_f1"])
            raw_f1 = float(payload["raw_best_test_f1"])
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "git_commit": str(payload["git_commit"]),
                "config": str(payload["config"]),
                "checkpoint": str(payload["checkpoint"]),
                "val_selected_epoch": int(payload["val_selected_epoch"]),
                "val_selected_test_f1": val_f1,
                "delta_val_selected_vs_pe_fraudgt": (
                    val_f1 - PE_FRAUDGT_PAPER[dataset]),
                "delta_val_selected_vs_multi_fraudgt": (
                    val_f1 - MULTI_FRAUDGT_PAPER[dataset]),
                "delta_val_selected_vs_initial_a2": (
                    val_f1 - a2["val_selected_test_f1"]),
                "raw_best_epoch": int(payload["raw_best_epoch"]),
                "raw_best_test_f1": raw_f1,
                "delta_raw_best_vs_initial_a2": (
                    raw_f1 - a2["raw_best_test_f1"]),
                "manifest": str(path.resolve()),
            })
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "three-seed result set is incomplete: " + ", ".join(missing))
    return rows, missing


def metric_group(rows, metric, baseline):
    values = [float(row[metric]) for row in rows]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "delta_mean": statistics.fmean(values) - float(baseline),
    }


def build_summary(
    phase1_root, phase2_root, phase3_root, allow_incomplete=False
):
    rows, missing = load_rows(
        Path(phase1_root), Path(phase2_root), Path(phase3_root),
        allow_incomplete=allow_incomplete)
    groups = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        a2 = INITIAL_A2[dataset]
        val = metric_group(
            dataset_rows,
            "val_selected_test_f1",
            PE_FRAUDGT_PAPER[dataset],
        )
        if val is not None:
            val["delta_mean_vs_pe_fraudgt"] = val.pop("delta_mean")
            val["delta_mean_vs_multi_fraudgt"] = (
                val["mean"] - MULTI_FRAUDGT_PAPER[dataset])
            val["delta_mean_vs_initial_a2"] = (
                val["mean"] - a2["val_selected_test_f1"])
        raw = metric_group(
            dataset_rows,
            "raw_best_test_f1",
            a2["raw_best_test_f1"],
        )
        if raw is not None:
            raw["delta_mean_vs_initial_a2"] = raw.pop("delta_mean")
        groups.append({
            "dataset": dataset,
            "seeds": [row["seed"] for row in dataset_rows],
            "published_references": {
                "pe_fraudgt": PE_FRAUDGT_PAPER[dataset],
                "multi_fraudgt": MULTI_FRAUDGT_PAPER[dataset],
            },
            "initial_a2": a2,
            "val_selected": val,
            "raw_best": raw,
        })
    return {
        "model": "CDVT dual_view without sampling consistency",
        "sampling_protocol": "dynamic_random",
        "primary_baseline": {
            "name": "PE-FraudGT",
            "source": FRAUDGT_PAPER_REFERENCE,
        },
        "published_strong_reference": {
            "name": "Multi-FraudGT",
            "source": FRAUDGT_PAPER_REFERENCE,
        },
        "required_seeds": list(SEEDS),
        "rows": rows,
        "datasets": groups,
        "missing_manifests": missing,
        "complete": not missing and len(rows) == len(DATASETS) * len(SEEDS),
    }


def markdown(summary):
    lines = [
        "# CDVT Representative-Dataset Three-Seed Summary",
        "",
        "All entries use independent seeds 42, 43, and 44 under the "
        "`dynamic_random` protocol.",
        "",
        "| Dataset | Seed | Val-selected test F1 | Delta vs PE | "
        "Delta vs Multi | Delta vs A2 | Raw-best test F1 | Delta vs A2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['seed']} "
            f"| {row['val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected_vs_pe_fraudgt']:+.5f} "
            f"| {row['delta_val_selected_vs_multi_fraudgt']:+.5f} "
            f"| {row['delta_val_selected_vs_initial_a2']:+.5f} "
            f"| {row['raw_best_test_f1']:.5f} "
            f"| {row['delta_raw_best_vs_initial_a2']:+.5f} |")
    lines.extend([
        "",
        "| Dataset | Val-selected mean +/- std | Delta vs PE | "
        "Delta vs Multi | Delta vs A2 | Raw-best mean +/- std | "
        "Delta vs A2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in summary["datasets"]:
        val = group["val_selected"]
        raw = group["raw_best"]
        if val is None or raw is None:
            continue
        val_std = "TBD" if val["sample_std"] is None else (
            f"{val['sample_std']:.5f}")
        raw_std = "TBD" if raw["sample_std"] is None else (
            f"{raw['sample_std']:.5f}")
        lines.append(
            f"| {group['dataset']} "
            f"| {val['mean']:.5f} +/- {val_std} "
            f"| {val['delta_mean_vs_pe_fraudgt']:+.5f} "
            f"| {val['delta_mean_vs_multi_fraudgt']:+.5f} "
            f"| {val['delta_mean_vs_initial_a2']:+.5f} "
            f"| {raw['mean']:.5f} +/- {raw_std} "
            f"| {raw['delta_mean_vs_initial_a2']:+.5f} |")
    if summary["missing_manifests"]:
        lines.extend([
            "",
            "Incomplete manifests:",
            *[f"- `{path}`" for path in summary["missing_manifests"]],
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(
        args.phase1_root,
        args.phase2_root,
        args.phase3_root,
        allow_incomplete=args.allow_incomplete,
    )
    if args.write:
        args.phase3_root.mkdir(parents=True, exist_ok=True)
        (args.phase3_root / "phase3_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.phase3_root / "phase3_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
