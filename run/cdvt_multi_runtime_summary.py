#!/usr/bin/env python3
"""Audit and summarize matched Multi-FraudGT runtime benchmarks."""

import argparse
import json
from pathlib import Path


VARIANTS = {
    "multi_account_only": "account_only",
    "multi_cdvt": "dual_view",
}

PHASES = {
    "quick": "CDVT_multi_runtime_quick",
    "formal_256": "CDVT_multi_runtime_formal_256",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--evidence-tier", choices=tuple(PHASES), default="quick")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def validate(payload, path, dataset, variant, evidence_tier):
    expected = {
        "phase": PHASES[evidence_tier],
        "benchmark_mode": "normal_only_end_to_end_inference",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": VARIANTS[variant],
        "account_backbone": "Multi-FraudGT",
        "seed": 42,
        "checkpoint_loaded": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    actual_tier = payload.get("evidence_tier", "quick")
    if actual_tier != evidence_tier:
        raise ValueError(
            f"{path}: expected evidence_tier={evidence_tier!r}, "
            f"found {actual_tier!r}")
    if int(payload.get("repeats", 0)) < 2:
        raise ValueError(f"{path}: at least two repeats are required")
    if int(payload.get("batches_per_repeat", 0)) < 1:
        raise ValueError(f"{path}: invalid batch budget")
    if len(payload.get("repeat_rows", [])) != int(payload["repeats"]):
        raise ValueError(f"{path}: repeat rows are incomplete")
    total = int(payload.get("total_measured_batches", 0))
    expected_total = (
        int(payload["repeats"]) * int(payload["batches_per_repeat"]))
    if total != expected_total:
        raise ValueError(f"{path}: measured batch count is inconsistent")
    if evidence_tier == "formal_256" and total != 256:
        raise ValueError(f"{path}: formal runtime must use 256 batches")
    if evidence_tier == "formal_256":
        if payload.get("loader_restart_policy") != (
                "continue_dynamic_random_after_loader_exhaustion"):
            raise ValueError(f"{path}: formal loader restart policy is missing")
        if int(payload.get("total_loader_restarts", -1)) < 0:
            raise ValueError(f"{path}: formal loader restart count is invalid")
    if float(payload.get("mean_seconds_per_batch", 0)) <= 0:
        raise ValueError(f"{path}: latency must be positive")
    if float(payload.get("mean_targets_per_second", 0)) <= 0:
        raise ValueError(f"{path}: throughput must be positive")
    if int(payload.get("parameter_count", 0)) <= 0:
        raise ValueError(f"{path}: parameter count must be positive")
    if int(payload.get("peak_gpu_memory_bytes", 0)) <= 0:
        raise ValueError(f"{path}: peak GPU memory must be positive")
    loader_rows = payload.get("loader_audit", [])
    if {row.get("split") for row in loader_rows} != {
            "train", "val", "test"}:
        raise ValueError(f"{path}: loader audit is incomplete")
    if any(
            row.get("shuffle") is not True
            or row.get("generator") is not None
            for row in loader_rows):
        raise ValueError(f"{path}: dynamic-random loader audit failed")


def build_summary(runtime_root, datasets, evidence_tier="quick"):
    runtime_root = Path(runtime_root)
    rows = []
    for dataset in datasets:
        for variant in VARIANTS:
            path = (
                runtime_root / f"{dataset}_{variant}_seed42"
                / "benchmark.json")
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            validate(payload, path, dataset, variant, evidence_tier)
            payload["benchmark"] = str(path.resolve())
            rows.append(payload)

    comparisons = []
    for dataset in datasets:
        account = next(row for row in rows if (
            row["dataset"] == dataset
            and row["variant"] == "multi_account_only"))
        cdvt = next(row for row in rows if (
            row["dataset"] == dataset
            and row["variant"] == "multi_cdvt"))
        if account["device_name"] != cdvt["device_name"]:
            raise ValueError(f"{dataset}: benchmark devices differ")
        if account["batches_per_repeat"] != cdvt["batches_per_repeat"]:
            raise ValueError(f"{dataset}: batch budgets differ")
        if account["repeats"] != cdvt["repeats"]:
            raise ValueError(f"{dataset}: repeat counts differ")
        latency_ratio = (
            cdvt["mean_seconds_per_batch"]
            / account["mean_seconds_per_batch"])
        comparisons.append({
            "dataset": dataset,
            "parameter_ratio_cdvt_vs_multi_fraudgt": (
                cdvt["parameter_count"] / account["parameter_count"]),
            "memory_ratio_cdvt_vs_multi_fraudgt": (
                cdvt["peak_gpu_memory_bytes"]
                / account["peak_gpu_memory_bytes"]),
            "latency_ratio_cdvt_vs_multi_fraudgt": latency_ratio,
            "latency_overhead_percent": (latency_ratio - 1.0) * 100.0,
            "throughput_ratio_cdvt_vs_multi_fraudgt": (
                cdvt["mean_targets_per_second"]
                / account["mean_targets_per_second"]),
        })
    return {
        "phase": f"{PHASES[evidence_tier]}_summary",
        "evidence_tier": evidence_tier,
        "benchmark_mode": "normal_only_end_to_end_inference",
        "sampling_protocol": "dynamic_random",
        "datasets": list(datasets),
        "rows": rows,
        "comparisons": comparisons,
        "complete": True,
    }


def markdown(summary):
    lines = [
        "# Multi-CDVT "
        f"{'Formal 256-Batch' if summary['evidence_tier'] == 'formal_256' else 'Quick'} "
        "Runtime Summary",
        "",
        "Freshly initialized weights are used because parameter count, memory "
        "shape, and operator runtime do not depend on trained values. Each row "
        "is normal-only end-to-end inference under dynamic-random sampling.",
        "",
        "| Dataset | Model | Parameters | Peak GPU memory (GiB) | "
        "Seconds / batch (mean +/- std) | Targets / second |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} "
            f"| {row['parameter_count']:,} "
            f"| {row['peak_gpu_memory_bytes'] / 2**30:.3f} "
            f"| {row['mean_seconds_per_batch']:.4f} +/- "
            f"{row['std_seconds_per_batch']:.4f} "
            f"| {row['mean_targets_per_second']:.2f} |")
    lines.extend([
        "",
        "| Dataset | Parameter ratio | Memory ratio | Latency ratio | "
        "Latency overhead | Throughput ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['parameter_ratio_cdvt_vs_multi_fraudgt']:.3f}x "
            f"| {row['memory_ratio_cdvt_vs_multi_fraudgt']:.3f}x "
            f"| {row['latency_ratio_cdvt_vs_multi_fraudgt']:.3f}x "
            f"| {row['latency_overhead_percent']:+.1f}% "
            f"| {row['throughput_ratio_cdvt_vs_multi_fraudgt']:.3f}x |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(
        args.runtime_root, args.datasets, args.evidence_tier)
    if args.write:
        (args.runtime_root / "runtime_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.runtime_root / "runtime_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
