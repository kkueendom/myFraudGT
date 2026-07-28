#!/usr/bin/env python3
"""Summarize normal-only runtime benchmarks for FraudGT and CDVT."""

import argparse
import json
from pathlib import Path


DATASETS = ("Small-LI", "Medium-LI", "Large-LI")
VARIANTS = ("account_only", "dual_view")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def build_summary(runtime_root, allow_incomplete=False):
    runtime_root = Path(runtime_root)
    rows = []
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            path = (
                runtime_root / f"{dataset}_{variant}_seed42"
                / "benchmark.json")
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            expected = {
                "phase": "CDVT_runtime",
                "sampling_protocol": "dynamic_random",
                "dataset": dataset,
                "variant": variant,
                "architecture_variant": variant,
                "seed": 42,
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(
                        f"{path}: expected {key}={value!r}, "
                        f"found {payload.get(key)!r}")
            if int(payload["steps"]) != 256:
                raise ValueError(f"{path}: runtime pass must use 256 batches")
            rows.append(payload | {"benchmark": str(path.resolve())})
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "runtime benchmark set is incomplete: " + ", ".join(missing))
    comparisons = []
    for dataset in DATASETS:
        account = next((row for row in rows if (
            row["dataset"] == dataset
            and row["variant"] == "account_only")), None)
        cdvt = next((row for row in rows if (
            row["dataset"] == dataset
            and row["variant"] == "dual_view")), None)
        if account is None or cdvt is None:
            continue
        comparisons.append({
            "dataset": dataset,
            "parameter_ratio_cdvt_vs_account": (
                cdvt["parameter_count"] / account["parameter_count"]),
            "memory_ratio_cdvt_vs_account": (
                cdvt["peak_gpu_memory_bytes"]
                / max(account["peak_gpu_memory_bytes"], 1)),
            "latency_ratio_cdvt_vs_account": (
                cdvt["seconds_per_batch"] / account["seconds_per_batch"]),
        })
    return {
        "sampling_protocol": "dynamic_random",
        "benchmark_mode": "normal_only",
        "rows": rows,
        "comparisons": comparisons,
        "missing_benchmarks": missing,
        "complete": not missing,
    }


def markdown(summary):
    lines = [
        "# CDVT Normal-Only Runtime Summary",
        "",
        "| Dataset | Model | Parameters | Peak GPU memory (bytes) | "
        "Seconds / batch | Seconds / target |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} "
            f"| {row['parameter_count']} "
            f"| {row['peak_gpu_memory_bytes']} "
            f"| {row['seconds_per_batch']:.6f} "
            f"| {row['seconds_per_target']:.9f} |")
    lines.extend([
        "",
        "| Dataset | Parameter ratio | Memory ratio | Latency ratio |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['parameter_ratio_cdvt_vs_account']:.3f}x "
            f"| {row['memory_ratio_cdvt_vs_account']:.3f}x "
            f"| {row['latency_ratio_cdvt_vs_account']:.3f}x |")
    if summary["missing_benchmarks"]:
        lines.extend([
            "",
            "Incomplete benchmarks:",
            *[f"- `{path}`" for path in summary["missing_benchmarks"]],
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(
        args.runtime_root, allow_incomplete=args.allow_incomplete)
    if args.write:
        args.runtime_root.mkdir(parents=True, exist_ok=True)
        (args.runtime_root / "runtime_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.runtime_root / "runtime_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
