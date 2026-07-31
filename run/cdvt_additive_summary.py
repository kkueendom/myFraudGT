#!/usr/bin/env python3
"""Summarize the additive-fusion control against frozen CDVT."""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.cdvt_ablation_summary import (
    DATASETS,
    compact_result,
    final_path,
    validate_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--additive-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def load(path, missing, allow_incomplete):
    if not path.is_file():
        missing.append(str(path))
        if allow_incomplete:
            return None
        raise FileNotFoundError(f"missing experiment manifest: {path}")
    return json.loads(path.read_text())


def build_summary(
    phase1_root, phase2_root, additive_root, allow_incomplete=False
):
    phase1_root = Path(phase1_root)
    phase2_root = Path(phase2_root)
    additive_root = Path(additive_root)
    missing = []
    rows = []
    for dataset in DATASETS:
        full_manifest = final_path(dataset, phase1_root, phase2_root)
        additive_manifest = (
            additive_root
            / f"{dataset}_causal_event_add_seed42"
            / "manifest.json"
        )
        full = load(full_manifest, missing, allow_incomplete)
        additive = load(additive_manifest, missing, allow_incomplete)
        if full is None or additive is None:
            continue
        validate_manifest(
            full, full_manifest, dataset, "dual_view", "dual_view")
        validate_manifest(
            additive,
            additive_manifest,
            dataset,
            "causal_event_add",
            "additive_view",
            require_ablation_phase=True,
        )
        full_result = compact_result(full, full_manifest)
        additive_result = compact_result(additive, additive_manifest)
        rows.append({
            "dataset": dataset,
            "additive": additive_result,
            "cross_attention": full_result,
            "delta_cross_attention_minus_additive": (
                full_result["val_selected_test_f1"]
                - additive_result["val_selected_test_f1"]
            ),
        })
    deltas = [
        row["delta_cross_attention_minus_additive"] for row in rows
    ]
    return {
        "sampling_protocol": "dynamic_random",
        "selection_metric": "validation F1",
        "rows": rows,
        "mean_delta_cross_attention_minus_additive": (
            statistics.fmean(deltas) if deltas else None
        ),
        "cross_attention_win_count": sum(delta > 0 for delta in deltas),
        "missing_manifests": sorted(set(missing)),
        "complete": not missing,
    }


def markdown(summary):
    lines = [
        "# CDVT Additive-Fusion Control",
        "",
        "| Dataset | Additive F1 | Cross-attention F1 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['additive']['val_selected_test_f1']:.5f} "
            f"| {row['cross_attention']['val_selected_test_f1']:.5f} "
            f"| {row['delta_cross_attention_minus_additive']:+.5f} |"
        )
    mean_delta = summary["mean_delta_cross_attention_minus_additive"]
    if mean_delta is not None:
        lines.extend([
            "",
            f"Cross-attention wins: {summary['cross_attention_win_count']}/3.",
            f"Mean cross-attention minus additive delta: {mean_delta:+.5f}.",
        ])
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
        args.additive_root,
        allow_incomplete=args.allow_incomplete,
    )
    if args.write:
        args.additive_root.mkdir(parents=True, exist_ok=True)
        (args.additive_root / "additive_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.additive_root / "additive_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
