#!/usr/bin/env python3
"""Summarize the frozen six-dataset CDVT evaluation."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.cdvt_protocol import (
    FRAUDGT_PAPER_REFERENCE,
    INITIAL_A2,
    MULTI_FRAUDGT_PAPER,
    PE_FRAUDGT_PAPER,
)


DATASETS = (
    "Small-LI", "Small-HI", "Medium-LI",
    "Medium-HI", "Large-LI", "Large-HI",
)
PHASE1_DATASETS = frozenset(("Small-LI", "Large-LI"))
SAMPLING_BAND = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def validate_manifest(payload, path, dataset):
    expected = {
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": "dual_view",
        "architecture_variant": "dual_view",
        "seed": 42,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: frozen CDVT must use lambda_cons=0")

    snapshot = payload.get("config_snapshot", {})
    protocol = (
        snapshot.get("dataset", {}).get("tier_evidence") is True
        and snapshot.get("train", {}).get("sampler") == "link_neighbor"
        and int(snapshot.get("train", {}).get("iter_per_epoch", -1)) == 256
        and int(snapshot.get("train", {}).get("batch_size", -1)) == 2048
        and int(snapshot.get("train", {}).get("eval_period", -1)) == 4
        and int(snapshot.get("val", {}).get("iter_per_epoch", -1)) == 256
        and snapshot.get("val", {}).get("fixed_target_panel") is False
    )
    if not protocol:
        raise ValueError(f"{path}: config snapshot violates frozen protocol")


def result_path(dataset, phase1_root, phase2_root):
    root = phase1_root if dataset in PHASE1_DATASETS else phase2_root
    return root / f"{dataset}_dual_view_seed42" / "manifest.json"


def interpretation(delta):
    if delta >= SAMPLING_BAND:
        return "clear_gain"
    if delta <= -SAMPLING_BAND:
        return "clear_loss"
    return "possible_sampling_variation"


def load_rows(phase1_root, phase2_root, allow_incomplete=False):
    rows = []
    missing = []
    for dataset in DATASETS:
        path = result_path(dataset, phase1_root, phase2_root)
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        validate_manifest(payload, path, dataset)
        a2 = INITIAL_A2[dataset]
        pe_fraudgt = PE_FRAUDGT_PAPER[dataset]
        multi_fraudgt = MULTI_FRAUDGT_PAPER[dataset]
        val_f1 = float(payload["val_selected_test_f1"])
        raw_f1 = float(payload["raw_best_test_f1"])
        val_delta_pe = val_f1 - pe_fraudgt
        val_delta_multi = val_f1 - multi_fraudgt
        val_delta_a2 = val_f1 - float(a2["val_selected_test_f1"])
        raw_delta_a2 = raw_f1 - float(a2["raw_best_test_f1"])
        rows.append({
            "dataset": dataset,
            "seed": int(payload["seed"]),
            "git_commit": str(payload["git_commit"]),
            "config": str(payload["config"]),
            "checkpoint": str(payload["checkpoint"]),
            "val_selected_epoch": int(payload["val_selected_epoch"]),
            "val_selected_test_f1": val_f1,
            "pe_fraudgt_paper_val_selected_test_f1": pe_fraudgt,
            "delta_val_selected_vs_pe_fraudgt": val_delta_pe,
            "pe_fraudgt_interpretation": interpretation(val_delta_pe),
            "multi_fraudgt_paper_val_selected_test_f1": multi_fraudgt,
            "delta_val_selected_vs_multi_fraudgt": val_delta_multi,
            "multi_fraudgt_interpretation": interpretation(val_delta_multi),
            "initial_a2_val_selected_test_f1": float(
                a2["val_selected_test_f1"]),
            "delta_val_selected_vs_initial_a2": val_delta_a2,
            "initial_a2_val_interpretation": interpretation(val_delta_a2),
            "raw_best_epoch": int(payload["raw_best_epoch"]),
            "raw_best_test_f1": raw_f1,
            "initial_a2_raw_best_test_f1": float(
                a2["raw_best_test_f1"]),
            "delta_raw_best_vs_initial_a2": raw_delta_a2,
            "initial_a2_raw_interpretation": interpretation(raw_delta_a2),
            "parameter_count": int(payload["parameter_count"]),
            "peak_gpu_memory_bytes": int(
                payload.get("peak_gpu_memory_bytes", 0)),
            "elapsed_seconds": float(payload["elapsed_seconds"]),
            "sampling_protocol": "dynamic_random",
            "manifest": str(path.resolve()),
        })
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "six-dataset result set is incomplete: " + ", ".join(missing))
    return rows, missing


def metric_summary(rows, metric, delta):
    deltas = [row[delta] for row in rows]
    count = len(deltas)
    return {
        "datasets_complete": count,
        "wins": sum(value > 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "clear_gains": sum(value >= SAMPLING_BAND for value in deltas),
        "possible_sampling_variation": sum(
            abs(value) < SAMPLING_BAND for value in deltas),
        "mean_f1": (
            sum(row[metric] for row in rows) / count if count else None),
        "mean_delta": sum(deltas) / count if count else None,
    }


def build_summary(phase1_root, phase2_root, allow_incomplete=False):
    rows, missing = load_rows(
        phase1_root, phase2_root, allow_incomplete=allow_incomplete)
    val_pe = metric_summary(
        rows,
        "val_selected_test_f1",
        "delta_val_selected_vs_pe_fraudgt",
    )
    val_multi = metric_summary(
        rows,
        "val_selected_test_f1",
        "delta_val_selected_vs_multi_fraudgt",
    )
    val_a2 = metric_summary(
        rows,
        "val_selected_test_f1",
        "delta_val_selected_vs_initial_a2",
    )
    raw_a2 = metric_summary(
        rows,
        "raw_best_test_f1",
        "delta_raw_best_vs_initial_a2",
    )
    gate = (
        len(rows) == len(DATASETS)
        and val_pe["wins"] >= 4
        and val_pe["mean_delta"] > 0
    )
    return {
        "model": "CDVT dual_view without sampling consistency",
        "sampling_protocol": "dynamic_random",
        "selection_metric": "validation F1",
        "sampling_variation_threshold": SAMPLING_BAND,
        "primary_baseline": {
            "name": "PE-FraudGT",
            "reason": "direct parent architecture of the CDVT account view",
            "source": FRAUDGT_PAPER_REFERENCE,
        },
        "published_strong_reference": {
            "name": "Multi-FraudGT",
            "reason": "strongest overall FraudGT variant in paper Table 2",
            "source": FRAUDGT_PAPER_REFERENCE,
        },
        "internal_strong_reference": {
            "name": "initial A2",
            "note": (
                "historical point estimates; raw-best is supplementary and "
                "must not be compared with published val-selected results"
            ),
        },
        "rows": rows,
        "missing_manifests": missing,
        "val_selected_vs_pe_fraudgt": val_pe,
        "val_selected_vs_multi_fraudgt": val_multi,
        "val_selected_vs_initial_a2": val_a2,
        "raw_best_vs_initial_a2": raw_a2,
        "phase2_gate": {
            "baseline": "PE-FraudGT paper Table 2",
            "at_least_four_val_selected_wins": (
                len(rows) == len(DATASETS) and val_pe["wins"] >= 4),
            "positive_val_selected_mean_delta": (
                len(rows) == len(DATASETS) and val_pe["mean_delta"] > 0),
            "advance_to_phase3": gate,
        },
    }


def markdown(summary):
    lines = [
        "# CDVT Phase 2 Six-Dataset Summary",
        "",
        "Frozen model: causal event graph plus dual-view cross-attention, "
        "without sampling consistency.",
        "",
        "Sampling protocol: `dynamic_random`. Formal selection uses validation "
        "F1 against PE-FraudGT, the direct parent architecture reported in "
        "FraudGT Table 2. Multi-FraudGT is the stronger published reference. "
        "Raw-best is supplementary and is compared only with A2 raw-best.",
        "",
        "## Published FraudGT references",
        "",
        "| Dataset | PE-FraudGT | CDVT | Delta vs PE | Status | "
        "Multi-FraudGT | Delta vs Multi |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['pe_fraudgt_paper_val_selected_test_f1']:.5f} "
            f"| {row['val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected_vs_pe_fraudgt']:+.5f} "
            f"| {row['pe_fraudgt_interpretation']} "
            f"| {row['multi_fraudgt_paper_val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected_vs_multi_fraudgt']:+.5f} |")
    lines.extend([
        "",
        "## Internal A2 reference",
        "",
        "| Dataset | A2 val-selected | CDVT val-selected | Delta | Status | "
        "A2 raw-best | CDVT raw-best | Delta |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ])
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} "
            f"| {row['initial_a2_val_selected_test_f1']:.5f} "
            f"| {row['val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected_vs_initial_a2']:+.5f} "
            f"| {row['initial_a2_val_interpretation']} "
            f"| {row['initial_a2_raw_best_test_f1']:.5f} "
            f"| {row['raw_best_test_f1']:.5f} "
            f"| {row['delta_raw_best_vs_initial_a2']:+.5f} |")
    val_pe = summary["val_selected_vs_pe_fraudgt"]
    val_multi = summary["val_selected_vs_multi_fraudgt"]
    val_a2 = summary["val_selected_vs_initial_a2"]
    raw_a2 = summary["raw_best_vs_initial_a2"]
    lines.extend([
        "",
        f"Val-selected vs PE-FraudGT: {val_pe['wins']} wins, "
        f"{val_pe['losses']} losses, mean delta "
        f"{val_pe['mean_delta']:+.5f}." if val_pe["mean_delta"] is not None
        else "Val-selected vs PE-FraudGT: no completed datasets.",
        f"Val-selected vs Multi-FraudGT: {val_multi['wins']} wins, "
        f"{val_multi['losses']} losses, mean delta "
        f"{val_multi['mean_delta']:+.5f}."
        if val_multi["mean_delta"] is not None
        else "Val-selected vs Multi-FraudGT: no completed datasets.",
        f"Val-selected vs A2: {val_a2['wins']} wins, "
        f"{val_a2['losses']} losses, mean delta "
        f"{val_a2['mean_delta']:+.5f}." if val_a2["mean_delta"] is not None
        else "Val-selected vs A2: no completed datasets.",
        f"Raw-best vs A2: {raw_a2['wins']} wins, "
        f"{raw_a2['losses']} losses, mean delta "
        f"{raw_a2['mean_delta']:+.5f}." if raw_a2["mean_delta"] is not None
        else "Raw-best vs A2: no completed datasets.",
        "",
        f"Advance to Phase 3: "
        f"{summary['phase2_gate']['advance_to_phase3']}.",
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
        allow_incomplete=args.allow_incomplete,
    )
    if args.write:
        args.phase2_root.mkdir(parents=True, exist_ok=True)
        (args.phase2_root / "phase2_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.phase2_root / "phase2_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
