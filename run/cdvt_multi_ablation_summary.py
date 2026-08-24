#!/usr/bin/env python3
"""Audit current Multi-CDVT seed-42 ablations against audited Full runs."""

import argparse
import json
from pathlib import Path

DATASETS = ("Small-LI", "Small-HI", "Medium-LI", "Medium-HI", "Large-LI", "Large-HI")
LABELS = ("multi_account_only", "multi_causal_event_add", "multi_dual_view_no_relation")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--large-hi-full-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()

def read_manifest(path):
    payload = json.loads(path.read_text())
    required = {
        "sampling_protocol": "dynamic_random",
        "seed": 42,
        "max_epochs": 500,
        "early_stopping_enabled": False,
        "account_backbone": "Multi-FraudGT",
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
    if payload.get("epochs_completed") != 500:
        raise ValueError(f"{path}: incomplete 500-epoch run")
    payload["manifest_path"] = str(path.resolve())
    return payload

def ablation_manifest(root, dataset, label):
    path = root / f"{dataset}_{label}_seed42" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_manifest(path)

def full_manifest(root, large_hi_root, dataset):
    base = large_hi_root if dataset == "Large-HI" else root
    for name in (f"{dataset}_multi_cdvt_seed42", f"{dataset}_dual_view_seed42"):
        path = base / name / "manifest.json"
        if path.is_file():
            payload = read_manifest(path)
            if payload.get("variant") in {"multi_cdvt", "dual_view"}:
                return payload
    raise FileNotFoundError(base / f"{dataset}_multi_cdvt_seed42" / "manifest.json")

def build(ablation_root, full_root, large_hi_root):
    rows = []
    for dataset in DATASETS:
        full = full_manifest(full_root, large_hi_root, dataset)
        labels = LABELS if dataset.endswith("LI") else ("multi_account_only",)
        for label in labels:
            item = ablation_manifest(ablation_root, dataset, label)
            rows.append({
                "dataset": dataset,
                "variant": label,
                "val_selected_test_f1": item["val_selected_test_f1"],
                "full_val_selected_test_f1": full["val_selected_test_f1"],
                "full_minus_variant": full["val_selected_test_f1"] - item["val_selected_test_f1"],
                "manifest": item["manifest_path"],
                "full_manifest": full["manifest_path"],
            })
    means = {}
    for label in LABELS:
        selected = [row for row in rows if row["variant"] == label]
        means[label] = {
            "datasets": len(selected),
            "mean_f1": sum(row["val_selected_test_f1"] for row in selected) / len(selected),
            "mean_full_minus_variant": sum(row["full_minus_variant"] for row in selected) / len(selected),
        }
    return {"sampling_protocol": "dynamic_random", "metric": "val_selected_test_f1", "rows": rows, "means": means, "complete": True}

def markdown(summary):
    lines = [
        "# Multi-CDVT Seed-42 Ablation Summary",
        "",
        "All rows are audited 500-epoch runs under dynamic_random. The primary comparison is Val-selected Test F1; full_minus_variant is the contribution of the complete Multi-CDVT model.",
        "",
        "| Dataset | Variant | Variant F1 | Full F1 | Full - Variant |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['val_selected_test_f1']:.5f} | {row['full_val_selected_test_f1']:.5f} | {row['full_minus_variant']:+.5f} |")
    lines.extend(["", "| Variant | Datasets | Mean F1 | Mean Full - Variant |", "|---|---:|---:|---:|"])
    for label, value in summary["means"].items():
        lines.append(f"| {label} | {value['datasets']} | {value['mean_f1']:.5f} | {value['mean_full_minus_variant']:+.5f} |")
    return "\n".join(lines) + "\n"

def main():
    options = parse_args()
    summary = build(options.ablation_root, options.full_root, options.large_hi_full_root)
    if options.write:
        (options.ablation_root / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (options.ablation_root / "ablation_summary.md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
