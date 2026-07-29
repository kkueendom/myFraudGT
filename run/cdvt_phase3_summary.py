#!/usr/bin/env python3
"""Summarize paired three-seed FraudGT and CDVT representative runs."""

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
VARIANTS = ("account_only", "dual_view")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def result_path(
    dataset, seed, variant, phase1_root, phase2_root, phase3_root,
    ablation_root,
):
    name = f"{dataset}_{variant}_seed{seed}/manifest.json"
    if seed in {43, 44}:
        return phase3_root / name
    if variant == "account_only":
        root = (
            phase1_root
            if dataset in {"Small-LI", "Large-LI"}
            else ablation_root
        )
    else:
        root = (
            phase1_root
            if dataset in {"Small-LI", "Large-LI"}
            else phase2_root
        )
    return root / name


def validate_manifest(payload, path, dataset, seed, variant):
    expected = {
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": variant,
        "seed": seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: final comparison must use lambda_cons=0")
    if seed in {43, 44} and payload.get("phase") != "CDVT_phase3":
        raise ValueError(f"{path}: follow-up seed must be CDVT_phase3")
    if (
        seed == 42
        and dataset == "Medium-LI"
        and variant == "account_only"
        and payload.get("phase") != "CDVT_ablation"
    ):
        raise ValueError(
            f"{path}: Medium-LI seed-42 account control must be an ablation")

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
    phase1_root, phase2_root, phase3_root, ablation_root,
    allow_incomplete=False,
):
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = result_path(
                    dataset,
                    seed,
                    variant,
                    phase1_root,
                    phase2_root,
                    phase3_root,
                    ablation_root,
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text())
                validate_manifest(payload, path, dataset, seed, variant)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "git_commit": str(payload["git_commit"]),
                    "config": str(payload["config"]),
                    "checkpoint": str(payload["checkpoint"]),
                    "val_selected_epoch": int(
                        payload["val_selected_epoch"]),
                    "val_selected_test_f1": float(
                        payload["val_selected_test_f1"]),
                    "raw_best_epoch": int(payload["raw_best_epoch"]),
                    "raw_best_test_f1": float(
                        payload["raw_best_test_f1"]),
                    "manifest": str(path.resolve()),
                })
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "paired three-seed result set is incomplete: "
            + ", ".join(missing))
    return rows, missing


def distribution(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def metric_distribution(rows, metric):
    return distribution([float(row[metric]) for row in rows])


def build_summary(
    phase1_root,
    phase2_root,
    phase3_root,
    ablation_root,
    allow_incomplete=False,
):
    rows, missing = load_rows(
        Path(phase1_root),
        Path(phase2_root),
        Path(phase3_root),
        Path(ablation_root),
        allow_incomplete=allow_incomplete,
    )
    paired_rows = []
    groups = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        account_rows = [
            row for row in dataset_rows
            if row["variant"] == "account_only"
        ]
        cdvt_rows = [
            row for row in dataset_rows if row["variant"] == "dual_view"
        ]
        for seed in SEEDS:
            account = next(
                (row for row in account_rows if row["seed"] == seed), None)
            cdvt = next(
                (row for row in cdvt_rows if row["seed"] == seed), None)
            if account is None or cdvt is None:
                continue
            paired_rows.append({
                "dataset": dataset,
                "seed": seed,
                "fraudgt_val_selected_test_f1": account[
                    "val_selected_test_f1"],
                "cdvt_val_selected_test_f1": cdvt[
                    "val_selected_test_f1"],
                "delta_val_selected": (
                    cdvt["val_selected_test_f1"]
                    - account["val_selected_test_f1"]
                ),
                "fraudgt_raw_best_test_f1": account["raw_best_test_f1"],
                "cdvt_raw_best_test_f1": cdvt["raw_best_test_f1"],
                "delta_raw_best": (
                    cdvt["raw_best_test_f1"]
                    - account["raw_best_test_f1"]
                ),
            })
        dataset_pairs = [
            row for row in paired_rows if row["dataset"] == dataset]
        account_val = metric_distribution(
            account_rows, "val_selected_test_f1")
        cdvt_val = metric_distribution(cdvt_rows, "val_selected_test_f1")
        account_raw = metric_distribution(account_rows, "raw_best_test_f1")
        cdvt_raw = metric_distribution(cdvt_rows, "raw_best_test_f1")
        if cdvt_val is not None:
            cdvt_val["delta_mean_vs_pe_fraudgt_paper"] = (
                cdvt_val["mean"] - PE_FRAUDGT_PAPER[dataset])
            cdvt_val["delta_mean_vs_multi_fraudgt_paper"] = (
                cdvt_val["mean"] - MULTI_FRAUDGT_PAPER[dataset])
            cdvt_val["delta_mean_vs_initial_a2"] = (
                cdvt_val["mean"]
                - INITIAL_A2[dataset]["val_selected_test_f1"])
        if cdvt_raw is not None:
            cdvt_raw["delta_mean_vs_initial_a2"] = (
                cdvt_raw["mean"] - INITIAL_A2[dataset]["raw_best_test_f1"])
        groups.append({
            "dataset": dataset,
            "seeds": sorted({row["seed"] for row in dataset_rows}),
            "published_references": {
                "pe_fraudgt": PE_FRAUDGT_PAPER[dataset],
                "multi_fraudgt": MULTI_FRAUDGT_PAPER[dataset],
            },
            "initial_a2": INITIAL_A2[dataset],
            "fraudgt": {
                "val_selected": account_val,
                "raw_best": account_raw,
            },
            "cdvt": {
                "val_selected": cdvt_val,
                "raw_best": cdvt_raw,
            },
            "paired_delta": {
                "val_selected": distribution([
                    row["delta_val_selected"] for row in dataset_pairs]),
                "raw_best": distribution([
                    row["delta_raw_best"] for row in dataset_pairs]),
            },
        })
    return {
        "model": "CDVT dual_view without sampling consistency",
        "matched_baseline": "FraudGT account_only (PE-FraudGT architecture)",
        "sampling_protocol": "dynamic_random",
        "primary_baseline": {
            "name": "matched FraudGT account_only",
            "published_architecture": "PE-FraudGT",
        },
        "published_references": {
            "source": FRAUDGT_PAPER_REFERENCE,
            "direct_parent": "PE-FraudGT",
            "strongest_variant": "Multi-FraudGT",
        },
        "required_seeds": list(SEEDS),
        "rows": rows,
        "paired_rows": paired_rows,
        "datasets": groups,
        "missing_manifests": missing,
        "complete": (
            not missing
            and len(rows) == len(DATASETS) * len(SEEDS) * len(VARIANTS)
            and len(paired_rows) == len(DATASETS) * len(SEEDS)
        ),
    }


def fmt_mean_std(payload):
    if payload is None:
        return "TBD"
    std = (
        "TBD"
        if payload["sample_std"] is None
        else f"{payload['sample_std']:.5f}"
    )
    return f"{payload['mean']:.5f} +/- {std}"


def markdown(summary):
    lines = [
        "# CDVT Representative-Dataset Paired Three-Seed Summary",
        "",
        "FraudGT/account-only and CDVT use independent seeds 42, 43, and 44 "
        "under the same `dynamic_random` protocol. The primary robustness "
        "evidence is the same-seed paired delta. Published PE-FraudGT and "
        "Multi-FraudGT means are descriptive references.",
        "",
        "## Seed-level paired results",
        "",
        "| Dataset | Seed | FraudGT val-selected | CDVT val-selected | "
        "Paired delta | FraudGT raw-best | CDVT raw-best | Paired delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["paired_rows"]:
        lines.append(
            f"| {row['dataset']} | {row['seed']} "
            f"| {row['fraudgt_val_selected_test_f1']:.5f} "
            f"| {row['cdvt_val_selected_test_f1']:.5f} "
            f"| {row['delta_val_selected']:+.5f} "
            f"| {row['fraudgt_raw_best_test_f1']:.5f} "
            f"| {row['cdvt_raw_best_test_f1']:.5f} "
            f"| {row['delta_raw_best']:+.5f} |")
    lines.extend([
        "",
        "## Three-seed aggregate",
        "",
        "| Dataset | FraudGT val mean +/- std | CDVT val mean +/- std | "
        "Paired delta mean +/- std | Delta vs PE paper | "
        "Delta vs Multi paper |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for group in summary["datasets"]:
        account = group["fraudgt"]["val_selected"]
        cdvt = group["cdvt"]["val_selected"]
        paired = group["paired_delta"]["val_selected"]
        if account is None or cdvt is None or paired is None:
            continue
        lines.append(
            f"| {group['dataset']} "
            f"| {fmt_mean_std(account)} "
            f"| {fmt_mean_std(cdvt)} "
            f"| {fmt_mean_std(paired)} "
            f"| {cdvt['delta_mean_vs_pe_fraudgt_paper']:+.5f} "
            f"| {cdvt['delta_mean_vs_multi_fraudgt_paper']:+.5f} |")
    lines.extend([
        "",
        "## Supplementary raw-best aggregate",
        "",
        "| Dataset | FraudGT raw mean +/- std | CDVT raw mean +/- std | "
        "Paired delta mean +/- std | Delta vs A2 raw-best |",
        "|---|---:|---:|---:|---:|",
    ])
    for group in summary["datasets"]:
        account = group["fraudgt"]["raw_best"]
        cdvt = group["cdvt"]["raw_best"]
        paired = group["paired_delta"]["raw_best"]
        if account is None or cdvt is None or paired is None:
            continue
        lines.append(
            f"| {group['dataset']} "
            f"| {fmt_mean_std(account)} "
            f"| {fmt_mean_std(cdvt)} "
            f"| {fmt_mean_std(paired)} "
            f"| {cdvt['delta_mean_vs_initial_a2']:+.5f} |")
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
        args.ablation_root,
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
