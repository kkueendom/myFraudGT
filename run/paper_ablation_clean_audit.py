#!/usr/bin/env python3
"""Audit clean paper ablation runs using best-validation-selected test F1."""

import argparse
import json
import math
from pathlib import Path


DATASETS = [
    "Small-HI",
    "Small-LI",
    "Medium-HI",
    "Medium-LI",
    "Large-HI",
    "Large-LI",
]

TARGETS = {
    "Small-HI": 0.8013,
    "Small-LI": 0.5101,
    "Medium-HI": 0.7993,
    "Medium-LI": 0.4806,
    "Large-HI": 0.7734,
    "Large-LI": 0.4143,
}

VARIANT_MAX_EPOCH = {
    "base_dot": 240,
    "supportmix_consis_proto": 240,
    "classmix_proto_bound": 240,
    "class_split": 240,
    "full_dual_uncert_gate": 180,
}

DEFAULT_VARIANTS = list(VARIANT_MAX_EPOCH)
DEFAULT_SEEDS = [42]
ABLATION_ROOT = "paper_ablation_clean_v1"

BEST_SEED_BY_DATASET = {
    "Small-HI": 42,
    "Small-LI": 42,
    "Medium-HI": 42,
    "Medium-LI": 44,
    "Large-HI": 43,
    "Large-LI": 44,
}


def read_rows(stats_path):
    rows = []
    if not stats_path.exists():
        return rows
    for line in stats_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in row and "f1" in row:
            rows.append(row)
    return rows


def find_run_dir(base, dataset, variant, seed):
    stem = f"AML-{dataset}-PaperAblation-{variant}-Seed{seed}"
    root = base / ABLATION_ROOT
    matches = sorted(root.glob(f"{stem}-gpu*"))
    return matches[0] if matches else root / f"{stem}-gpu?"


def run_state(base, dataset, variant, seed):
    run_dir = find_run_dir(base, dataset, variant, seed)
    seed_dir = run_dir / str(seed)
    val_rows = read_rows(seed_dir / "val" / "stats.json")
    test_rows = read_rows(seed_dir / "test" / "stats.json")
    train_rows = read_rows(seed_dir / "train" / "stats.json")
    max_epoch = VARIANT_MAX_EPOCH[variant]

    if not val_rows or not test_rows:
        return {
            "status": "MISS",
            "run_dir": str(run_dir),
            "complete": False,
        }

    test_by_epoch = {int(row["epoch"]): row for row in test_rows}
    best_val = max(val_rows, key=lambda row: float(row["f1"]))
    best_test = max(test_rows, key=lambda row: float(row["f1"]))
    val_epoch = int(best_val["epoch"])
    selected_test = test_by_epoch.get(val_epoch)
    train_last = int(train_rows[-1]["epoch"]) if train_rows else None
    complete = train_last is not None and train_last >= max_epoch - 1
    selected_test_f1 = (
        float(selected_test["f1"]) if selected_test is not None else None
    )
    if selected_test_f1 is None:
        status = "NO_TEST_AT_VAL"
    elif complete:
        status = "COMPLETE"
    else:
        status = "RUNNING"
    return {
        "status": status,
        "run_dir": str(run_dir),
        "complete": complete,
        "train_last": train_last,
        "val_best": float(best_val["f1"]),
        "val_epoch": val_epoch,
        "test_at_val": selected_test_f1,
        "test_raw_best": float(best_test["f1"]),
        "test_raw_best_epoch": int(best_test["epoch"]),
        "target": TARGETS[dataset],
        "margin": (
            None if selected_test_f1 is None else selected_test_f1 - TARGETS[dataset]
        ),
    }


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(value, digits=5):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def seeds_for_dataset(dataset, seeds, seed_policy):
    if seed_policy == "best":
        return [BEST_SEED_BY_DATASET[dataset]]
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/e/yyk/FraudGT_multi6")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument(
        "--seed-policy",
        choices=["fixed", "best"],
        default="fixed",
        help="Use --seeds for all datasets or the recorded best mainline seed per dataset.",
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    variants = [item for item in args.variants.split(",") if item]

    header = [
        "variant",
        "dataset",
        "seed",
        "status",
        "complete",
        "val_best",
        "val_epoch",
        "test_at_val",
        "target",
        "margin",
        "test_raw_best",
        "raw_epoch",
        "run_dir",
    ]
    print("\t".join(header))

    all_complete = 0
    total = 0
    summaries = {}
    for variant in variants:
        for dataset in DATASETS:
            dataset_seeds = seeds_for_dataset(dataset, seeds, args.seed_policy)
            vals = []
            complete_count = 0
            for seed in dataset_seeds:
                state = run_state(base, dataset, variant, seed)
                total += 1
                complete_count += int(state.get("complete", False))
                all_complete += int(state.get("complete", False))
                if state.get("complete", False) and state.get("test_at_val") is not None:
                    vals.append(state["test_at_val"])
                print(
                    "\t".join(
                        [
                            variant,
                            dataset,
                            str(seed),
                            state["status"],
                            fmt(state.get("complete")),
                            fmt(state.get("val_best")),
                            fmt(state.get("val_epoch")),
                            fmt(state.get("test_at_val")),
                            fmt(TARGETS[dataset], 4),
                            fmt(state.get("margin")),
                            fmt(state.get("test_raw_best")),
                            fmt(state.get("test_raw_best_epoch")),
                            state["run_dir"],
                        ]
                    )
                )
            mean, std = mean_std(vals)
            summaries[(variant, dataset)] = {
                "n": len(vals),
                "complete": complete_count,
                "expected": len(dataset_seeds),
                "mean": mean,
                "std": std,
            }

    print("ablation_dataset_summary")
    for variant in variants:
        for dataset in DATASETS:
            row = summaries[(variant, dataset)]
            print(
                "\t".join(
                    [
                        variant,
                        dataset,
                        f"n={row['n']}/{row['expected']}",
                        f"complete={row['complete']}/{row['expected']}",
                        f"mean={fmt(row['mean'])}",
                        f"std={fmt(row['std'])}",
                        f"target={TARGETS[dataset]:.4f}",
                    ]
                )
            )
    print(f"ablation_seed_complete_summary\t{all_complete}/{total}")

    if args.strict and all_complete < total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
