#!/usr/bin/env python3
"""Audit original FraudGT baseline runs using raw best test F1."""

import argparse
import json
import math
from pathlib import Path


DATASETS = ["Small-HI", "Small-LI", "Medium-HI", "Medium-LI", "Large-HI", "Large-LI"]
SEEDS = [42, 43, 44]
RESULT_ROOT = "original_raw_baseline"
MAX_EPOCH = 500


def stem(dataset, seed):
    return f"AML-{dataset}-OriginalSparseNodeGTPortsEgoRawSeed{seed}"


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


def find_run_dir(base, dataset, seed):
    root = base / RESULT_ROOT
    matches = sorted(root.glob(f"{stem(dataset, seed)}-gpu*"))
    return matches[0] if matches else root / f"{stem(dataset, seed)}-gpu?"


def run_state(base, dataset, seed):
    run_dir = find_run_dir(base, dataset, seed)
    seed_dir = run_dir / str(seed)
    val_rows = read_rows(seed_dir / "val" / "stats.json")
    test_rows = read_rows(seed_dir / "test" / "stats.json")
    train_rows = read_rows(seed_dir / "train" / "stats.json")
    if not val_rows or not test_rows:
        return {"status": "MISS", "complete": False, "run_dir": str(run_dir)}

    test_by_epoch = {int(row["epoch"]): row for row in test_rows}
    best_val = max(val_rows, key=lambda row: float(row["f1"]))
    best_test = max(test_rows, key=lambda row: float(row["f1"]))
    val_epoch = int(best_val["epoch"])
    selected_test = test_by_epoch.get(val_epoch)
    train_last = int(train_rows[-1]["epoch"]) if train_rows else None
    complete = train_last is not None and train_last >= MAX_EPOCH - 1
    status = "DONE" if complete else "RUNNING"
    return {
        "status": status,
        "complete": complete,
        "train_last": train_last,
        "val_best": float(best_val["f1"]),
        "val_epoch": val_epoch,
        "test_at_val": float(selected_test["f1"]) if selected_test else None,
        "raw_best_test": float(best_test["f1"]),
        "raw_best_test_epoch": int(best_test["epoch"]),
        "run_dir": str(run_dir),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/e/yyk/FraudGT_multi6")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    print(
        "\t".join(
            [
                "dataset",
                "seed",
                "status",
                "complete",
                "train_last",
                "raw_best_test",
                "raw_best_test_epoch",
                "test_at_best_val",
                "val_best",
                "val_epoch",
                "run_dir",
            ]
        )
    )

    total = 0
    complete_total = 0
    summaries = {}
    for dataset in DATASETS:
        values = []
        complete_count = 0
        for seed in seeds:
            state = run_state(base, dataset, seed)
            total += 1
            complete = bool(state.get("complete", False))
            complete_total += int(complete)
            complete_count += int(complete)
            if complete and state.get("raw_best_test") is not None:
                values.append(state["raw_best_test"])
            print(
                "\t".join(
                    [
                        dataset,
                        str(seed),
                        state["status"],
                        fmt(complete),
                        fmt(state.get("train_last")),
                        fmt(state.get("raw_best_test")),
                        fmt(state.get("raw_best_test_epoch")),
                        fmt(state.get("test_at_val")),
                        fmt(state.get("val_best")),
                        fmt(state.get("val_epoch")),
                        state["run_dir"],
                    ]
                )
            )
        mean, std = mean_std(values)
        summaries[dataset] = (len(values), complete_count, mean, std)

    print("original_raw_baseline_dataset_summary")
    for dataset in DATASETS:
        n, complete_count, mean, std = summaries[dataset]
        print(
            "\t".join(
                [
                    dataset,
                    f"n={n}/{len(seeds)}",
                    f"complete={complete_count}/{len(seeds)}",
                    f"raw_best_mean={fmt(mean)}",
                    f"raw_best_std={fmt(std)}",
                ]
            )
        )
    print(f"original_raw_baseline_complete_summary\t{complete_total}/{total}")
    if args.strict and complete_total < total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
