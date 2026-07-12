#!/usr/bin/env python3
"""Audit evidence-gate v4 ablation runs with validation-selected test F1."""

import argparse
import json
import math
import re
from pathlib import Path

DATASETS = ["Small-HI", "Small-LI", "Medium-HI", "Medium-LI", "Large-HI", "Large-LI"]
DEFAULT_VARIANTS = ["proto_only", "no_gate", "no_prototype", "no_aux", "no_budget"]
BEST_SEED_BY_DATASET = {
    "Small-HI": 42,
    "Small-LI": 42,
    "Medium-HI": 42,
    "Medium-LI": 44,
    "Large-HI": 43,
    "Large-LI": 44,
}
DONE_EPOCH = 499


def rows(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            out.append(row)
    return out


def run_dirs(out, dataset, variant, seed):
    stems = [
        f"AML-{dataset}-V4BestSeedAblation-{variant}-Seed{seed}",
        f"AML-{dataset}-V4FormalAblation-{variant}-Seed{seed}",
        f"AML-{dataset}-V4Ablation-{variant}-Seed{seed}",
    ]
    matches = []
    for stem in stems:
        matches.extend(sorted(out.glob(stem + "-gpu*")))
    return matches


def best_test(rows_):
    f1_rows = [r for r in rows_ if "f1" in r]
    if not f1_rows:
        return None, None
    best = max(f1_rows, key=lambda r: float(r["f1"]))
    return float(best["f1"]), int(best["epoch"])


def val_selected_test(val_rows, test_rows):
    val_f1 = [r for r in val_rows if "f1" in r]
    by_epoch = {int(r["epoch"]): r for r in test_rows if "f1" in r}
    if not val_f1:
        return None, None, None
    selected = max(val_f1, key=lambda r: float(r["f1"]))
    epoch = int(selected["epoch"])
    test = by_epoch.get(epoch)
    return (float(test["f1"]) if test else None, epoch,
            float(selected["f1"]))


def state(out, dataset, variant, seed):
    matches = run_dirs(out, dataset, variant, seed)
    if not matches:
        return {"status": "MISS", "complete": False, "run_dir": "-"}
    run_dir = matches[0]
    seed_dir = run_dir / str(seed)
    train = rows(seed_dir / "train" / "stats.json")
    val = rows(seed_dir / "val" / "stats.json")
    test = rows(seed_dir / "test" / "stats.json")
    train_last = int(train[-1]["epoch"]) if train else None
    test_best, test_epoch = best_test(test)
    selected_test, selected_epoch, selected_val = val_selected_test(val, test)
    complete = bool(train and val and test and train_last >= DONE_EPOCH)
    status = "COMPLETE" if complete else ("RUNNING" if train_last is not None else "MISS")
    return {
        "status": status,
        "complete": complete,
        "train_last": train_last,
        "test_raw_best": test_best,
        "test_raw_best_epoch": test_epoch,
        "test_val_select": selected_test,
        "val_select_epoch": selected_epoch,
        "val_select_f1": selected_val,
        "run_dir": str(run_dir),
    }


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.5f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/e/yky/FraudGT_evidence_gate_v4/results/evidence_gate_v4_best_seed_ablation")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seed-policy", choices=["best", "fixed"], default="best")
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    out = Path(args.out)
    variants = [v for v in args.variants.split(",") if v]
    fixed_seeds = [int(s) for s in args.seeds.split(",") if s]
    print("variant\tdataset\tseed\tstatus\tcomplete\ttrain_last\ttest_val_select\tval_epoch\tval_f1\ttest_raw_best\traw_epoch\trun_dir")
    summary = {}
    for variant in variants:
        for dataset in DATASETS:
            seeds = [BEST_SEED_BY_DATASET[dataset]] if args.seed_policy == "best" else fixed_seeds
            vals = []
            for seed in seeds:
                row = state(out, dataset, variant, seed)
                vals.append(row.get("test_val_select") if row.get("complete") else None)
                print("\t".join([
                    variant, dataset, str(seed), row["status"], "yes" if row.get("complete") else "no",
                    fmt(row.get("train_last")), fmt(row.get("test_val_select")), fmt(row.get("val_select_epoch")),
                    fmt(row.get("val_select_f1")), fmt(row.get("test_raw_best")), fmt(row.get("test_raw_best_epoch")), row["run_dir"],
                ]))
            mean, std = mean_std(vals)
            summary[(variant, dataset)] = (len([v for v in vals if v is not None]), len(seeds), mean, std)
    print("ablation_dataset_summary")
    for variant in variants:
        for dataset in DATASETS:
            n, total, mean, std = summary[(variant, dataset)]
            print("\t".join([variant, dataset, f"n={n}/{total}", f"mean={fmt(mean)}", f"std={fmt(std)}"]))


if __name__ == "__main__":
    main()
