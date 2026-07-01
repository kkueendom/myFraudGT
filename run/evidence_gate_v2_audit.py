#!/usr/bin/env python3
"""Audit evidence-gate v2 formal runs."""

import argparse
import json
from pathlib import Path


DATASETS = [
    "Small-HI",
    "Small-LI",
    "Medium-HI",
    "Medium-LI",
    "Large-HI",
    "Large-LI",
]
SEEDS = [42, 43, 44, 45, 46]
DONE_EPOCH = 239


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            output.append(row)
    return output


def run_dirs(root, dataset):
    return sorted(Path(root).glob(f"AML-{dataset}-gpu*"))


def summarize_run(seed_dir):
    train_rows = rows(seed_dir / "train" / "stats.json")
    val_rows = rows(seed_dir / "val" / "stats.json")
    test_rows = rows(seed_dir / "test" / "stats.json")
    if not train_rows or not val_rows or not test_rows:
        return None

    test_by_epoch = {int(row["epoch"]): row for row in test_rows}
    best_val = max(val_rows, key=lambda row: float(row.get("f1", 0.0)))
    val_epoch = int(best_val["epoch"])
    test_at_val = test_by_epoch.get(val_epoch)
    raw_best_test = max(test_rows, key=lambda row: float(row.get("f1", 0.0)))

    return {
        "train_last": int(train_rows[-1]["epoch"]),
        "done": int(train_rows[-1]["epoch"]) >= DONE_EPOCH,
        "val_epoch": val_epoch,
        "val_f1": float(best_val.get("f1", 0.0)),
        "test_at_val_f1": float(test_at_val.get("f1", 0.0)) if test_at_val else None,
        "raw_best_epoch": int(raw_best_test["epoch"]),
        "raw_best_test_f1": float(raw_best_test.get("f1", 0.0)),
    }


def find_summary(roots, dataset, seed):
    for root in roots:
        for run_dir in run_dirs(root, dataset):
            summary = summarize_run(run_dir / str(seed))
            if summary:
                summary["root"] = str(root)
                summary["run_dir"] = str(run_dir)
                return summary
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_v2_fixed",
        ],
    )
    args = parser.parse_args()
    roots = [Path(root) for root in args.roots]

    print("| Dataset | Seed | Done | Train last | Val epoch | Val F1 | Test@Val F1 | Raw-best test F1 | Run dir |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    done_count = 0
    total = len(DATASETS) * len(SEEDS)
    for dataset in DATASETS:
        for seed in SEEDS:
            summary = find_summary(roots, dataset, seed)
            if not summary:
                print(f"| {dataset} | {seed} | 0 | - | - | - | - | - | - |")
                continue
            done_count += int(summary["done"])
            test_at_val = summary["test_at_val_f1"]
            print(
                "| {dataset} | {seed} | {done} | {train_last} | {val_epoch} | "
                "{val_f1:.5f} | {test_at_val} | {raw_best:.5f} | `{run_dir}` |".format(
                    dataset=dataset,
                    seed=seed,
                    done=int(summary["done"]),
                    train_last=summary["train_last"],
                    val_epoch=summary["val_epoch"],
                    val_f1=summary["val_f1"],
                    test_at_val=f"{test_at_val:.5f}" if test_at_val is not None else "-",
                    raw_best=summary["raw_best_test_f1"],
                    run_dir=summary["run_dir"],
                )
            )
    print()
    print(f"completed_runs: {done_count}/{total}")


if __name__ == "__main__":
    main()
