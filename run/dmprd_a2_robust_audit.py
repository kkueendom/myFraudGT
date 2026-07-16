#!/usr/bin/env python3
"""Aggregate the formal and supplemental DMPRD-A2 runs over three seeds."""

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
BEST_SEED = {
    "Small-HI": 42,
    "Small-LI": 42,
    "Medium-HI": 42,
    "Medium-LI": 44,
    "Large-HI": 43,
    "Large-LI": 44,
}
BASELINE = {
    "Small-HI": {"raw": 0.78599, "val": 0.76683},
    "Small-LI": {"raw": 0.50474, "val": 0.50355},
    "Medium-HI": {"raw": 0.76768, "val": 0.74601},
    "Medium-LI": {"raw": 0.51852, "val": 0.43231},
    "Large-HI": {"raw": 0.72120, "val": 0.68048},
    "Large-LI": {"raw": 0.41379, "val": 0.35000},
}


def rows(path, epoch_limit):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" not in row or int(row["epoch"]) > epoch_limit:
            continue
        output.append(row)
    return output


def summarize(seed_dir, epoch_limit):
    train = rows(seed_dir / "train" / "stats.json", epoch_limit)
    val = rows(seed_dir / "val" / "stats.json", epoch_limit)
    test = rows(seed_dir / "test" / "stats.json", epoch_limit)
    if not train or not val or not test:
        return None
    test_by_epoch = {
        int(row["epoch"]): row for row in test if "f1" in row}
    best_val = max(
        (row for row in val if "f1" in row),
        key=lambda row: float(row["f1"]))
    val_epoch = int(best_val["epoch"])
    selected_test = test_by_epoch.get(val_epoch)
    raw_best = max(
        (row for row in test if "f1" in row),
        key=lambda row: float(row["f1"]))
    return {
        "last_epoch": int(train[-1]["epoch"]),
        "test_at_val": (
            float(selected_test["f1"]) if selected_test else None),
        "val_epoch": val_epoch,
        "raw_best": float(raw_best["f1"]),
        "raw_epoch": int(raw_best["epoch"]),
    }


def find_run(root, stem, seed, epoch_limit):
    for run_dir in sorted(root.glob(stem + "-gpu*")):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            return result
    return None


def result_for(formal_root, robust_root, dataset, seed, epoch_limit):
    if seed == BEST_SEED[dataset]:
        stem = f"AML-{dataset}-DMPRDFormal500-a2_multi-Seed{seed}"
        return find_run(formal_root, stem, seed, epoch_limit)
    stem = f"AML-{dataset}-DMPRDA2Robust500-a2_multi-Seed{seed}"
    return find_run(robust_root, stem, seed, epoch_limit)


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal", default=str(repo / "results" / "dmprd_formal500"))
    parser.add_argument(
        "--robust", default=str(repo / "results" / "dmprd_a2_robust500"))
    parser.add_argument("--epoch-limit", type=int, default=499)
    args = parser.parse_args()

    formal_root = Path(args.formal)
    robust_root = Path(args.robust)
    dataset_results = {}
    print(
        "dataset\tseed\tlast_epoch\ttest_val_select\tval_epoch"
        "\ttest_raw_best\traw_epoch\tbaseline_val\tbaseline_raw")
    for dataset in BEST_SEED:
        available = []
        for seed in SEEDS:
            result = result_for(
                formal_root, robust_root, dataset, seed, args.epoch_limit)
            if result:
                available.append(result)
            print("\t".join([
                dataset,
                str(seed),
                str(result["last_epoch"]) if result else "-",
                fmt(result.get("test_at_val") if result else None),
                str(result["val_epoch"]) if result else "-",
                fmt(result.get("raw_best") if result else None),
                str(result["raw_epoch"]) if result else "-",
                fmt(BASELINE[dataset]["val"]),
                fmt(BASELINE[dataset]["raw"]),
            ]))
        dataset_results[dataset] = available

    print("dmprd_a2_robust_summary")
    robust_datasets = 0
    complete_datasets = 0
    for dataset, results in dataset_results.items():
        if not results:
            print(f"{dataset}\tmatched=0/3")
            continue
        raw = [item["raw_best"] for item in results]
        val = [
            item["test_at_val"] for item in results
            if item["test_at_val"] is not None]
        raw_wins = sum(value > BASELINE[dataset]["raw"] for value in raw)
        val_wins = sum(value > BASELINE[dataset]["val"] for value in val)
        raw_mean = statistics.mean(raw)
        raw_std = statistics.pstdev(raw) if len(raw) > 1 else 0.0
        val_mean = statistics.mean(val) if val else None
        val_std = statistics.pstdev(val) if len(val) > 1 else 0.0
        print(
            f"{dataset}\tmatched={len(results)}/3 "
            f"val_mean={fmt(val_mean)} val_std={fmt(val_std)} "
            f"raw_mean={raw_mean:.5f} raw_std={raw_std:.5f} "
            f"val_wins={val_wins}/{len(val)} "
            f"raw_wins={raw_wins}/{len(raw)}")
        if len(results) == 3:
            complete_datasets += 1
            if raw_mean > BASELINE[dataset]["raw"] and raw_wins >= 2:
                robust_datasets += 1
    print(
        f"complete_datasets={complete_datasets}/6 "
        f"robust_raw_datasets={robust_datasets}/6")
    if complete_datasets == 6:
        print(
            "three_seed_mainline_ready="
            f"{'PASS' if robust_datasets == 6 else 'FAIL'}")


if __name__ == "__main__":
    main()
