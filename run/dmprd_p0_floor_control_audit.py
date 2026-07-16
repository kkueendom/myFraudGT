#!/usr/bin/env python3
"""Audit P0 against a fixed 0.25 residual-gate control."""

import argparse
import json
from pathlib import Path


DATASETS = [
    ("Small-LI", 42),
    ("Medium-HI", 42),
    ("Large-LI", 44),
]


def rows(path, epoch_limit):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row and int(row["epoch"]) <= epoch_limit:
            output.append(row)
    return output


def summarize(seed_dir, epoch_limit):
    train = rows(seed_dir / "train" / "stats.json", epoch_limit)
    val = rows(seed_dir / "val" / "stats.json", epoch_limit)
    test = rows(seed_dir / "test" / "stats.json", epoch_limit)
    if not train or not val or not test:
        return None
    val_f1 = [row for row in val if "f1" in row]
    test_f1 = [row for row in test if "f1" in row]
    if not val_f1 or not test_f1:
        return None
    test_by_epoch = {int(row["epoch"]): row for row in test_f1}
    best_val = max(val_f1, key=lambda row: float(row["f1"]))
    val_epoch = int(best_val["epoch"])
    selected_test = test_by_epoch.get(val_epoch)
    raw_best = max(test_f1, key=lambda row: float(row["f1"]))
    return {
        "last_epoch": int(train[-1]["epoch"]),
        "val_epoch": val_epoch,
        "test_at_val": (
            float(selected_test["f1"]) if selected_test else None),
        "raw_epoch": int(raw_best["epoch"]),
        "raw_best": float(raw_best["f1"]),
    }


def find_run(root, stem, seed, epoch_limit):
    candidates = []
    for run_dir in sorted(root.glob(stem + "-gpu*")):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            result["run_dir"] = str(run_dir)
            candidates.append(result)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["last_epoch"])


def original_run(root, dataset, variant, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDP0Quick120-{variant}-Seed{seed}"
    return find_run(root, stem, seed, epoch_limit)


def control_run(root, dataset, variant, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDP0FloorControl120-{variant}-Seed{seed}"
    return find_run(root, stem, seed, epoch_limit)


def complete(run, epoch_limit):
    return run is not None and run["last_epoch"] >= epoch_limit


def metric(run, key):
    return run.get(key) if run else None


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", default=str(repo / "results" / "dmprd_p0_quick120"))
    parser.add_argument(
        "--control", default=str(
            repo / "results" / "dmprd_p0_floor_control120"))
    parser.add_argument("--epoch-limit", type=int, default=119)
    args = parser.parse_args()
    original_root = Path(args.original)
    control_root = Path(args.control)

    print(
        "dataset\tseed\ta2_last\tfloor_last\tp0_last\ta2_val\tfloor_val"
        "\tp0_val\tp0_minus_floor_val\ta2_raw\tfloor_raw\tp0_raw"
        "\tp0_minus_floor_raw")
    comparisons = []
    for dataset, seed in DATASETS:
        a2 = original_run(
            original_root, dataset, "a2_control", seed, args.epoch_limit)
        p0 = original_run(
            original_root, dataset, "p0_full", seed, args.epoch_limit)
        floor = control_run(
            control_root, dataset, "floor025_control", seed,
            args.epoch_limit)
        p0_val = metric(p0, "test_at_val")
        floor_val = metric(floor, "test_at_val")
        p0_raw = metric(p0, "raw_best")
        floor_raw = metric(floor, "raw_best")
        delta_val = (
            p0_val - floor_val
            if p0_val is not None and floor_val is not None else None)
        delta_raw = (
            p0_raw - floor_raw
            if p0_raw is not None and floor_raw is not None else None)
        if (
                complete(p0, args.epoch_limit) and
                complete(floor, args.epoch_limit) and
                delta_val is not None and delta_raw is not None):
            comparisons.append((dataset, delta_val, delta_raw))
        print("\t".join([
            dataset,
            str(seed),
            str(a2["last_epoch"]) if a2 else "-",
            str(floor["last_epoch"]) if floor else "-",
            str(p0["last_epoch"]) if p0 else "-",
            fmt(metric(a2, "test_at_val")),
            fmt(floor_val),
            fmt(p0_val),
            fmt(delta_val),
            fmt(metric(a2, "raw_best")),
            fmt(floor_raw),
            fmt(p0_raw),
            fmt(delta_raw),
        ]))

    print("dynamic_gate_go_no_go")
    if comparisons:
        mean_val = sum(item[1] for item in comparisons) / len(comparisons)
        mean_raw = sum(item[2] for item in comparisons) / len(comparisons)
        val_wins = sum(1 for item in comparisons if item[1] > 0.0)
        print(
            f"paired={len(comparisons)}/3 mean_delta_val={mean_val:+.5f} "
            f"mean_delta_raw={mean_raw:+.5f} val_wins={val_wins}/3")
        if len(comparisons) == 3:
            passed = (
                mean_val > 0.005 and val_wins >= 2 and
                mean_raw >= -0.005)
            print(f"dynamic_gate_contributes={'PASS' if passed else 'FAIL'}")
        else:
            print("dynamic_gate_contributes=INCOMPLETE")
    else:
        print("paired=0/3 dynamic_gate_contributes=INCOMPLETE")

    seed43_a2 = control_run(
        control_root, "Small-LI", "a2_seed43", 43, args.epoch_limit)
    seed43_p0 = control_run(
        control_root, "Small-LI", "p0_seed43", 43, args.epoch_limit)
    print("small_li_seed43")
    print(
        "variant\tlast_epoch\tval_select\traw_best\tval_epoch\traw_epoch")
    for name, run in (("a2_seed43", seed43_a2), ("p0_seed43", seed43_p0)):
        print("\t".join([
            name,
            str(run["last_epoch"]) if run else "-",
            fmt(metric(run, "test_at_val")),
            fmt(metric(run, "raw_best")),
            str(run["val_epoch"]) if run else "-",
            str(run["raw_epoch"]) if run else "-",
        ]))


if __name__ == "__main__":
    main()
