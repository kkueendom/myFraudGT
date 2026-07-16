#!/usr/bin/env python3
"""Audit matched A2/P0 120-epoch runs using val-selected and raw-best F1."""

import argparse
import json
from pathlib import Path


PAIRS = [
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


def find_run(root, dataset, variant, seed, epoch_limit):
    stem = (
        f"AML-{dataset}-DMPRDP0Quick120-{variant}-Seed{seed}-gpu*")
    candidates = []
    for run_dir in sorted(root.glob(stem)):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            result["run_dir"] = str(run_dir)
            candidates.append(result)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["last_epoch"])


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def complete(run, epoch_limit):
    return run is not None and run["last_epoch"] >= epoch_limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(
            Path(__file__).resolve().parents[1] /
            "results" / "dmprd_p0_quick120"))
    parser.add_argument("--epoch-limit", type=int, default=119)
    args = parser.parse_args()
    root = Path(args.out)

    print(
        "dataset\tseed\ta2_last\tp0_last\ta2_val_select\tp0_val_select"
        "\tdelta_val\ta2_raw_best\tp0_raw_best\tdelta_raw"
        "\ta2_val_epoch\tp0_val_epoch\ta2_raw_epoch\tp0_raw_epoch")
    paired = []
    for dataset, seed in PAIRS:
        a2 = find_run(
            root, dataset, "a2_control", seed, args.epoch_limit)
        p0 = find_run(root, dataset, "p0_full", seed, args.epoch_limit)
        a2_val = a2.get("test_at_val") if a2 else None
        p0_val = p0.get("test_at_val") if p0 else None
        a2_raw = a2.get("raw_best") if a2 else None
        p0_raw = p0.get("raw_best") if p0 else None
        delta_val = (
            p0_val - a2_val
            if p0_val is not None and a2_val is not None else None)
        delta_raw = (
            p0_raw - a2_raw
            if p0_raw is not None and a2_raw is not None else None)
        if (
                complete(a2, args.epoch_limit) and
                complete(p0, args.epoch_limit) and
                delta_val is not None and delta_raw is not None):
            paired.append((dataset, delta_val, delta_raw))
        print("\t".join([
            dataset,
            str(seed),
            str(a2["last_epoch"]) if a2 else "-",
            str(p0["last_epoch"]) if p0 else "-",
            fmt(a2_val),
            fmt(p0_val),
            fmt(delta_val),
            fmt(a2_raw),
            fmt(p0_raw),
            fmt(delta_raw),
            str(a2["val_epoch"]) if a2 else "-",
            str(p0["val_epoch"]) if p0 else "-",
            str(a2["raw_epoch"]) if a2 else "-",
            str(p0["raw_epoch"]) if p0 else "-",
        ]))

    print("p0_go_no_go")
    if paired:
        mean_val = sum(item[1] for item in paired) / len(paired)
        mean_raw = sum(item[2] for item in paired) / len(paired)
        val_wins = sum(1 for item in paired if item[1] > 0.0)
        raw_large_losses = sum(1 for item in paired if item[2] < -0.010)
        print(
            f"paired={len(paired)}/3 mean_delta_val={mean_val:+.5f} "
            f"mean_delta_raw={mean_raw:+.5f} val_wins={val_wins}/3 "
            f"raw_losses_below_minus_0.010={raw_large_losses}")
        if len(paired) == 3:
            passed = (
                mean_val > 0.005 and val_wins >= 2 and
                mean_raw >= 0.0 and raw_large_losses <= 1)
            print(f"quick_screen={'PASS' if passed else 'FAIL'}")
        else:
            print("quick_screen=INCOMPLETE")
    else:
        print("paired=0/3 quick_screen=INCOMPLETE")

    supportvar = find_run(
        root, "Small-LI", "p0_supportvar", 42, args.epoch_limit)
    small_a2 = find_run(
        root, "Small-LI", "a2_control", 42, args.epoch_limit)
    small_full = find_run(
        root, "Small-LI", "p0_full", 42, args.epoch_limit)
    print("small_li_component_diagnostic")
    for name, run in (
            ("a2_control", small_a2),
            ("p0_supportvar", supportvar),
            ("p0_full", small_full)):
        print("\t".join([
            name,
            str(run["last_epoch"]) if run else "-",
            fmt(run.get("test_at_val") if run else None),
            fmt(run.get("raw_best") if run else None),
        ]))


if __name__ == "__main__":
    main()
