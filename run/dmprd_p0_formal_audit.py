#!/usr/bin/env python3
"""Audit six 500-epoch P0 runs against matched historical A2 runs."""

import argparse
import json
from pathlib import Path


TASKS = [
    ("Small-LI", 42),
    ("Small-HI", 42),
    ("Medium-LI", 44),
    ("Medium-HI", 42),
    ("Large-LI", 44),
    ("Large-HI", 43),
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


def a2_run(root, dataset, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDFormal500-a2_multi-Seed{seed}"
    return find_run(root, stem, seed, epoch_limit)


def p0_run(root, dataset, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDP0Formal500-p0_full-Seed{seed}"
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
        "--p0", default=str(repo / "results" / "dmprd_p0_formal500"))
    parser.add_argument(
        "--a2", default=str(repo / "results" / "dmprd_formal500"))
    parser.add_argument("--epoch-limit", type=int, default=499)
    args = parser.parse_args()
    p0_root = Path(args.p0)
    a2_root = Path(args.a2)

    print(
        "dataset\tseed\ta2_last\tp0_last\ta2_val_select\tp0_val_select"
        "\tdelta_val\ta2_raw_best\tp0_raw_best\tdelta_raw"
        "\ta2_val_epoch\tp0_val_epoch\ta2_raw_epoch\tp0_raw_epoch")
    paired = []
    for dataset, seed in TASKS:
        a2 = a2_run(a2_root, dataset, seed, args.epoch_limit)
        p0 = p0_run(p0_root, dataset, seed, args.epoch_limit)
        a2_val = metric(a2, "test_at_val")
        p0_val = metric(p0, "test_at_val")
        a2_raw = metric(a2, "raw_best")
        p0_raw = metric(p0, "raw_best")
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

    print("p0_formal_go_no_go")
    if paired:
        mean_val = sum(item[1] for item in paired) / len(paired)
        mean_raw = sum(item[2] for item in paired) / len(paired)
        val_wins = sum(1 for item in paired if item[1] > 0.0)
        raw_wins = sum(1 for item in paired if item[2] > 0.0)
        raw_large_losses = sum(1 for item in paired if item[2] < -0.010)
        print(
            f"paired={len(paired)}/6 mean_delta_val={mean_val:+.5f} "
            f"mean_delta_raw={mean_raw:+.5f} val_wins={val_wins}/6 "
            f"raw_wins={raw_wins}/6 "
            f"raw_losses_below_minus_0.010={raw_large_losses}")
        if len(paired) == 6:
            passed = (
                mean_val > 0.005 and val_wins >= 4 and
                mean_raw >= 0.0 and raw_large_losses <= 1)
            print(f"six_dataset_formal_screen={'PASS' if passed else 'FAIL'}")
        else:
            print("six_dataset_formal_screen=INCOMPLETE")
    else:
        print("paired=0/6 six_dataset_formal_screen=INCOMPLETE")


if __name__ == "__main__":
    main()
