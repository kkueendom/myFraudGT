#!/usr/bin/env python3
"""Audit ACDR-Help Small-LI/Large-LI runs against matched 500-epoch A2."""

import argparse
import json
import os
import subprocess
from pathlib import Path


TASKS = [
    ("Small-LI", 42),
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
        "best_val_f1": float(best_val["f1"]),
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
    return find_run(
        root, f"AML-{dataset}-DMPRDFormal500-a2_multi-Seed{seed}",
        seed, epoch_limit)


def acdr_run(root, dataset, seed, revision, epoch_limit):
    return find_run(
        root, f"AML-{dataset}-ACDRHelpPair500-Seed{seed}-{revision}",
        seed, epoch_limit)


def current_revision(repo):
    return subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=str(repo),
        text=True, capture_output=True, check=True).stdout.strip()


def value(run, key):
    return run.get(key) if run else None


def difference(left, right):
    if left is None or right is None:
        return None
    return left - right


def fmt(number):
    return "-" if number is None else f"{number:.5f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    revision = current_revision(repo)
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=revision)
    parser.add_argument("--acdr", default=None)
    parser.add_argument(
        "--a2", default=os.environ.get(
            "A2_RESULTS_ROOT",
            "/e/yky/FraudGT_dmprd_quickcheck/results/dmprd_formal500"))
    parser.add_argument("--epoch-limit", type=int, default=499)
    parser.add_argument(
        "--minimum-delta", type=float, default=0.005,
        help="required per-dataset val-selected Test F1 improvement")
    args = parser.parse_args()

    acdr_root = Path(args.acdr) if args.acdr else (
        repo / "results" / f"acdr_help_pair500_{args.revision}")
    a2_root = Path(args.a2)

    print(
        "dataset\tseed\ta2_last\tacdr_last\ta2_best_val\tacdr_best_val"
        "\tdelta_best_val\ta2_test_at_val\tacdr_test_at_val"
        "\tdelta_test_at_val\ta2_raw_best\tacdr_raw_best\tdelta_raw"
        "\ta2_val_epoch\tacdr_val_epoch")
    paired = []
    for dataset, seed in TASKS:
        a2 = a2_run(a2_root, dataset, seed, args.epoch_limit)
        acdr = acdr_run(
            acdr_root, dataset, seed, args.revision, args.epoch_limit)
        delta_best_val = difference(
            value(acdr, "best_val_f1"), value(a2, "best_val_f1"))
        delta_selected = difference(
            value(acdr, "test_at_val"), value(a2, "test_at_val"))
        delta_raw = difference(
            value(acdr, "raw_best"), value(a2, "raw_best"))
        complete = (
            a2 is not None and acdr is not None and
            a2["last_epoch"] >= args.epoch_limit and
            acdr["last_epoch"] >= args.epoch_limit and
            delta_selected is not None)
        if complete:
            paired.append((dataset, delta_selected, delta_raw))
        print("\t".join([
            dataset,
            str(seed),
            str(a2["last_epoch"]) if a2 else "-",
            str(acdr["last_epoch"]) if acdr else "-",
            fmt(value(a2, "best_val_f1")),
            fmt(value(acdr, "best_val_f1")),
            fmt(delta_best_val),
            fmt(value(a2, "test_at_val")),
            fmt(value(acdr, "test_at_val")),
            fmt(delta_selected),
            fmt(value(a2, "raw_best")),
            fmt(value(acdr, "raw_best")),
            fmt(delta_raw),
            str(a2["val_epoch"]) if a2 else "-",
            str(acdr["val_epoch"]) if acdr else "-",
        ]))

    print("acdr_help_pair_go_no_go")
    if len(paired) != len(TASKS):
        print(
            f"status=INCOMPLETE paired={len(paired)}/{len(TASKS)} "
            f"epoch_limit={args.epoch_limit}")
        return

    mean_delta = sum(item[1] for item in paired) / len(paired)
    raw_mean = sum(
        item[2] for item in paired if item[2] is not None) / len(paired)
    wins = sum(1 for item in paired if item[1] >= args.minimum_delta)
    print(
        f"paired={len(paired)}/{len(TASKS)} "
        f"mean_delta_test_at_val={mean_delta:+.5f} "
        f"mean_delta_raw={raw_mean:+.5f} "
        f"wins_at_{args.minimum_delta:+.3f}={wins}/{len(TASKS)}")
    if args.epoch_limit < 499:
        print("status=INTERIM_ONLY final decision requires epoch_limit=499")
    else:
        print(
            "status=" +
            ("PASS" if wins == len(TASKS) else "FAIL"))


if __name__ == "__main__":
    main()
