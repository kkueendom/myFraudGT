#!/usr/bin/env python3
"""Audit CPAR-K4 Small-LI/Large-LI runs against matched A2 trajectories."""

import argparse
import json
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
        "val_f1": float(best_val["f1"]),
        "test_at_val": (
            float(selected_test["f1"]) if selected_test else None),
        "raw_epoch": int(raw_best["epoch"]),
        "raw_best": float(raw_best["f1"]),
    }


def find_run(root, patterns, seed, epoch_limit):
    candidates = []
    seen = set()
    for pattern in patterns:
        for run_dir in sorted(root.glob(pattern)):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            result = summarize(run_dir / str(seed), epoch_limit)
            if result:
                result["run_dir"] = str(run_dir)
                candidates.append(result)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["last_epoch"])


def a2_run(root, dataset, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDFormal500-a2_multi-Seed{seed}-gpu*"
    return find_run(root, [stem], seed, epoch_limit)


def cpar_run(root, dataset, seed, epoch_limit, commit):
    commit_pattern = f"{commit}*" if commit else "*"
    stem = (
        f"AML-{dataset}-CPARK4Formal500-full-Seed{seed}-"
        f"{commit_pattern}-gpu*")
    # The second pattern permits deliberate OUT_DIR/name-tag overrides while
    # still requiring the CPAR-K4 method and matched dataset/seed identifiers.
    if commit:
        fallback = (
            f"AML-{dataset}-*CPARK4*full*Seed{seed}*{commit}*-gpu*")
    else:
        fallback = f"AML-{dataset}-*CPARK4*full*Seed{seed}*-gpu*"
    return find_run(root, [stem, fallback], seed, epoch_limit)


def metric(run, key):
    return run.get(key) if run else None


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    default_a2 = repo / "results" / "dmprd_formal500"
    shared_a2 = (
        repo.parent / "FraudGT_dmprd_quickcheck" /
        "results" / "dmprd_formal500")
    if not default_a2.exists() and shared_a2.exists():
        default_a2 = shared_a2
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpar", default=str(repo / "results" / "cpar_k4_formal500"))
    parser.add_argument(
        "--a2", default=str(default_a2))
    parser.add_argument(
        "--commit", default="",
        help="optional short commit prefix embedded in CPAR run names")
    parser.add_argument("--epoch-limit", type=int, default=499)
    parser.add_argument("--min-delta", type=float, default=0.005)
    args = parser.parse_args()

    cpar_root = Path(args.cpar)
    a2_root = Path(args.a2)
    print(
        "dataset\tseed\ta2_last\tcpar_last\ta2_val_f1\tcpar_val_f1"
        "\ta2_selected_test\tcpar_selected_test\tdelta_selected_test"
        "\ta2_raw_best\tcpar_raw_best\tdelta_raw\ta2_val_epoch"
        "\tcpar_val_epoch\ta2_raw_epoch\tcpar_raw_epoch")

    paired = []
    for dataset, seed in TASKS:
        a2 = a2_run(a2_root, dataset, seed, args.epoch_limit)
        cpar = cpar_run(
            cpar_root, dataset, seed, args.epoch_limit, args.commit)
        a2_selected = metric(a2, "test_at_val")
        cpar_selected = metric(cpar, "test_at_val")
        a2_raw = metric(a2, "raw_best")
        cpar_raw = metric(cpar, "raw_best")
        selected_delta = delta(cpar_selected, a2_selected)
        raw_delta = delta(cpar_raw, a2_raw)
        complete = (
            a2 is not None and cpar is not None and
            a2["last_epoch"] >= args.epoch_limit and
            cpar["last_epoch"] >= args.epoch_limit and
            selected_delta is not None)
        if complete:
            paired.append((dataset, selected_delta, raw_delta))
        print("\t".join([
            dataset,
            str(seed),
            str(a2["last_epoch"]) if a2 else "-",
            str(cpar["last_epoch"]) if cpar else "-",
            fmt(metric(a2, "val_f1")),
            fmt(metric(cpar, "val_f1")),
            fmt(a2_selected),
            fmt(cpar_selected),
            fmt(selected_delta),
            fmt(a2_raw),
            fmt(cpar_raw),
            fmt(raw_delta),
            str(a2["val_epoch"]) if a2 else "-",
            str(cpar["val_epoch"]) if cpar else "-",
            str(a2["raw_epoch"]) if a2 else "-",
            str(cpar["raw_epoch"]) if cpar else "-",
        ]))

    print("cpar_k4_pair_go_no_go")
    if paired:
        mean_selected = sum(item[1] for item in paired) / len(paired)
        wins = sum(1 for item in paired if item[1] >= args.min_delta)
        raw_values = [item[2] for item in paired if item[2] is not None]
        mean_raw = (
            sum(raw_values) / len(raw_values) if raw_values else None)
        print(
            f"paired={len(paired)}/2 "
            f"mean_delta_selected_test={mean_selected:+.5f} "
            f"wins_at_plus_{args.min_delta:.3f}={wins}/2 "
            f"mean_delta_raw={fmt(mean_raw)}")
    else:
        wins = 0
        print("paired=0/2")

    formal_complete = args.epoch_limit >= 499 and len(paired) == 2
    if not formal_complete:
        print("formal_pair_screen=INCOMPLETE")
    elif wins == 2:
        print("formal_pair_screen=PASS")
    else:
        print("formal_pair_screen=FAIL")


if __name__ == "__main__":
    main()
