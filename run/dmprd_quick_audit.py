#!/usr/bin/env python3
"""Audit the DMPRD quick screen against matched historical references."""

import argparse
import json
from pathlib import Path


TASKS = [
    ("Small-LI", "a1_single", 42),
    ("Small-LI", "a2_multi", 42),
    ("Small-LI", "a3_full", 42),
    ("Medium-HI", "a3_full", 42),
    ("Large-LI", "a3_full", 44),
    ("Small-HI", "a3_full", 42),
    ("Medium-LI", "a3_full", 44),
    ("Large-HI", "a3_full", 43),
]

BASELINE = {
    "Small-HI": {"raw": 0.78599, "val": 0.76683},
    "Small-LI": {"raw": 0.50474, "val": 0.50355},
    "Medium-HI": {"raw": 0.76768, "val": 0.74601},
    "Medium-LI": {"raw": 0.51852, "val": 0.43231},
    "Large-HI": {"raw": 0.72120, "val": 0.68048},
    "Large-LI": {"raw": 0.41379, "val": 0.35000},
}


def rows(path, epoch_limit=None):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" not in row:
            continue
        if epoch_limit is not None and int(row["epoch"]) > epoch_limit:
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
    selected_epoch = int(best_val["epoch"])
    selected_test = test_by_epoch.get(selected_epoch)
    raw_best = max(
        (row for row in test if "f1" in row),
        key=lambda row: float(row["f1"]))
    return {
        "last_epoch": int(train[-1]["epoch"]),
        "val_epoch": selected_epoch,
        "val_f1": float(best_val["f1"]),
        "test_at_val": (
            float(selected_test["f1"]) if selected_test else None),
        "raw_best": float(raw_best["f1"]),
        "raw_epoch": int(raw_best["epoch"]),
    }


def quick_run(root, dataset, variant, seed, epoch_limit):
    stem = f"AML-{dataset}-DMPRDQuick120-{variant}-Seed{seed}-gpu*"
    for run_dir in sorted(root.glob(stem)):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            result["run_dir"] = str(run_dir)
            return result
    return None


def proto_reference(root, dataset, seed, epoch_limit):
    stem = (
        f"AML-{dataset}-V4BestSeedAblation-proto_only-Seed{seed}-gpu*")
    for run_dir in sorted(root.glob(stem)):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            return result
    return None


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(
            Path(__file__).resolve().parents[1] / "results" / "dmprd_quick120"))
    parser.add_argument(
        "--proto-reference",
        default=(
            "/e/yky/FraudGT_evidence_gate_v4/results/"
            "evidence_gate_v4_best_seed_ablation"))
    parser.add_argument("--epoch-limit", type=int, default=119)
    args = parser.parse_args()

    out = Path(args.out)
    proto_root = Path(args.proto_reference)
    print(
        "dataset\tvariant\tseed\tlast_epoch\ttest_val_select\tval_epoch"
        "\ttest_raw_best\traw_epoch\tproto120_val\tproto120_raw"
        "\tdelta_val_vs_proto120\tdelta_raw_vs_proto120"
        "\tbaseline_final_val\tbaseline_final_raw")
    full_deltas = []
    for dataset, variant, seed in TASKS:
        current = quick_run(
            out, dataset, variant, seed, args.epoch_limit)
        proto = proto_reference(
            proto_root, dataset, seed, args.epoch_limit)
        current_val = current.get("test_at_val") if current else None
        current_raw = current.get("raw_best") if current else None
        proto_val = proto.get("test_at_val") if proto else None
        proto_raw = proto.get("raw_best") if proto else None
        delta_val = (
            current_val - proto_val
            if current_val is not None and proto_val is not None else None)
        delta_raw = (
            current_raw - proto_raw
            if current_raw is not None and proto_raw is not None else None)
        if variant == "a3_full" and delta_val is not None and delta_raw is not None:
            full_deltas.append((dataset, delta_val, delta_raw))
        print("\t".join([
            dataset,
            variant,
            str(seed),
            str(current["last_epoch"]) if current else "-",
            fmt(current_val),
            str(current["val_epoch"]) if current else "-",
            fmt(current_raw),
            str(current["raw_epoch"]) if current else "-",
            fmt(proto_val),
            fmt(proto_raw),
            fmt(delta_val),
            fmt(delta_raw),
            fmt(BASELINE[dataset]["val"]),
            fmt(BASELINE[dataset]["raw"]),
        ]))

    print("dmprd_quick_summary")
    if full_deltas:
        mean_val = sum(item[1] for item in full_deltas) / len(full_deltas)
        mean_raw = sum(item[2] for item in full_deltas) / len(full_deltas)
        raw_bad = sum(1 for item in full_deltas if item[2] < -0.005)
        print(
            f"a3_matched={len(full_deltas)}/6 "
            f"mean_delta_val_vs_proto120={mean_val:+.5f} "
            f"mean_delta_raw_vs_proto120={mean_raw:+.5f} "
            f"raw_losses_below_minus_0.005={raw_bad}")
        if len(full_deltas) == 6:
            passed = mean_val > 0.0 and mean_raw >= 0.0 and raw_bad <= 2
            print(f"six_dataset_go_no_go={'PASS' if passed else 'FAIL'}")
    else:
        print("a3_matched=0/6")


if __name__ == "__main__":
    main()
