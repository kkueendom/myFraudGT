#!/usr/bin/env python3
"""Audit CAMPR full-budget screens against matched A2 references."""

import argparse
import json
from pathlib import Path


FULL_TASKS = [
    ("Large-LI", 44),
    ("Medium-HI", 42),
    ("Small-LI", 42),
]
NO_AUX_TASKS = [
    ("Large-LI", 44),
    ("Small-LI", 42),
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


def campr_run(root, dataset, variant, seed, epoch_limit):
    stem = f"AML-{dataset}-CAMPRFormal500-{variant}-Seed{seed}"
    return find_run(root, stem, seed, epoch_limit)


def complete(run, epoch_limit):
    return run is not None and run["last_epoch"] >= epoch_limit


def metric(run, key):
    return run.get(key) if run else None


def delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def print_full_table(campr_root, a2_root, epoch_limit):
    print(
        "dataset\tseed\ta2_last\tcampr_last\ta2_val_select"
        "\tcampr_val_select\tdelta_val\ta2_raw_best\tcampr_raw_best"
        "\tdelta_raw\ta2_val_epoch\tcampr_val_epoch\ta2_raw_epoch"
        "\tcampr_raw_epoch")
    paired = []
    for dataset, seed in FULL_TASKS:
        a2 = a2_run(a2_root, dataset, seed, epoch_limit)
        current = campr_run(
            campr_root, dataset, "campr_full", seed, epoch_limit)
        a2_val = metric(a2, "test_at_val")
        current_val = metric(current, "test_at_val")
        a2_raw = metric(a2, "raw_best")
        current_raw = metric(current, "raw_best")
        delta_val = delta(current_val, a2_val)
        delta_raw = delta(current_raw, a2_raw)
        if (
                complete(a2, epoch_limit) and
                complete(current, epoch_limit) and
                delta_val is not None and delta_raw is not None):
            paired.append((dataset, delta_val, delta_raw))
        print("\t".join([
            dataset,
            str(seed),
            str(a2["last_epoch"]) if a2 else "-",
            str(current["last_epoch"]) if current else "-",
            fmt(a2_val),
            fmt(current_val),
            fmt(delta_val),
            fmt(a2_raw),
            fmt(current_raw),
            fmt(delta_raw),
            str(a2["val_epoch"]) if a2 else "-",
            str(current["val_epoch"]) if current else "-",
            str(a2["raw_epoch"]) if a2 else "-",
            str(current["raw_epoch"]) if current else "-",
        ]))
    return paired


def print_aux_table(campr_root, epoch_limit):
    print(
        "dataset\tseed\tfull_last\tno_aux_last\tfull_val_select"
        "\tno_aux_val_select\tdelta_val_full_minus_no_aux\tfull_raw_best"
        "\tno_aux_raw_best\tdelta_raw_full_minus_no_aux")
    paired = []
    for dataset, seed in NO_AUX_TASKS:
        full = campr_run(
            campr_root, dataset, "campr_full", seed, epoch_limit)
        no_aux = campr_run(
            campr_root, dataset, "campr_no_aux", seed, epoch_limit)
        full_val = metric(full, "test_at_val")
        no_aux_val = metric(no_aux, "test_at_val")
        full_raw = metric(full, "raw_best")
        no_aux_raw = metric(no_aux, "raw_best")
        delta_val = delta(full_val, no_aux_val)
        delta_raw = delta(full_raw, no_aux_raw)
        if (
                complete(full, epoch_limit) and
                complete(no_aux, epoch_limit) and
                delta_val is not None and delta_raw is not None):
            paired.append((dataset, delta_val, delta_raw))
        print("\t".join([
            dataset,
            str(seed),
            str(full["last_epoch"]) if full else "-",
            str(no_aux["last_epoch"]) if no_aux else "-",
            fmt(full_val),
            fmt(no_aux_val),
            fmt(delta_val),
            fmt(full_raw),
            fmt(no_aux_raw),
            fmt(delta_raw),
        ]))
    return paired


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campr", default=str(repo / "results" / "campr_formal500"))
    parser.add_argument(
        "--a2", default=str(repo / "results" / "dmprd_formal500"))
    parser.add_argument("--epoch-limit", type=int, default=499)
    args = parser.parse_args()
    campr_root = Path(args.campr)
    a2_root = Path(args.a2)

    full_pairs = print_full_table(campr_root, a2_root, args.epoch_limit)
    print("campr_representative_go_no_go")
    if full_pairs:
        mean_val = sum(item[1] for item in full_pairs) / len(full_pairs)
        mean_raw = sum(item[2] for item in full_pairs) / len(full_pairs)
        val_wins = sum(1 for item in full_pairs if item[1] > 0.0)
        raw_wins = sum(1 for item in full_pairs if item[2] > 0.0)
        raw_large_losses = sum(
            1 for item in full_pairs if item[2] < -0.010)
        print(
            f"paired={len(full_pairs)}/3 mean_delta_val={mean_val:+.5f} "
            f"mean_delta_raw={mean_raw:+.5f} val_wins={val_wins}/3 "
            f"raw_wins={raw_wins}/3 "
            f"raw_losses_below_minus_0.010={raw_large_losses}")
        if len(full_pairs) == 3:
            passed = (
                mean_val > 0.005 and val_wins >= 2 and
                mean_raw >= 0.0 and raw_large_losses <= 1)
            print(
                "representative_formal_screen=" +
                ("PASS" if passed else "FAIL"))
        else:
            print("representative_formal_screen=INCOMPLETE")
    else:
        print("paired=0/3 representative_formal_screen=INCOMPLETE")

    print("campr_counterfactual_aux_ablation")
    aux_pairs = print_aux_table(campr_root, args.epoch_limit)
    if aux_pairs:
        mean_val = sum(item[1] for item in aux_pairs) / len(aux_pairs)
        mean_raw = sum(item[2] for item in aux_pairs) / len(aux_pairs)
        val_wins = sum(1 for item in aux_pairs if item[1] > 0.0)
        print(
            f"paired={len(aux_pairs)}/2 "
            f"mean_delta_val_full_minus_no_aux={mean_val:+.5f} "
            f"mean_delta_raw_full_minus_no_aux={mean_raw:+.5f} "
            f"val_wins={val_wins}/2")
        if len(aux_pairs) < 2:
            print("counterfactual_aux_screen=INCOMPLETE")
        elif mean_val > 0.0 and val_wins >= 1:
            print("counterfactual_aux_screen=SUPPORTED")
        else:
            print("counterfactual_aux_screen=NOT_SUPPORTED")
    else:
        print("paired=0/2 counterfactual_aux_screen=INCOMPLETE")


if __name__ == "__main__":
    main()
