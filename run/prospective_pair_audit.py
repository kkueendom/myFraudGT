#!/usr/bin/env python3
"""Audit one prospective pair, optionally against a fixed-panel A2 pair."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def json_rows(path, epoch_limit):
    output = []
    if not path.exists():
        return output
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row and int(row["epoch"]) <= epoch_limit:
            output.append(row)
    return output


def summarize(seed_dir, epoch_limit):
    train = json_rows(seed_dir / "train" / "stats.json", epoch_limit)
    val = json_rows(seed_dir / "val" / "stats.json", epoch_limit)
    test = json_rows(seed_dir / "test" / "stats.json", epoch_limit)
    if not train or not val or not test:
        return None
    test_by_epoch = {int(row["epoch"]): row for row in test if "f1" in row}
    val = [row for row in val if "f1" in row]
    test = [row for row in test if "f1" in row]
    if not val or not test:
        return None
    best_val = max(val, key=lambda row: float(row["f1"]))
    val_epoch = int(best_val["epoch"])
    selected = test_by_epoch.get(val_epoch)
    raw = max(test, key=lambda row: float(row["f1"]))
    return {
        "last": int(train[-1]["epoch"]),
        "val_epoch": val_epoch,
        "val_f1": float(best_val["f1"]),
        "selected": float(selected["f1"]) if selected else None,
        "raw_epoch": int(raw["epoch"]),
        "raw": float(raw["f1"]),
    }


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_run(root, method_tag, commit, dataset, seed, epoch_limit):
    pattern = (
        f"AML-{dataset}-{method_tag}Pair500-Seed{seed}-{commit}*-gpu*")
    candidates = []
    for run_dir in sorted(root.glob(pattern)):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            result["run_dir"] = str(run_dir)
            candidates.append(result)
    return max(candidates, key=lambda item: item["last"]) if candidates else None


def git_sha(repo):
    result = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=str(repo),
        text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def subtract(left, right):
    if left is None or right is None:
        return None
    return left - right


def main():
    repo = Path(__file__).resolve().parents[1]
    spec = json.loads((repo / "run" / "prospective_pair_spec.json").read_text())
    current_tag = str(spec["method_tag"])
    current_commit = git_sha(repo)
    default_root = (
        repo / "results" /
        f"{slug(current_tag)}_{current_commit}_pair500")

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--commit", default=current_commit)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-tag", default="FixedEvalA2")
    parser.add_argument("--reference-commit", default="")
    parser.add_argument("--epoch-limit", type=int, default=499)
    parser.add_argument("--min-delta", type=float, default=0.005)
    args = parser.parse_args()

    header = [
        "dataset", "seed", "current_last", "current_val_f1",
        "current_selected", "current_raw", "current_val_epoch",
        "current_raw_epoch",
    ]
    if args.reference_root:
        header += [
            "a2_last", "a2_selected", "delta_selected", "a2_raw",
            "delta_raw", "a2_val_epoch", "a2_raw_epoch",
        ]
    print("\t".join(header))

    pairs = []
    for dataset, seed in spec["tasks"]:
        current = find_run(
            args.root, current_tag, args.commit, dataset, int(seed),
            args.epoch_limit)
        values = [
            dataset,
            str(seed),
            str(current["last"]) if current else "-",
            fmt(current.get("val_f1") if current else None),
            fmt(current.get("selected") if current else None),
            fmt(current.get("raw") if current else None),
            str(current["val_epoch"]) if current else "-",
            str(current["raw_epoch"]) if current else "-",
        ]
        if args.reference_root:
            reference = find_run(
                args.reference_root, args.reference_tag,
                args.reference_commit, dataset, int(seed), args.epoch_limit)
            selected_delta = subtract(
                current.get("selected") if current else None,
                reference.get("selected") if reference else None)
            raw_delta = subtract(
                current.get("raw") if current else None,
                reference.get("raw") if reference else None)
            values += [
                str(reference["last"]) if reference else "-",
                fmt(reference.get("selected") if reference else None),
                fmt(selected_delta),
                fmt(reference.get("raw") if reference else None),
                fmt(raw_delta),
                str(reference["val_epoch"]) if reference else "-",
                str(reference["raw_epoch"]) if reference else "-",
            ]
            if (
                current and reference and
                current["last"] >= args.epoch_limit and
                reference["last"] >= args.epoch_limit and
                selected_delta is not None
            ):
                pairs.append((selected_delta, raw_delta))
        print("\t".join(values))

    if args.reference_root:
        wins = sum(delta >= args.min_delta for delta, _ in pairs)
        mean_delta = (
            sum(delta for delta, _ in pairs) / len(pairs) if pairs else None)
        raw = [delta for _, delta in pairs if delta is not None]
        mean_raw = sum(raw) / len(raw) if raw else None
        print(
            f"formal_pairs={len(pairs)}/2 mean_selected={fmt(mean_delta)} "
            f"wins_at_plus_{args.min_delta:.3f}={wins}/2 "
            f"mean_raw={fmt(mean_raw)}")
        complete = args.epoch_limit >= 499 and len(pairs) == 2
        verdict = "PASS" if complete and wins == 2 else (
            "FAIL" if complete else "INCOMPLETE")
        print(f"prospective_pair_screen={verdict}")


if __name__ == "__main__":
    main()
