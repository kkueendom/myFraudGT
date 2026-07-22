#!/usr/bin/env python3
"""Audit dynamic-random FraudGT runs against the initial A2 baseline."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def read_rows(path, epoch_limit):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row and int(row["epoch"]) <= epoch_limit:
            rows.append(row)
    return rows


def summarize(seed_dir, epoch_limit):
    train = read_rows(seed_dir / "train" / "stats.json", epoch_limit)
    val = [row for row in read_rows(
        seed_dir / "val" / "stats.json", epoch_limit) if "f1" in row]
    test = [row for row in read_rows(
        seed_dir / "test" / "stats.json", epoch_limit) if "f1" in row]
    if not train or not val or not test:
        return None
    test_by_epoch = {int(row["epoch"]): row for row in test}
    best_val = max(val, key=lambda row: float(row["f1"]))
    selected_epoch = int(best_val["epoch"])
    selected = test_by_epoch.get(selected_epoch)
    if selected is None:
        return None
    raw = max(test, key=lambda row: float(row["f1"]))
    checkpoints = sorted(
        (seed_dir / "ckpt").glob("*.ckpt"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else -1)
    return {
        "last_epoch": int(train[-1]["epoch"]),
        "best_val_f1": float(best_val["f1"]),
        "selected_epoch": selected_epoch,
        "val_selected_test_f1": float(selected["f1"]),
        "raw_best_epoch": int(raw["epoch"]),
        "raw_best_test_f1": float(raw["f1"]),
        "checkpoint": str(checkpoints[-1]) if checkpoints else "not_persisted",
    }


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def git_sha(repo):
    result = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=str(repo),
        text=True, capture_output=True, check=True)
    return result.stdout.strip()


def find_run(root, tag, commit, dataset, seed, epoch_limit):
    pattern = f"AML-{dataset}-{tag}Dynamic500-Seed{seed}-{commit}*-gpu*"
    candidates = []
    for run_dir in sorted(root.glob(pattern)):
        result = summarize(run_dir / str(seed), epoch_limit)
        if result:
            result["run_dir"] = str(run_dir)
            candidates.append(result)
    return max(candidates, key=lambda item: item["last_epoch"]) \
        if candidates else None


def delta_status(delta, threshold):
    if delta >= threshold:
        return "gain"
    if delta <= -threshold:
        return "loss"
    return "sampling_noise_range"


def fmt(value):
    return "-" if value is None else f"{value:.5f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--commit", default=git_sha(repo))
    parser.add_argument("--epoch-limit", type=int, default=499)
    parser.add_argument("--noise-threshold", type=float, default=0.005)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    if spec.get("sampling_protocol") != "dynamic_random":
        raise ValueError("spec must declare sampling_protocol=dynamic_random")
    tag = str(spec["method_tag"])
    config_template = str(spec.get(
        "config_template", "configs/evidence_gate_v4/AML-{dataset}.yaml"))
    root = args.root or repo / "results" / f"{slug(tag)}_{args.commit}_dynamic500"
    baseline = json.loads(
        (repo / "run" / "dynamic_random_a2_baseline.json").read_text())
    baseline = baseline["datasets"]

    header = [
        "dataset", "variant", "seed", "last_epoch", "selected_epoch",
        "val_selected", "a2_val_selected", "delta_val_selected", "val_status",
        "raw_epoch", "raw_best", "a2_raw_best", "delta_raw", "raw_status",
        "sampling_protocol",
    ]
    print("\t".join(header))
    manifests = []
    completed = []
    for dataset, seed in spec["tasks"]:
        result = find_run(
            root, tag, args.commit, dataset, int(seed), args.epoch_limit)
        base = baseline[dataset]
        if result:
            delta_val = (
                result["val_selected_test_f1"] -
                base["val_selected_test_f1"])
            delta_raw = (
                result["raw_best_test_f1"] - base["raw_best_test_f1"])
            completed.append((delta_val, delta_raw))
            manifest = {
                "dataset": dataset,
                "model": tag,
                "variant": spec.get("variant", tag),
                "seed": int(seed),
                "git_commit": args.commit,
                "config": config_template.format(dataset=dataset),
                "checkpoint": result["checkpoint"],
                "selected_epoch": result["selected_epoch"],
                "val_selected_test_f1": result["val_selected_test_f1"],
                "raw_best_epoch": result["raw_best_epoch"],
                "raw_best_test_f1": result["raw_best_test_f1"],
                "delta_val_selected_f1": delta_val,
                "delta_raw_best_f1": delta_raw,
                "sampling_protocol": "dynamic_random",
                "run_dir": result["run_dir"],
            }
            manifests.append(manifest)
            values = [
                dataset, manifest["variant"], str(seed),
                str(result["last_epoch"]), str(result["selected_epoch"]),
                fmt(result["val_selected_test_f1"]),
                fmt(base["val_selected_test_f1"]), fmt(delta_val),
                delta_status(delta_val, args.noise_threshold),
                str(result["raw_best_epoch"]), fmt(result["raw_best_test_f1"]),
                fmt(base["raw_best_test_f1"]), fmt(delta_raw),
                delta_status(delta_raw, args.noise_threshold), "dynamic_random",
            ]
        else:
            values = [dataset, spec.get("variant", tag), str(seed)] + \
                ["-"] * 11 + ["dynamic_random"]
        print("\t".join(values))

    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifests))

    if completed:
        val_deltas = [item[0] for item in completed]
        raw_deltas = [item[1] for item in completed]
        for name, values in (("val_selected", val_deltas), ("raw_best", raw_deltas)):
            wins = sum(value > 0 for value in values)
            failures = sum(value <= 0 for value in values)
            stable = sum(value >= args.noise_threshold for value in values)
            print(
                f"summary metric={name} completed={len(values)}/{len(spec['tasks'])} "
                f"mean_delta={sum(values) / len(values):.5f} wins={wins} "
                f"failures={failures} gains_ge_0.005={stable}")


if __name__ == "__main__":
    main()
