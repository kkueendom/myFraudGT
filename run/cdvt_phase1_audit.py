#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


INITIAL_A2 = {
    "Small-LI": {"val_selected": 0.46247, "raw_best": 0.50667},
    "Large-LI": {"val_selected": 0.30108, "raw_best": 0.44720},
}
DATASETS = ("Small-LI", "Large-LI")
VARIANTS = ("account_only", "event_only", "dual_view", "full_cdvt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def load_manifests(root, expected_commit=""):
    rows = {}
    for path in sorted(root.glob("*/manifest.json")):
        payload = json.loads(path.read_text())
        key = (payload.get("dataset"), payload.get("variant"))
        if key in rows:
            raise ValueError(f"duplicate Phase 1 result: {key}")
        if payload.get("sampling_protocol") != "dynamic_random":
            raise ValueError(f"{path}: sampling protocol differs")
        if int(payload.get("seed", -1)) != 42:
            raise ValueError(f"{path}: seed differs from Phase 1 seed 42")
        commit = str(payload.get("git_commit", ""))
        if expected_commit and not commit.startswith(expected_commit):
            raise ValueError(f"{path}: commit {commit} differs")
        rows[key] = (payload, path)
    expected = {
        (dataset, variant)
        for dataset in DATASETS
        for variant in VARIANTS
    }
    missing = expected - set(rows)
    extra = set(rows) - expected
    if missing or extra:
        raise ValueError(
            f"Phase 1 matrix differs: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")
    return rows


def supporting_events(payload, baseline):
    trajectory_path = Path(payload["checkpoint"]).parent / "trajectory.jsonl"
    events = [
        json.loads(line) for line in trajectory_path.read_text().splitlines()
        if line.strip()
    ]
    selected_epoch = int(payload["val_selected_epoch"])
    selected = next(
        event for event in events if int(event["epoch"]) == selected_epoch)
    selected_val = float(selected["val_f1"])
    support = [
        event for event in events
        if int(event["epoch"]) != selected_epoch
        and float(event["val_f1"]) >= selected_val - 0.01
        and float(event["test"]["normal"]["f1"]) > baseline
    ]
    return len(support)


def audit(root, expected_commit=""):
    manifests = load_manifests(root, expected_commit)
    table = []
    dataset_gates = {}
    for dataset in DATASETS:
        baseline = INITIAL_A2[dataset]["val_selected"]
        account = manifests[(dataset, "account_only")][0]
        account_f1 = float(account["val_selected_test_f1"])
        for variant in VARIANTS:
            payload = manifests[(dataset, variant)][0]
            best = payload["best_event"]
            normal = float(best["test"]["normal"]["f1"])
            shuffled = float(best["test"]["shuffled"]["f1"])
            off = float(best["test"]["off"]["f1"])
            table.append({
                "dataset": dataset,
                "variant": variant,
                "val_selected_test_f1": normal,
                "delta_vs_initial_a2": normal - baseline,
                "delta_vs_matched_account": normal - account_f1,
                "normal_minus_shuffled": normal - shuffled,
                "normal_minus_off": normal - off,
                "selected_epoch": int(payload["val_selected_epoch"]),
                "raw_best_test_f1": float(payload["raw_best_test_f1"]),
                "parameter_count": int(payload["parameter_count"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
            })

        full = manifests[(dataset, "full_cdvt")][0]
        full_best = full["best_event"]
        full_normal = float(full_best["test"]["normal"]["f1"])
        full_shuffled = float(full_best["test"]["shuffled"]["f1"])
        full_off = float(full_best["test"]["off"]["f1"])
        delta_initial = full_normal - baseline
        delta_account = full_normal - account_f1
        mechanism_margin = full_normal - max(full_shuffled, full_off)
        support = supporting_events(full, baseline)
        dataset_gates[dataset] = {
            "full_f1": full_normal,
            "initial_a2_f1": baseline,
            "matched_account_f1": account_f1,
            "delta_vs_initial_a2": delta_initial,
            "delta_vs_matched_account": delta_account,
            "mechanism_margin": mechanism_margin,
            "supporting_evaluation_events": support,
            "beats_initial_a2": delta_initial > 0,
            "beats_matched_account": delta_account > 0,
            "normal_beats_shuffled_off_by_0_01": mechanism_margin >= 0.01,
            "not_single_anomalous_event": support >= 1,
        }

    full_deltas = [
        dataset_gates[dataset]["delta_vs_initial_a2"]
        for dataset in DATASETS
    ]
    advancement = {
        "full_beats_initial_a2_both": all(
            dataset_gates[dataset]["beats_initial_a2"]
            for dataset in DATASETS),
        "full_beats_matched_account_both": all(
            dataset_gates[dataset]["beats_matched_account"]
            for dataset in DATASETS),
        "at_least_one_gain_ge_0_01": max(full_deltas) >= 0.01,
        "normal_beats_shuffled_off_both": all(
            dataset_gates[dataset]["normal_beats_shuffled_off_by_0_01"]
            for dataset in DATASETS),
        "not_single_anomalous_event_both": all(
            dataset_gates[dataset]["not_single_anomalous_event"]
            for dataset in DATASETS),
    }
    advancement["advance_to_phase2"] = all(advancement.values())
    return {
        "sampling_protocol": "dynamic_random",
        "root": str(root.resolve()),
        "table": table,
        "dataset_gates": dataset_gates,
        "advancement": advancement,
    }


def main():
    args = parse_args()
    result = audit(args.root, args.expected_commit)
    if args.write:
        output = args.root / "phase1_audit.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

