#!/usr/bin/env python3
"""Formal validation and aggregation for directional GTF1C v2."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from run.gtf1c_phase0_real_graph import (
    NEGATIVE_REGIMES,
    POSITIVE_REGIMES,
    REGIMES,
)


METHODS = (
    "row_net",
    "gtprc_row_harm",
    "iid_paired_f1",
    "time_block_paired_f1",
    "gtf1c_graph_time",
)


def load_manifests(root):
    paths = sorted(root.glob("*/gtf1c_phase0_manifest.json"))
    return [json.loads(path.read_text()) for path in paths], paths


def validate(manifests, expected_replicates):
    if len(manifests) != len(REGIMES):
        raise ValueError("expected seven formal GTF1C manifests")
    if {row["regime"] for row in manifests} != set(REGIMES):
        raise ValueError("formal GTF1C regime set is incomplete")
    if len({row["git_commit"] for row in manifests}) != 1:
        raise ValueError("formal GTF1C tasks use different commits")
    if {row["seed"] for row in manifests} != set(range(74001, 74008)):
        raise ValueError("formal GTF1C seeds do not match registration")
    for row in manifests:
        if row.get("method_version") != 2:
            raise ValueError("non-v2 GTF1C manifest")
        if row["replicates"] != int(expected_replicates):
            raise ValueError("unexpected formal replicate count")
        if row["sampling_protocol"] != "dynamic_random":
            raise ValueError("non-dynamic formal manifest")
        if row["validation_loader_iterations"] != 0:
            raise ValueError("formal GTF1C opened validation loader")
        if row["test_loader_iterations"] != 0:
            raise ValueError("formal GTF1C opened test loader")
        if row["candidate_delta"] != row["delta"] / 3.0:
            raise ValueError("candidate multiplicity correction changed")
        if set(row["datasets"]) != {"Small-LI", "Large-LI"}:
            raise ValueError("formal dataset set is incomplete")
        for dataset in row["datasets"].values():
            if dataset["trials"] != 3 * int(expected_replicates):
                raise ValueError("unexpected formal rotation count")


def _trial_path(manifest, manifest_path):
    declared = Path(manifest["trial_records"])
    if declared.exists():
        return declared
    fallback = manifest_path.parent / "gtf1c_v2_trials.jsonl"
    if not fallback.exists():
        raise FileNotFoundError("formal GTF1C trial records missing")
    return fallback


def _median(values):
    return statistics.median(values) if values else 0.0


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def trial_diagnostics(records, dataset, method):
    rows = [row for row in records if row["dataset"] == dataset]
    qualified = [
        row for row in rows
        if row["chosen_candidate"][method] >= 0
    ]
    deltas = [
        row["normal_evaluations"][method]["paired_f1_delta"]
        for row in qualified
    ]
    changes = [
        row["normal_evaluations"][method]["changed"]
        for row in qualified
    ]
    fractions = [
        min(
            1.0,
            row["normal_evaluations"][method]["changed"]
            / row["oracle_changed"],
        )
        for row in qualified
        if row["oracle_changed"] > 0
    ]
    by_fold = {}
    for fold in range(3):
        fold_rows = [
            row for row in rows if row["evaluation_fold"] == fold]
        fold_qualified = [
            row for row in fold_rows
            if row["chosen_candidate"][method] >= 0]
        by_fold[str(fold)] = {
            "trials": len(fold_rows),
            "qualified": len(fold_qualified),
            "violations": sum(
                row["normal_evaluations"][method][
                    "paired_f1_delta"
                ] <= 0.0
                for row in fold_qualified
            ),
            "practical_failures": sum(
                row["normal_evaluations"][method][
                    "paired_f1_delta"
                ] < 0.005
                for row in fold_qualified
            ),
        }

    by_replicate = defaultdict(list)
    for row in rows:
        by_replicate[row["replicate"]].append(row)
    replicate_qualified = 0
    replicate_violations = 0
    replicate_practical_failures = 0
    for replicate_rows in by_replicate.values():
        selected = [
            row for row in replicate_rows
            if row["chosen_candidate"][method] >= 0]
        replicate_qualified += int(bool(selected))
        replicate_violations += int(any(
            row["normal_evaluations"][method][
                "paired_f1_delta"
            ] <= 0.0
            for row in selected
        ))
        replicate_practical_failures += int(any(
            row["normal_evaluations"][method][
                "paired_f1_delta"
            ] < 0.005
            for row in selected
        ))
    replicate_count = len(by_replicate)
    return {
        "qualified_trial_count": len(qualified),
        "conditional_delta_mean": _mean(deltas),
        "conditional_delta_median": _median(deltas),
        "conditional_changes_mean": _mean(changes),
        "conditional_changes_median": _median(changes),
        "conditional_oracle_fraction_mean": _mean(fractions),
        "conditional_oracle_fraction_median": _median(fractions),
        "qualified_with_oracle_count": len(fractions),
        "by_fold": by_fold,
        "replicate_count": replicate_count,
        "replicate_any_qualification_rate": (
            replicate_qualified / max(replicate_count, 1)),
        "replicate_any_violation_rate": (
            replicate_violations / max(replicate_count, 1)),
        "replicate_any_practical_failure_rate": (
            replicate_practical_failures / max(replicate_count, 1)),
    }


def build_aggregate(manifests, paths, expected_replicates):
    validate(manifests, expected_replicates)
    units = {}
    for manifest, path in sorted(
            zip(manifests, paths), key=lambda item: item[0]["regime"]):
        regime = manifest["regime"]
        records = [
            json.loads(line)
            for line in _trial_path(manifest, path).read_text().splitlines()
            if line
        ]
        expected_records = 2 * 3 * int(expected_replicates)
        if len(records) != expected_records:
            raise ValueError("unexpected formal trial-record count")
        units[regime] = {}
        for dataset, row in sorted(manifest["datasets"].items()):
            methods = {}
            for method in METHODS:
                methods[method] = {
                    **row["methods"][method],
                    "practical_failure_rate": (
                        row["methods"][method]["practical_failures"]
                        / row["methods"][method]["trials"]
                    ),
                    "conditional": trial_diagnostics(
                        records, dataset, method),
                }
            units[regime][dataset] = {
                "oracle_opportunity_rate": row[
                    "oracle_opportunity_rate"],
                "methods": methods,
                "controls": row["controls"],
            }

    negative_units = [
        units[regime][dataset]
        for regime in NEGATIVE_REGIMES
        for dataset in ("Small-LI", "Large-LI")
    ]
    normal_units = [
        units[regime][dataset]
        for regime in REGIMES
        for dataset in ("Small-LI", "Large-LI")
    ]
    positive_units = {
        dataset: [
            units[regime][dataset]
            for regime in POSITIVE_REGIMES
        ]
        for dataset in ("Small-LI", "Large-LI")
    }
    exposed_negative_units = sum(
        max(
            row["methods"]["row_net"]["false_qualification_rate"],
            row["methods"]["gtprc_row_harm"][
                "false_qualification_rate"],
        ) > 0.10
        for row in negative_units
    )
    positive_power = {
        dataset: sum(
            row["methods"]["gtf1c_graph_time"][
                "qualification_rate"
            ] >= 0.25
            for row in rows
        )
        for dataset, rows in positive_units.items()
    }
    conditional_coverage = {
        dataset: sum(
            row["methods"]["gtf1c_graph_time"]["conditional"][
                "conditional_oracle_fraction_median"
            ] >= 0.30
            for row in rows
        )
        for dataset, rows in positive_units.items()
    }
    iid_retention = {}
    for dataset in ("Small-LI", "Large-LI"):
        row = units["iid_positive"][dataset]["methods"]
        iid_rate = row["iid_paired_f1"]["qualification_rate"]
        gtf1c_rate = row["gtf1c_graph_time"]["qualification_rate"]
        iid_retention[dataset] = (
            gtf1c_rate / iid_rate if iid_rate > 0.0 else 0.0)

    gate = {
        "row_harm_exposes_two_negative_units": (
            exposed_negative_units >= 2),
        "gtf1c_controls_all_normal_units": all(
            row["methods"]["gtf1c_graph_time"][
                "false_qualification_rate"
            ] <= 0.07
            for row in normal_units
        ),
        "controls_rejected_all_units": all(
            condition["qualification_rate"] <= 0.05
            for row in normal_units
            for condition in row["controls"].values()
        ),
        "practical_failure_control_all_positive_units": all(
            row["methods"]["gtf1c_graph_time"][
                "practical_failure_rate"
            ] <= 0.10
            for rows in positive_units.values()
            for row in rows
        ),
        "small_positive_power_three_regimes": (
            positive_power["Small-LI"] >= 3),
        "large_positive_power_two_regimes": (
            positive_power["Large-LI"] >= 2),
        "small_conditional_coverage_three_regimes": (
            conditional_coverage["Small-LI"] >= 3),
        "large_conditional_coverage_two_regimes": (
            conditional_coverage["Large-LI"] >= 2),
        "iid_power_retention_both_datasets": all(
            value >= 0.80 for value in iid_retention.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "experiment": "GTF1C_Phase0_v2_directional",
        "mode": "formal",
        "git_commit": manifests[0]["git_commit"],
        "manifest_count": len(manifests),
        "replicates_per_regime": int(expected_replicates),
        "trial_count": sum(
            row["datasets"][dataset]["trials"]
            for row in manifests
            for dataset in ("Small-LI", "Large-LI")
        ),
        "exposed_negative_units": exposed_negative_units,
        "positive_power_regimes": positive_power,
        "conditional_coverage_regimes": conditional_coverage,
        "iid_power_retention": iid_retention,
        "units": units,
        "gate": gate,
        "decision": (
            "PROCEED_TO_THEORY_AND_REAL_EVIDENCE_AUDIT"
            if gate["passed"]
            else "STOP_GTF1C"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-replicates", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifests, paths = load_manifests(args.input_root)
    aggregate = build_aggregate(
        manifests, paths, args.expected_replicates)
    aggregate["manifest_paths"] = [str(path) for path in paths]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output_json),
        "gate": aggregate["gate"],
        "decision": aggregate["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
