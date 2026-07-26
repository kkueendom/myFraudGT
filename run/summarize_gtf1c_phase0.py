#!/usr/bin/env python3
"""Validate and aggregate GTF1C Phase 0 development results."""

import argparse
import json
from pathlib import Path

from run.gtf1c_phase0_real_graph import (
    NEGATIVE_REGIMES,
    POSITIVE_REGIMES,
    REGIMES,
)


def load_manifests(root):
    paths = sorted(root.glob("*/gtf1c_phase0_manifest.json"))
    return [json.loads(path.read_text()) for path in paths], paths


def validate(manifests, expected_replicates):
    if len(manifests) != len(REGIMES):
        raise ValueError("expected seven GTF1C manifests")
    if {row["regime"] for row in manifests} != set(REGIMES):
        raise ValueError("GTF1C regime set is incomplete or duplicated")
    if len({row["git_commit"] for row in manifests}) != 1:
        raise ValueError("GTF1C tasks use different commits")
    for row in manifests:
        if row["replicates"] != int(expected_replicates):
            raise ValueError("unexpected GTF1C replicate count")
        if row["sampling_protocol"] != "dynamic_random":
            raise ValueError("non-dynamic GTF1C manifest")
        if row["validation_loader_iterations"] != 0:
            raise ValueError("GTF1C opened validation loader")
        if row["test_loader_iterations"] != 0:
            raise ValueError("GTF1C opened test loader")
        if set(row["datasets"]) != {"Small-LI", "Large-LI"}:
            raise ValueError("GTF1C dataset set is incomplete")
        for dataset in row["datasets"].values():
            if dataset["trials"] != 3 * int(expected_replicates):
                raise ValueError("unexpected fold-rotation trial count")


def build_aggregate(manifests, expected_replicates):
    validate(manifests, expected_replicates)
    by_regime = {row["regime"]: row for row in manifests}
    units = {}
    for regime, manifest in sorted(by_regime.items()):
        units[regime] = {}
        for dataset, row in sorted(manifest["datasets"].items()):
            methods = row["methods"]
            controls = row["controls"]
            units[regime][dataset] = {
                "oracle_opportunity_rate": row[
                    "oracle_opportunity_rate"],
                "row_net": methods["row_net"],
                "gtprc_row_harm": methods["gtprc_row_harm"],
                "iid_paired_f1": methods["iid_paired_f1"],
                "time_block_paired_f1": methods[
                    "time_block_paired_f1"],
                "gtf1c_graph_time": methods["gtf1c_graph_time"],
                "controls": controls,
            }

    positive_units = [
        units[regime][dataset]
        for regime in sorted(POSITIVE_REGIMES)
        for dataset in ("Small-LI", "Large-LI")
    ]
    negative_units = [
        units[regime][dataset]
        for regime in sorted(NEGATIVE_REGIMES)
        for dataset in ("Small-LI", "Large-LI")
    ]
    exposed_negative_units = sum(
        max(
            row["row_net"]["false_qualification_rate"],
            row["gtprc_row_harm"]["false_qualification_rate"],
        ) > 0.10
        for row in negative_units
    )
    positive_power = {
        dataset: sum(
            units[regime][dataset]["gtf1c_graph_time"][
                "qualification_rate"
            ] >= 0.25
            for regime in POSITIVE_REGIMES
        )
        for dataset in ("Small-LI", "Large-LI")
    }
    coverage_units = sum(
        row["gtf1c_graph_time"]["oracle_fraction_median"] >= 0.30
        for row in positive_units
    )
    iid_retention = {}
    for dataset in ("Small-LI", "Large-LI"):
        row = units["iid_positive"][dataset]
        iid_rate = row["iid_paired_f1"]["qualification_rate"]
        gtf1c_rate = row["gtf1c_graph_time"]["qualification_rate"]
        iid_retention[dataset] = (
            gtf1c_rate / iid_rate if iid_rate > 0.0 else 0.0)

    gate = {
        "positive_oracle_opportunities_all_units": all(
            row["oracle_opportunity_rate"] > 0.0
            for row in positive_units
        ),
        "row_harm_exposes_two_negative_units": (
            exposed_negative_units >= 2),
        "gtf1c_controls_all_negative_units": all(
            row["gtf1c_graph_time"]["false_qualification_rate"]
            <= 0.07
            for row in negative_units
        ),
        "controls_rejected_all_units": all(
            condition["qualification_rate"] <= 0.05
            for regime in units.values()
            for row in regime.values()
            for condition in row["controls"].values()
        ),
        "small_positive_power_three_regimes": (
            positive_power["Small-LI"] >= 3),
        "large_positive_power_two_regimes": (
            positive_power["Large-LI"] >= 2),
        "oracle_coverage_three_units": coverage_units >= 3,
        "iid_power_retention_both_datasets": all(
            value >= 0.80 for value in iid_retention.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "experiment": "GTF1C_Phase0_real_graph_stress",
        "mode": "development",
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
        "oracle_coverage_units": coverage_units,
        "iid_power_retention": iid_retention,
        "units": units,
        "gate": gate,
        "decision": (
            "PROCEED_TO_FORMAL_PREREGISTRATION"
            if gate["passed"]
            else "REDESIGN_OR_STOP_GTF1C"
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
    aggregate = build_aggregate(manifests, args.expected_replicates)
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
