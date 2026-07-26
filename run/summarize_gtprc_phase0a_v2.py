#!/usr/bin/env python3
"""Validate GTPRC v2 development or formal boundary-stress results."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.gtprc_phase0a_simulation import V2_REGIMES


def load_manifests(root):
    paths = sorted(root.glob("*/gtprc_phase0a_manifest.json"))
    return [json.loads(path.read_text()) for path in paths], paths


def validate(manifests, expected_replicates):
    if len(manifests) != len(V2_REGIMES):
        raise ValueError("expected seven v2 manifests")
    if {row["regime"] for row in manifests} != set(V2_REGIMES):
        raise ValueError("v2 regime set is incomplete or duplicated")
    if len({row["git_commit"] for row in manifests}) != 1:
        raise ValueError("v2 tasks use different commits")
    for row in manifests:
        if row.get("stress_version") != 2:
            raise ValueError("non-v2 manifest detected")
        if row["replicates"] != expected_replicates:
            raise ValueError("unexpected replicate count")
        if row["alpha"] != 0.40 or row["delta"] != 0.05:
            raise ValueError("registered alpha/delta changed")


def build_aggregate(manifests, expected_replicates, mode):
    validate(manifests, expected_replicates)
    regimes = {}
    dependent = []
    for manifest in sorted(manifests, key=lambda row: row["regime"]):
        name = manifest["regime"]
        row = {
            "methods": manifest["methods"],
            "controls": manifest["controls"],
        }
        row_iid = manifest["methods"]["row_iid"]
        gtprc = manifest["methods"]["gtprc_graph_time"]
        row["row_iid_violation_over_0_10"] = (
            row_iid["violation_rate"] > 0.10)
        row["gtprc_violation_at_most_0_07"] = (
            gtprc["violation_rate"] <= 0.07)
        row["gtprc_oracle_fraction_at_least_0_30"] = (
            gtprc["oracle_fraction_median"] >= 0.30)
        row["coverage_difference_at_least_0_02"] = (
            abs(
                row_iid["coverage_median"]
                - gtprc["coverage_median"]
            ) >= 0.02
        )
        row["controls_pass"] = all(
            item["false_qualification_rate"] <= 0.05
            for item in manifest["controls"].values()
        )
        row["gtprc_improvement_at_least_0_05"] = (
            row_iid["violation_rate"] - gtprc["violation_rate"]
            >= 0.05
        )
        regimes[name] = row
        if name != "iid":
            dependent.append(row)

    iid_row = regimes["iid"]["methods"]["row_iid"]
    iid_gtprc = regimes["iid"]["methods"]["gtprc_graph_time"]
    iid_retention = (
        iid_gtprc["coverage_median"]
        / max(iid_row["coverage_median"], 1e-12)
    )
    gate = {
        "row_iid_fails_three_dependent_regimes": sum(
            row["row_iid_violation_over_0_10"]
            for row in dependent
        ) >= 3,
        "gtprc_controls_all_regimes": all(
            row["gtprc_violation_at_most_0_07"]
            for row in regimes.values()
        ),
        "gtprc_retains_oracle_coverage_five_regimes": sum(
            row["gtprc_oracle_fraction_at_least_0_30"]
            for row in regimes.values()
        ) >= 5,
        "iid_coverage_retention": iid_retention >= 0.80,
        "controls_rejected_all_regimes": all(
            row["controls_pass"] for row in regimes.values()
        ),
        "coverage_differs_three_dependent_regimes": sum(
            row["coverage_difference_at_least_0_02"]
            for row in dependent
        ) >= 3,
    }
    if mode == "formal":
        gate["gtprc_improves_three_dependent_regimes"] = sum(
            row["gtprc_improvement_at_least_0_05"]
            for row in dependent
        ) >= 3
    gate["passed"] = all(gate.values())
    return {
        "experiment": "gtprc_phase0a_v2_boundary_stress",
        "mode": mode,
        "git_commit": manifests[0]["git_commit"],
        "manifest_count": len(manifests),
        "replicate_count": sum(
            row["replicates"] for row in manifests),
        "iid_coverage_retention": iid_retention,
        "regimes": regimes,
        "gate": gate,
        "decision": (
            "PROCEED_TO_FORMAL_V2"
            if mode == "development" and gate["passed"]
            else "PROCEED_TO_PHASE0B"
            if mode == "formal" and gate["passed"]
            else "REDESIGN_STRESS_TEST"
            if mode == "development"
            else "STOP_OR_REDESIGN_GTPRC"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--expected-replicates", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifests, paths = load_manifests(args.input_root)
    aggregate = build_aggregate(
        manifests, args.expected_replicates, args.mode)
    aggregate["manifest_paths"] = [str(path) for path in paths]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output_json),
        "gate": aggregate["gate"],
        "decision": aggregate["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

