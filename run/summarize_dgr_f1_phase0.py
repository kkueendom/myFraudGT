#!/usr/bin/env python3
"""Validate and gate the DGR-F1 controlled development screen."""

import argparse
import json
from pathlib import Path

from fraudGT.evidence.dgr_f1 import (
    DECISION_HARM,
    DECISION_IMPROVEMENT,
    DECISION_INSUFFICIENT,
)
from run.dgr_f1_phase0 import SCENARIO_TARGETS


POSITIVE_SCENARIOS = (
    "stable_iid_improvement",
    "stable_graph_improvement",
    "stable_temporal_improvement",
)
UNSTABLE_SCENARIOS = (
    "positive_mean_low_replication",
    "base_ratio_sign_reversal",
)
TEMPLATES = ("Small-LI", "Large-LI")


def load_manifests(root):
    paths = sorted(root.glob("*/dgr_f1_phase0_manifest.json"))
    return [json.loads(path.read_text()) for path in paths], paths


def validate(manifests, expected_replicates):
    if len(manifests) != 7:
        raise ValueError("expected seven DGR-F1 manifests")
    if {row["scenario"] for row in manifests} != set(SCENARIO_TARGETS):
        raise ValueError("DGR-F1 scenario set is incomplete")
    if len({row["git_commit"] for row in manifests}) != 1:
        raise ValueError("DGR-F1 tasks use different commits")
    if {row["seed"] for row in manifests} != set(range(76001, 76008)):
        raise ValueError("DGR-F1 task seeds do not match registration")
    for row in manifests:
        if row["experiment"] != "DGR_F1_Phase0_controlled_development":
            raise ValueError("unexpected DGR-F1 experiment")
        if row["method_version"] != 1:
            raise ValueError("unexpected DGR-F1 method version")
        if row["replicates"] != int(expected_replicates):
            raise ValueError("unexpected DGR-F1 replicate count")
        if row["streams_per_replicate"] != 64:
            raise ValueError("DGR-F1 stream budget changed")
        if row["checkpoints"] != [8, 16, 32, 64]:
            raise ValueError("DGR-F1 checkpoints changed")
        if row["epsilon"] != 0.005:
            raise ValueError("DGR-F1 practical margin changed")
        if row["pi0"] != 0.75:
            raise ValueError("DGR-F1 replication target changed")
        if row["delta"] != 0.05:
            raise ValueError("DGR-F1 family delta changed")
        if row["endpoint_delta"] != row["delta"] / 12:
            raise ValueError("DGR-F1 endpoint correction changed")
        if row["validation_loader_iterations"] != 0:
            raise ValueError("DGR-F1 opened validation loader")
        if row["test_loader_iterations"] != 0:
            raise ValueError("DGR-F1 opened test loader")
        if set(row["templates"]) != set(TEMPLATES):
            raise ValueError("DGR-F1 template set is incomplete")
        if row["registered_target"] != SCENARIO_TARGETS[row["scenario"]]:
            raise ValueError("registered DGR-F1 target changed")
        for template in row["templates"].values():
            if (
                template["population_target"]["decision"]
                != template["registered_target"]
            ):
                raise ValueError("scenario generator target mismatch")


def build_aggregate(manifests, expected_replicates):
    validate(manifests, expected_replicates)
    units = {
        row["scenario"]: row["templates"]
        for row in manifests
    }
    identity_failures = sum(
        template["identity_failures"]
        + template["reference_identity_failures"]
        for row in units.values()
        for template in row.values()
    )

    false_improvement_control = all(
        units[scenario][template]["methods"]["dgr_f1"][
            "improvement_rate"
        ] <= 0.05
        for scenario in ("null_intervention", "stable_harm")
        for template in TEMPLATES
    )
    false_harm_control = all(
        units[scenario][template]["methods"]["dgr_f1"][
            "harm_rate"
        ] <= 0.05
        for scenario in POSITIVE_SCENARIOS + ("null_intervention",)
        for template in TEMPLATES
    )
    positive_power = {
        template: {
            scenario: units[scenario][template]["methods"]["dgr_f1"][
                "improvement_rate"
            ]
            for scenario in POSITIVE_SCENARIOS
        }
        for template in TEMPLATES
    }
    positive_power_pass = all(
        rate >= 0.80
        for row in positive_power.values()
        for rate in row.values()
    )
    iid_stopping_pass = all(
        units["stable_iid_improvement"][template]["methods"]["dgr_f1"][
            "median_stopping_checkpoint"
        ] is not None
        and units["stable_iid_improvement"][template]["methods"]["dgr_f1"][
            "median_stopping_checkpoint"
        ] <= 32
        for template in TEMPLATES
    )
    unstable_abstention = {
        template: {
            scenario: units[scenario][template]["methods"]["dgr_f1"][
                "insufficient_rate"
            ]
            for scenario in UNSTABLE_SCENARIOS
        }
        for template in TEMPLATES
    }
    unstable_abstention_pass = all(
        rate >= 0.90
        for row in unstable_abstention.values()
        for rate in row.values()
    )
    unstable_naive_claim_rates = {
        template: {
            scenario: max(
                1.0 - units[scenario][template]["methods"][method][
                    "insufficient_rate"
                ]
                for method in ("one_stream", "mean_only")
            )
            for scenario in UNSTABLE_SCENARIOS
        }
        for template in TEMPLATES
    }
    nontrivial_benchmark = any(
        rate >= 0.15
        for row in unstable_naive_claim_rates.values()
        for rate in row.values()
    )
    coverage = {
        template: {
            scenario: units[scenario][template][
                "simultaneous_mean_coverage_rate"
            ]
            for scenario in (
                "null_intervention", "base_ratio_sign_reversal")
        }
        for template in TEMPLATES
    }
    coverage_pass = all(
        rate >= 0.94
        for row in coverage.values()
        for rate in row.values()
    )
    small_iid = positive_power["Small-LI"]["stable_iid_improvement"]
    large_iid = positive_power["Large-LI"]["stable_iid_improvement"]
    cross_scale_power = large_iid >= small_iid - 0.15
    harm_power = {
        template: units["stable_harm"][template]["methods"]["dgr_f1"][
            "harm_rate"
        ]
        for template in TEMPLATES
    }
    harm_power_pass = all(rate >= 0.80 for rate in harm_power.values())

    gate = {
        "exact_f1_identity_zero_failures": identity_failures == 0,
        "false_improvement_control": false_improvement_control,
        "false_harm_control": false_harm_control,
        "positive_power_all_scenarios_both_templates": positive_power_pass,
        "stable_iid_median_stop_by_32": iid_stopping_pass,
        "unstable_scenarios_abstain": unstable_abstention_pass,
        "nontrivial_unstable_benchmark": nontrivial_benchmark,
        "simultaneous_mean_coverage": coverage_pass,
        "cross_scale_iid_power": cross_scale_power,
        "stable_harm_power": harm_power_pass,
    }
    gate["passed"] = all(gate.values())
    return {
        "experiment": "DGR_F1_Phase0_controlled_development",
        "mode": "development",
        "git_commit": manifests[0]["git_commit"],
        "manifest_count": len(manifests),
        "replicates_per_scenario": int(expected_replicates),
        "identity_failures": identity_failures,
        "positive_power": positive_power,
        "unstable_abstention": unstable_abstention,
        "unstable_naive_claim_rates": unstable_naive_claim_rates,
        "simultaneous_mean_coverage": coverage,
        "harm_power": harm_power,
        "units": units,
        "gate": gate,
        "decision": (
            "PROCEED_TO_PROSPECTIVE_FROZEN_CHECKPOINT_PLAN"
            if gate["passed"]
            else "STOP_DGR_F1"
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
    manifests, _ = load_manifests(args.input_root)
    aggregate = build_aggregate(
        manifests, args.expected_replicates)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": aggregate["decision"],
        "gate": aggregate["gate"],
        "output": str(args.output_json),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
