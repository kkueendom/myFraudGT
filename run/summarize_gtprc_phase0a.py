#!/usr/bin/env python3
"""Validate and summarize the seven GTPRC Phase 0A regimes."""

import argparse
import json
from pathlib import Path

from run.gtprc_phase0a_simulation import REGIMES


def load_manifests(root):
    paths = sorted(root.glob("*/gtprc_phase0a_manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    return manifests, paths


def validate_manifests(manifests):
    if len(manifests) != len(REGIMES):
        raise ValueError(
            "expected {} manifests, found {}".format(
                len(REGIMES), len(manifests)))
    regimes = {row["regime"] for row in manifests}
    if regimes != set(REGIMES):
        raise ValueError("regime set is incomplete or duplicated")
    commits = {row["git_commit"] for row in manifests}
    if len(commits) != 1:
        raise ValueError("formal tasks use different commits")
    for row in manifests:
        if row["experiment_protocol"] != (
                "controlled_graph_time_simulation"):
            raise ValueError("unexpected experiment protocol")
        if row["downstream_sampling_protocol"] != "dynamic_random":
            raise ValueError("downstream protocol mismatch")
        if row["alpha"] != 0.40 or row["delta"] != 0.05:
            raise ValueError("registered alpha/delta changed")


def build_aggregate(manifests):
    validate_manifests(manifests)
    regimes = {}
    coverage_passes = 0
    violation_passes = 0
    control_passes = 0
    nonuniform_passes = 0
    for row in sorted(manifests, key=lambda item: item["regime"]):
        gtprc = row["methods"]["gtprc_graph_time"]
        best_single = max(
            row["methods"]["entity_block"]["coverage_median"],
            row["methods"]["time_block"]["coverage_median"],
        )
        violation_pass = gtprc["violation_rate"] <= 0.07
        control_pass = all(
            item["false_qualification_rate"] <= 0.05
            for item in row["controls"].values()
        )
        coverage_pass = gtprc["oracle_fraction_median"] >= 0.60
        nonuniform_pass = (
            gtprc["coverage_median"] >= 0.75 * best_single
            or gtprc["violation_rate"] <
            min(
                row["methods"]["entity_block"]["violation_rate"],
                row["methods"]["time_block"]["violation_rate"],
            )
        )
        coverage_passes += coverage_pass
        violation_passes += violation_pass
        control_passes += control_pass
        nonuniform_passes += nonuniform_pass
        regimes[row["regime"]] = {
            "gtprc": gtprc,
            "row_iid": row["methods"]["row_iid"],
            "time_block": row["methods"]["time_block"],
            "entity_block": row["methods"]["entity_block"],
            "controls": row["controls"],
            "violation_pass": violation_pass,
            "control_pass": control_pass,
            "coverage_pass": coverage_pass,
            "nonuniform_conservatism_pass": nonuniform_pass,
        }
    gate = {
        "all_regimes_control_violation": violation_passes == len(REGIMES),
        "all_regimes_reject_controls": control_passes == len(REGIMES),
        "coverage_in_at_least_five_regimes": coverage_passes >= 5,
        "not_uniformly_dominated": nonuniform_passes >= 1,
    }
    gate["passed"] = all(gate.values())
    return {
        "experiment": "gtprc_phase0a_controlled_dependence",
        "git_commit": manifests[0]["git_commit"],
        "manifest_count": len(manifests),
        "replicate_count": sum(row["replicates"] for row in manifests),
        "alpha": 0.40,
        "delta": 0.05,
        "regimes": regimes,
        "gate": gate,
        "decision": (
            "PROCEED_TO_PHASE0B"
            if gate["passed"]
            else "STOP_AND_REDESIGN_GTPRC"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifests, paths = load_manifests(args.input_root)
    aggregate = build_aggregate(manifests)
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

