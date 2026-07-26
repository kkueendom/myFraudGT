#!/usr/bin/env python3
"""Aggregate and gate the preregistered TREFIC development screen."""

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

from fraudGT.evidence.gtf1c import f1_from_counts
from fraudGT.evidence.trefic import exact_f1_sign_utility
from run.gtf1c_phase0_real_graph import (
    NEGATIVE_REGIMES,
    POSITIVE_REGIMES,
)
from run.gtf1c_phase0_v2 import V2_REGIMES
from run.trefic_phase0 import METHODS


def load_manifests(root):
    paths = sorted(root.glob("*/trefic_phase0_manifest.json"))
    return [json.loads(path.read_text()) for path in paths], paths


def validate(manifests, expected_replicates):
    if len(manifests) != len(V2_REGIMES):
        raise ValueError("expected seven TREFIC manifests")
    if {row["regime"] for row in manifests} != set(V2_REGIMES):
        raise ValueError("TREFIC regime set is incomplete")
    if len({row["git_commit"] for row in manifests}) != 1:
        raise ValueError("TREFIC tasks use different commits")
    if {row["seed"] for row in manifests} != set(range(75001, 75008)):
        raise ValueError("TREFIC seeds do not match registration")
    for row in manifests:
        if row.get("method_version") != 1:
            raise ValueError("unexpected TREFIC method version")
        if row["replicates"] != int(expected_replicates):
            raise ValueError("unexpected TREFIC replicate count")
        if row["sampling_protocol"] != "dynamic_random":
            raise ValueError("non-dynamic TREFIC manifest")
        if row["validation_loader_iterations"] != 0:
            raise ValueError("TREFIC opened validation loader")
        if row["test_loader_iterations"] != 0:
            raise ValueError("TREFIC opened test loader")
        if row["locked_candidate_count"] != 3:
            raise ValueError("candidate family changed")
        if row["reference_candidate_delta"] != row["delta"] / 3.0:
            raise ValueError("reference multiplicity correction changed")
        if row["trefic_endpoint_delta"] != row["delta"] / 6.0:
            raise ValueError("TREFIC endpoint correction changed")
        if row["rho_lower"] != 0.0:
            raise ValueError("TREFIC lower ratio endpoint changed")
        if set(row["datasets"]) != {"Small-LI", "Large-LI"}:
            raise ValueError("TREFIC dataset set is incomplete")
        for dataset in row["datasets"].values():
            if dataset["trials"] != 3 * int(expected_replicates):
                raise ValueError("unexpected TREFIC rotation count")


def _trial_path(manifest, manifest_path):
    declared = Path(manifest["trial_records"])
    if declared.exists():
        return declared
    fallback = manifest_path.parent / "trefic_trials.jsonl"
    if not fallback.exists():
        raise FileNotFoundError("TREFIC trial records missing")
    return fallback


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _median(values):
    return statistics.median(values) if values else 0.0


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
    by_replicate = defaultdict(list)
    for row in rows:
        by_replicate[row["replicate"]].append(row)
    replicate_qualified = 0
    replicate_violations = 0
    replicate_practical_failures = 0
    for replicate_rows in by_replicate.values():
        selected = [
            row for row in replicate_rows
            if row["chosen_candidate"][method] >= 0
        ]
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
        "replicate_count": replicate_count,
        "replicate_any_qualification_rate": (
            replicate_qualified / max(replicate_count, 1)),
        "replicate_any_violation_rate": (
            replicate_violations / max(replicate_count, 1)),
        "replicate_any_practical_failure_rate": (
            replicate_practical_failures / max(replicate_count, 1)),
    }


def audit_exact_identity(max_count=3):
    checked = 0
    for tp, fp, fn, tn in itertools.product(
            range(int(max_count) + 1), repeat=4):
        denominator = tp + fp + fn
        if denominator == 0:
            continue
        base_f1 = f1_from_counts(tp, fp, fn)
        rho = tp / denominator
        for ac in range(fn + 1):
            for ab in range(tn + 1):
                for rc in range(fp + 1):
                    for rb in range(tp + 1):
                        routed_f1 = f1_from_counts(
                            tp + ac - rb,
                            fp + ab - rc,
                            fn - ac + rb,
                        )
                        utility = exact_f1_sign_utility(
                            ac, ab, rc, rb, rho)
                        actual = routed_f1 - base_f1
                        if (
                            (actual > 1e-12) != (utility > 1e-12)
                            or (actual < -1e-12) != (utility < -1e-12)
                        ):
                            return {
                                "passed": False,
                                "checked": checked,
                                "counterexample": {
                                    "tp": tp,
                                    "fp": fp,
                                    "fn": fn,
                                    "tn": tn,
                                    "add_corrected": ac,
                                    "add_broken": ab,
                                    "remove_corrected": rc,
                                    "remove_broken": rb,
                                    "actual_delta": actual,
                                    "utility": utility,
                                },
                            }
                        checked += 1
    return {"passed": True, "checked": checked}


def build_aggregate(manifests, paths, expected_replicates):
    validate(manifests, expected_replicates)
    units = {}
    for manifest, path in sorted(
            zip(manifests, paths), key=lambda item: item[0]["regime"]):
        regime = manifest["regime"]
        records = [
            json.loads(line)
            for line in _trial_path(
                manifest, path).read_text().splitlines()
            if line
        ]
        expected_records = 2 * 3 * int(expected_replicates)
        if len(records) != expected_records:
            raise ValueError("unexpected TREFIC trial-record count")
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

    datasets = ("Small-LI", "Large-LI")
    negative_units = [
        units[regime][dataset]
        for regime in NEGATIVE_REGIMES
        for dataset in datasets
    ]
    normal_units = [
        units[regime][dataset]
        for regime in V2_REGIMES
        for dataset in datasets
    ]
    positive_units = {
        dataset: [
            units[regime][dataset]
            for regime in POSITIVE_REGIMES
        ]
        for dataset in datasets
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
            row["methods"]["trefic"]["qualification_rate"] >= 0.25
            for row in rows
        )
        for dataset, rows in positive_units.items()
    }
    conditional_coverage = {
        dataset: sum(
            row["methods"]["trefic"]["conditional"][
                "conditional_oracle_fraction_median"
            ] >= 0.30
            for row in rows
        )
        for dataset, rows in positive_units.items()
    }
    iid_retention = {}
    for dataset in datasets:
        row = units["iid_positive"][dataset]["methods"]
        reference = row["iid_paired_f1"]["qualification_rate"]
        iid_retention[dataset] = (
            row["trefic"]["qualification_rate"] / reference
            if reference > 0.0
            else 0.0
        )
    identity_audit = audit_exact_identity()

    gate = {
        "exact_f1_sign_identity": identity_audit["passed"],
        "row_harm_exposes_two_negative_units": (
            exposed_negative_units >= 2),
        "trefic_controls_all_normal_units": all(
            row["methods"]["trefic"]["false_qualification_rate"]
            <= 0.07
            for row in normal_units
        ),
        "controls_rejected_all_units": all(
            condition["qualification_rate"] <= 0.05
            for row in normal_units
            for condition in row["controls"].values()
        ),
        "practical_failure_control_all_positive_units": all(
            row["methods"]["trefic"]["practical_failure_rate"]
            <= 0.10
            for rows in positive_units.values()
            for row in rows
        ),
        "small_positive_power_three_regimes": (
            positive_power["Small-LI"] >= 3),
        "large_positive_power_two_regimes": (
            positive_power["Large-LI"] >= 2),
        "conditional_delta_all_qualified_positive_units": all(
            row["methods"]["trefic"]["conditional"][
                "conditional_delta_median"
            ] >= 0.005
            for rows in positive_units.values()
            for row in rows
            if row["methods"]["trefic"]["qualified"] > 0
        ),
        "small_conditional_coverage_three_regimes": (
            conditional_coverage["Small-LI"] >= 3),
        "large_conditional_coverage_two_regimes": (
            conditional_coverage["Large-LI"] >= 2),
        "iid_power_retention_both_datasets": all(
            value >= 0.70 for value in iid_retention.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "experiment": "TREFIC_Phase0_ratio_envelope",
        "mode": "development",
        "git_commit": manifests[0]["git_commit"],
        "manifest_count": len(manifests),
        "replicates_per_regime": int(expected_replicates),
        "trial_count": sum(
            row["datasets"][dataset]["trials"]
            for row in manifests
            for dataset in datasets
        ),
        "identity_audit": identity_audit,
        "exposed_negative_units": exposed_negative_units,
        "positive_power_regimes": positive_power,
        "conditional_coverage_regimes": conditional_coverage,
        "iid_power_retention": iid_retention,
        "units": units,
        "gate": gate,
        "decision": (
            "PROCEED_TO_FRESH_FORMAL_PREREGISTRATION"
            if gate["passed"]
            else "STOP_TREFIC"
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
