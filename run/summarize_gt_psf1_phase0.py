#!/usr/bin/env python3
"""Summarize the preregistered GT-psF1 Phase 0 gate."""

import argparse
import glob
import json
import math
from pathlib import Path


def _load_manifests(root):
    paths = sorted(glob.glob(str(Path(root) / "task_*" / "phase0_manifest.json")))
    manifests = [json.loads(Path(path).read_text()) for path in paths]
    if len(manifests) != 7:
        raise RuntimeError("expected seven task manifests")
    task_ids = sorted(int(item["task"]["task_id"]) for item in manifests)
    if task_ids != list(range(7)):
        raise RuntimeError("task IDs are incomplete or duplicated")
    return manifests


def summarize(root):
    manifests = _load_manifests(root)
    commits = sorted(set(item["commit"] for item in manifests))
    if len(commits) != 1:
        raise RuntimeError("manifests do not share one commit")

    units = []
    allocations = {}
    for manifest in manifests:
        scenario = manifest["task"]["scenario"]
        for template, result in manifest["templates"].items():
            effects = result["effects"]
            null = effects["null"]["methods"]
            improvement = effects["improvement"]["methods"]["gt_psf1"]
            harm = effects["harm"]["methods"]["gt_psf1"]
            unit = {
                "task_id": int(manifest["task"]["task_id"]),
                "scenario": scenario,
                "template": template,
                "scope_status": result["scope_status"],
                "gt_null_coverage": null["gt_psf1"]["coverage"],
                "gt_false_improvement": null["gt_psf1"][
                    "false_or_true_improvement"],
                "gt_false_harm": null["gt_psf1"]["false_or_true_harm"],
                "gt_out_of_scope_rate": null["gt_psf1"][
                    "out_of_scope_rate"],
                "improvement_power": improvement[
                    "false_or_true_improvement"],
                "harm_power": harm["false_or_true_harm"],
                "row_iid_false_improvement": null["row_iid"][
                    "false_or_true_improvement"],
                "row_iid_false_harm": null["row_iid"][
                    "false_or_true_harm"],
                "target_iid_false_improvement": null["target_iid"][
                    "false_or_true_improvement"],
                "target_iid_false_harm": null["target_iid"][
                    "false_or_true_harm"],
                "dyadic_null_coverage": null["endpoint_dyadic"]["coverage"],
                "graph_hac_null_coverage": null["graph_hac"]["coverage"],
            }
            units.append(unit)
            if "allocation" in result:
                allocation = result["allocation"]
                regret = allocation["cost_regret"]
                if not math.isfinite(regret):
                    regret = None
                allocations[(scenario, template)] = {
                    "selected_replicates": allocation["plan"][
                        "selected_replicates"],
                    "cost_regret": regret,
                    "empirical_cheapest": allocation[
                        "empirical_cheapest"],
                    "selected_feasible": allocation[
                        "selected_feasible"],
                }

    in_scope = [
        unit for unit in units if unit["scope_status"] == "OK"]
    dense = [
        unit for unit in units
        if unit["scenario"] == "dense_hub_out_of_scope"]
    coverage_valid = all(
        0.94 <= unit["gt_null_coverage"] <= 0.97
        for unit in in_scope
    )
    directional_error_valid = all(
        unit["gt_false_improvement"] <= 0.05
        and unit["gt_false_harm"] <= 0.05
        for unit in in_scope
    )
    coverage_regret_valid = all(
        unit["gt_null_coverage"]
        >= max(
            value for value in (
                unit["dyadic_null_coverage"],
                unit["graph_hac_null_coverage"],
            )
            if value is not None
        ) - 0.01
        for unit in in_scope
    )
    dense_abstention_valid = all(
        unit["gt_out_of_scope_rate"] >= 0.90
        and unit["gt_false_improvement"] == 0.0
        and unit["gt_false_harm"] == 0.0
        for unit in dense
    )
    positive_power_valid = all(
        unit["improvement_power"] >= 0.80 for unit in in_scope)
    harm_power_valid = all(
        unit["harm_power"] >= 0.80 for unit in in_scope)

    gap_units = [
        unit for unit in in_scope
        if unit["scenario"] in ("two_hop_spillover", "temporal_ar1")
    ]
    cross_template_power_valid = True
    for scenario in ("two_hop_spillover", "temporal_ar1"):
        scenario_units = {
            unit["template"]: unit
            for unit in gap_units if unit["scenario"] == scenario
        }
        for key in ("improvement_power", "harm_power"):
            gap = abs(
                scenario_units["moderate"][key]
                - scenario_units["rare_sparse"][key]
            )
            cross_template_power_valid &= gap <= 0.15

    naive_type1_nontrivial = any(
        max(
            unit["row_iid_false_improvement"],
            unit["row_iid_false_harm"],
            unit["target_iid_false_improvement"],
            unit["target_iid_false_harm"],
        ) > 0.08
        for unit in in_scope
        if unit["scenario"] != "iid_deterministic"
    )
    spillover_units = [
        unit for unit in in_scope
        if unit["scenario"] == "two_hop_spillover"]
    dyadic_loss_nontrivial = any(
        (
            unit["gt_null_coverage"]
            - unit["dyadic_null_coverage"]
        ) >= 0.03
        for unit in spillover_units
    )
    allocation_valid = bool(allocations) and all(
        item["cost_regret"] is not None
        and item["cost_regret"] <= 0.10
        for item in allocations.values()
    )
    low_repeats = {
        item["selected_replicates"]
        for key, item in allocations.items()
        if key[0] == "low_sampler_variance"
    }
    high_repeats = {
        item["selected_replicates"]
        for key, item in allocations.items()
        if key[0] == "high_sampler_variance"
    }
    different_allocations = (
        len(low_repeats) == 1
        and len(high_repeats) == 1
        and low_repeats != high_repeats
    )
    novelty_value = allocation_valid and dyadic_loss_nontrivial

    gate = {
        "null_coverage_0.94_to_0.97": coverage_valid,
        "directional_type1_at_most_0.05": directional_error_valid,
        "coverage_regret_at_most_0.01": coverage_regret_valid,
        "dense_hub_abstention_at_least_0.90": dense_abstention_valid,
        "positive_power_at_least_0.80": positive_power_valid,
        "harm_power_at_least_0.80": harm_power_valid,
        "cross_template_power_gap_at_most_0.15":
            cross_template_power_valid,
        "naive_type1_benchmark_nontrivial": naive_type1_nontrivial,
        "dyadic_spillover_coverage_loss_at_least_0.03":
            dyadic_loss_nontrivial,
        "allocation_cost_regret_at_most_0.10": allocation_valid,
        "low_high_sampler_allocations_differ": different_allocations,
        "novelty_value_beyond_graph_hac": novelty_value,
    }
    overall = all(gate.values())
    return {
        "decision": "PASS_GT_PSF1" if overall else "STOP_GT_PSF1",
        "overall_pass": overall,
        "commit": commits[0],
        "root": str(Path(root).resolve()),
        "task_count": len(manifests),
        "experiments_per_task_unit": manifests[0]["experiments"],
        "reference_experiments_per_task_unit":
            manifests[0]["reference_experiments"],
        "gate": gate,
        "in_scope_unit_count": len(in_scope),
        "valid_coverage_unit_count": sum(
            0.94 <= unit["gt_null_coverage"] <= 0.97
            for unit in in_scope
        ),
        "positive_power_pass_count": sum(
            unit["improvement_power"] >= 0.80 for unit in in_scope),
        "harm_power_pass_count": sum(
            unit["harm_power"] >= 0.80 for unit in in_scope),
        "units": units,
        "allocations": {
            "{}::{}".format(*key): value
            for key, value in sorted(allocations.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = summarize(args.root)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "gate": result["gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
