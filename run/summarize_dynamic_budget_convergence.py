#!/usr/bin/env python3
"""Validate and summarize dynamic evaluation budget convergence manifests."""

import argparse
import collections
import glob
import json
import math
import statistics
from pathlib import Path


MODEL_SEEDS = (42, 43, 44)


def percentile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(values) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _load(root, manifest_filename):
    paths = glob.glob(
        str(Path(root) / "**" / manifest_filename),
        recursive=True,
    )
    return [json.loads(Path(path).read_text()) for path in paths]


def _adequate(row, rule):
    return (
        row["absolute_f1_error"]["median"]
        <= float(rule["median_absolute_f1_error_max"])
        and row["absolute_f1_error"]["p90"]
        <= float(rule["p90_absolute_f1_error_max"])
        and row["delta_sign_agreement"]
        >= float(rule["delta_sign_agreement_min"])
        and row["all_unique_edge_rates_one"]
    )


def summarize(root, spec):
    manifests = _load(root, spec["manifest_filename"])
    expected_budgets = tuple(int(value) for value in spec["budgets"])
    expected_audit_seeds = tuple(
        int(value) for value in spec["audit_seeds"])
    expected_repeats = int(spec["repeats"])
    protocol_failures = []
    hash_groups = collections.defaultdict(list)
    values = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)))
    commits = set()
    epochs = set()
    task_keys = set()
    event_count = 0

    for manifest in manifests:
        task_key = (
            manifest["dataset"],
            int(manifest["model_seed"]),
            int(manifest["audit_seed"]),
        )
        task_keys.add(task_key)
        commits.add(manifest["git_commit"])
        epochs.add(int(manifest["checkpoint_epoch"]))
        if (
            manifest["sampling_protocol"] != "dynamic_random"
            or manifest["fixed_target_panel"]
            or manifest["dedicated_evaluation_generator"]
            or manifest["sampler_rng_restoration"]
            or tuple(manifest["budgets"]) != expected_budgets
            or int(manifest["reference_budget"])
            != int(spec["reference_budget"])
            or int(manifest["repeats"]) != expected_repeats
        ):
            protocol_failures.append([*task_key, "manifest"])
        if len(manifest["events"]) != expected_repeats:
            protocol_failures.append([*task_key, "event_count"])
        for event in manifest["events"]:
            event_count += 1
            if (
                not event.get("nested_prefix_verified")
                or tuple(
                    sorted(int(key) for key in event["budgets"])
                ) != expected_budgets
            ):
                protocol_failures.append(
                    [*task_key, event["repeat"], "prefix"])
            for budget in expected_budgets:
                row = event["budgets"][str(budget)]
                if (
                    row["validation_loader_iterations"] != budget
                    or row["test_loader_iterations"] != budget
                ):
                    protocol_failures.append(
                        [*task_key, event["repeat"], budget, "steps"])
                hash_groups[(
                    manifest["dataset"],
                    int(manifest["audit_seed"]),
                    int(event["repeat"]),
                    budget,
                )].append({
                    "model_seed": int(manifest["model_seed"]),
                    "val": row["val_edge_id_sha256"],
                    "test": row["test_edge_id_sha256"],
                })
                bucket = values[manifest["dataset"]][budget]
                bucket["absolute_f1_error"].append(
                    row["paired_absolute_f1_error_vs_reference"])
                bucket["signed_f1_error"].append(
                    row["paired_f1_error_vs_reference"])
                bucket["squared_f1_error"].append(
                    row["paired_f1_error_vs_reference"] ** 2)
                bucket["absolute_threshold_error"].append(abs(
                    row["paired_threshold_error_vs_reference"]))
                bucket["delta_sign_agreement"].append(float(
                    row["delta_sign_agrees_with_reference"]))
                bucket["within_0_005"].append(float(
                    row["paired_absolute_f1_error_vs_reference"]
                    <= 0.005))
                bucket["within_0_01"].append(float(
                    row["paired_absolute_f1_error_vs_reference"]
                    <= 0.01))
                bucket["val_unique_edge_rate"].append(
                    row["val_unique_edge_rate"])
                bucket["test_unique_edge_rate"].append(
                    row["test_unique_edge_rate"])
                bucket["test_positive_count"].append(
                    row["test"]["positives"])
                bucket["test_prevalence"].append(
                    row["test"]["prevalence"])

    hash_failures = []
    for key, rows in sorted(hash_groups.items()):
        if (
            len(rows) != len(MODEL_SEEDS)
            or {row["model_seed"] for row in rows}
            != set(MODEL_SEEDS)
            or len({row["val"] for row in rows}) != 1
            or len({row["test"] for row in rows}) != 1
        ):
            hash_failures.append({"group": list(key), "rows": rows})

    datasets = {}
    adequate_by_budget = collections.defaultdict(list)
    for dataset, budget_rows in sorted(values.items()):
        summarized = {}
        for budget in expected_budgets:
            bucket = budget_rows[budget]
            absolute = distribution(bucket["absolute_f1_error"])
            row = {
                "absolute_f1_error": absolute,
                "signed_f1_error": distribution(
                    bucket["signed_f1_error"]),
                "rmse_f1_error": math.sqrt(statistics.fmean(
                    bucket["squared_f1_error"])),
                "absolute_threshold_error": distribution(
                    bucket["absolute_threshold_error"]),
                "delta_sign_agreement": statistics.fmean(
                    bucket["delta_sign_agreement"]),
                "within_0_005": statistics.fmean(
                    bucket["within_0_005"]),
                "within_0_01": statistics.fmean(
                    bucket["within_0_01"]),
                "test_positive_count": distribution(
                    bucket["test_positive_count"]),
                "test_prevalence": distribution(
                    bucket["test_prevalence"]),
                "all_unique_edge_rates_one": all(
                    value == 1.0
                    for value in (
                        bucket["val_unique_edge_rate"]
                        + bucket["test_unique_edge_rate"]
                    )
                ),
            }
            row["point_adequate"] = _adequate(
                row, spec["adequacy_rule"])
            summarized[str(budget)] = row
        for index, budget in enumerate(expected_budgets):
            monotone = all(
                summarized[str(larger)]["point_adequate"]
                for larger in expected_budgets[index:]
            )
            summarized[str(budget)]["monotone_adequate"] = monotone
            if monotone:
                adequate_by_budget[budget].append(dataset)
        minimum = next(
            (
                budget for budget in expected_budgets
                if summarized[str(budget)]["monotone_adequate"]
            ),
            None,
        )
        datasets[dataset] = {
            "minimum_adequate_budget": minimum,
            "budgets": summarized,
        }

    minimum_dataset_count = int(
        spec["adequacy_rule"]["minimum_adequate_datasets"])
    cross_dataset_recommendation = next(
        (
            budget for budget in expected_budgets
            if len(adequate_by_budget[budget])
            >= minimum_dataset_count
        ),
        None,
    )
    expected_hash_groups = (
        len(spec["datasets"])
        * len(expected_audit_seeds)
        * expected_repeats
        * len(expected_budgets)
    )
    expected_tasks = {
        (
            dataset["dataset"],
            int(model_seed),
            int(audit_seed),
        )
        for dataset in spec["datasets"]
        for model_seed in dataset["model_seeds"]
        for audit_seed in expected_audit_seeds
    }
    integrity_gate = {
        "manifest_count": (
            len(manifests) == int(spec["task_count"])),
        "task_keys_complete": task_keys == expected_tasks,
        "event_count": event_count == int(spec["event_count"]),
        "one_experiment_commit": len(commits) == 1,
        "checkpoint_epoch": epochs == {int(spec["checkpoint_epoch"])},
        "protocol_failures_zero": not protocol_failures,
        "hash_group_count": len(hash_groups) == expected_hash_groups,
        "hash_failures_zero": not hash_failures,
    }
    return {
        "decision": (
            "VALID_BUDGET_RECOMMENDATION"
            if all(integrity_gate.values())
            else "INVALID_FORMAL_RUN"
        ),
        "root": str(Path(root).resolve()),
        "manifest_count": len(manifests),
        "event_count": event_count,
        "commits": sorted(commits),
        "checkpoint_epochs": sorted(epochs),
        "integrity_gate": integrity_gate,
        "protocol_failures": protocol_failures,
        "hash_failures": hash_failures,
        "cross_dataset_recommended_budget": (
            cross_dataset_recommendation),
        "adequate_datasets_by_budget": {
            str(budget): sorted(adequate_by_budget[budget])
            for budget in expected_budgets
        },
        "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--spec",
        default=str(
            Path(__file__).with_name(
                "dynamic_budget_convergence_spec.json")),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    result = summarize(args.root, spec)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "integrity_gate": result["integrity_gate"],
        "cross_dataset_recommended_budget": (
            result["cross_dataset_recommended_budget"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

