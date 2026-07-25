#!/usr/bin/env python3
"""Aggregate fixed-A2 repeated dynamic-sampling stability manifests."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


DATASET_ORDER = (
    "Small-LI",
    "Small-HI",
    "Medium-LI",
    "Medium-HI",
    "Large-LI",
    "Large-HI",
)


def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def bootstrap_mean_interval(values, seed, draws=10000):
    values = [float(value) for value in values]
    if not values:
        return [None, None]
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(draws)):
        means.append(statistics.fmean(
            rng.choice(values) for _ in values))
    return [quantile(means, 0.025), quantile(means, 0.975)]


def distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q025": quantile(values, 0.025),
        "median": quantile(values, 0.5),
        "q975": quantile(values, 0.975),
        "max": max(values),
    }


def load_manifests(root):
    paths = sorted(root.glob("*/a2_stability_manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    if len(manifests) != 12:
        raise ValueError(f"expected 12 manifests, found {len(manifests)}")
    return manifests, paths


def validate_manifests(manifests):
    grouped = {}
    for manifest in manifests:
        if manifest["sampling_protocol"] != "dynamic_random":
            raise ValueError("non-dynamic manifest detected")
        if manifest["fixed_target_panel"] is not False:
            raise ValueError("fixed target panel detected")
        if manifest["dedicated_evaluation_generator"] is not False:
            raise ValueError("dedicated evaluation generator detected")
        if manifest["sampler_rng_restoration"] is not False:
            raise ValueError("sampler RNG restoration detected")
        if manifest["eval_step_cap"] is not None:
            raise ValueError("formal manifest uses an evaluation step cap")
        if len(manifest["events"]) != 8:
            raise ValueError("formal manifest does not contain eight events")
        if manifest["validation_loader_iterations"] != 8 * 256:
            raise ValueError("validation iteration count differs")
        if manifest["test_loader_iterations"] != 8 * 256:
            raise ValueError("test iteration count differs")
        grouped.setdefault(manifest["dataset"], []).append(manifest)
    if set(grouped) != set(DATASET_ORDER):
        raise ValueError("dataset set differs")
    for dataset, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"{dataset}: expected two streams")
        if len({row["audit_seed"] for row in rows}) != 2:
            raise ValueError(f"{dataset}: audit seeds are not distinct")
    return grouped


def aggregate_dataset(dataset, manifests, seed):
    events = [
        event
        for manifest in manifests
        for event in manifest["events"]
    ]
    historical = float(
        manifests[0]["initial_a2_val_selected_test_f1"])
    test_f1 = [event["test"]["f1"] for event in events]
    centered = [
        value - statistics.fmean(test_f1) for value in test_f1]
    delta = [value - historical for value in test_f1]
    std = statistics.stdev(test_f1)
    event_band = max(
        abs(quantile(centered, 0.025)),
        abs(quantile(centered, 0.975)),
    )
    independent_comparison_band = 1.96 * math.sqrt(2.0) * std
    return {
        "dataset": dataset,
        "streams": len(manifests),
        "events": len(events),
        "git_commits": sorted({
            manifest["git_commit"] for manifest in manifests}),
        "model_seeds": sorted({
            manifest["model_seed"] for manifest in manifests}),
        "audit_seeds": sorted(
            manifest["audit_seed"] for manifest in manifests),
        "historical_initial_a2_f1": historical,
        "test_f1": distribution(test_f1),
        "test_f1_bootstrap_mean_ci95": bootstrap_mean_interval(
            test_f1, seed),
        "delta_vs_historical_initial_a2": distribution(delta),
        "val_threshold": distribution([
            event["val"]["threshold"] for event in events]),
        "test_precision": distribution([
            event["test"]["precision"] for event in events]),
        "test_recall": distribution([
            event["test"]["recall"] for event in events]),
        "test_prevalence": distribution([
            event["test"]["prevalence"] for event in events]),
        "test_positive_count": distribution([
            event["test"]["positives"] for event in events]),
        "test_unique_edge_rate": distribution([
            event["test_unique_edge_rate"] for event in events]),
        "event_level_sampling_band_95": event_band,
        "independent_comparison_noise_band_approx95": (
            independent_comparison_band),
        "empirical_probability_delta_gt_0_005": sum(
            value > 0.005 for value in delta) / len(delta),
        "empirical_probability_delta_lt_minus_0_005": sum(
            value < -0.005 for value in delta) / len(delta),
        "empirical_probability_abs_delta_le_0_005": sum(
            abs(value) <= 0.005 for value in delta) / len(delta),
    }


def format_number(value):
    return f"{float(value):.5f}"


def render_markdown(aggregate, source_root):
    lines = [
        "# Initial A2 Dynamic-Sampling Stability Audit",
        "",
        "## Material Passport",
        "",
        "- Sampling protocol: `dynamic_random`",
        "- Model: fixed initial-A2 checkpoints; no retraining",
        "- Streams: 2 per dataset",
        "- Events: 8 per stream, 16 per dataset, 96 total",
        "- Val/test iterations: 256 per event",
        f"- Source root: `{source_root}`",
        "- Formal baseline: the registered historical initial A2 table",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Historical A2 | Re-evaluated F1 mean +/- sd | "
        "Min-max | Mean delta | Event band 95% | Independent comparison "
        "band | P(delta > .005) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        row = aggregate["datasets"][dataset]
        test = row["test_f1"]
        delta = row["delta_vs_historical_initial_a2"]
        lines.append(
            f"| {dataset} | "
            f"{format_number(row['historical_initial_a2_f1'])} | "
            f"{format_number(test['mean'])} +/- "
            f"{format_number(test['std'])} | "
            f"{format_number(test['min'])}-"
            f"{format_number(test['max'])} | "
            f"{format_number(delta['mean'])} | "
            f"+/-{format_number(row['event_level_sampling_band_95'])} | "
            f"+/-{format_number(row['independent_comparison_noise_band_approx95'])} | "
            f"{row['empirical_probability_delta_gt_0_005']:.3f} |"
        )
    lines.extend([
        "",
        "The re-evaluated mean is a sampling audit, not a replacement "
        "baseline. The event band is the central 95% spread around the "
        "re-evaluated A2 mean. The independent comparison band is "
        "`1.96 * sqrt(2) * event_sd`; it is a descriptive approximation, not "
        "a formal confidence guarantee.",
        "",
        "## Threshold and Label Variation",
        "",
        "| Dataset | Val threshold mean +/- sd | Test positives mean +/- sd | "
        "Test prevalence mean +/- sd | Unique-edge rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_ORDER:
        row = aggregate["datasets"][dataset]
        threshold = row["val_threshold"]
        positives = row["test_positive_count"]
        prevalence = row["test_prevalence"]
        unique = row["test_unique_edge_rate"]
        lines.append(
            f"| {dataset} | "
            f"{format_number(threshold['mean'])} +/- "
            f"{format_number(threshold['std'])} | "
            f"{format_number(positives['mean'])} +/- "
            f"{format_number(positives['std'])} | "
            f"{format_number(prevalence['mean'])} +/- "
            f"{format_number(prevalence['std'])} | "
            f"{format_number(unique['mean'])} +/- "
            f"{format_number(unique['std'])} |"
        )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "1. Keep the registered historical initial A2 result in the formal "
        "comparison table.",
        "2. Use the repeated A2 distribution to diagnose whether an apparent "
        "gain is compatible with dynamic sampling variation.",
        "3. The existing `0.005` rule remains a minimum warning threshold, but "
        "a stability claim additionally requires repeated streams or seeds.",
        "4. Do not claim model improvement when the paired or repeated delta "
        "interval overlaps zero.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def main():
    args = parse_args()
    manifests, paths = load_manifests(args.input_root)
    grouped = validate_manifests(manifests)
    datasets = {
        dataset: aggregate_dataset(
            dataset,
            grouped[dataset],
            args.bootstrap_seed + index,
        )
        for index, dataset in enumerate(DATASET_ORDER)
    }
    aggregate = {
        "experiment": "initial_a2_dynamic_sampling_stability",
        "sampling_protocol": "dynamic_random",
        "formal_baseline": "registered_historical_initial_a2",
        "manifest_count": len(manifests),
        "event_count": sum(
            row["events"] for row in datasets.values()),
        "manifest_paths": [str(path) for path in paths],
        "datasets": datasets,
    }
    args.output_json.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(
        render_markdown(aggregate, args.input_root) + "\n")
    print(json.dumps({
        "manifest_count": aggregate["manifest_count"],
        "event_count": aggregate["event_count"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
