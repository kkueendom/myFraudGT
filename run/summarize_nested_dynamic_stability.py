#!/usr/bin/env python3
"""Summarize nested model-seed and dynamic-sampling stability manifests."""

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path


MODEL_SEEDS = (42, 43, 44)
AUDIT_SEEDS = (93001, 93002)
REPEATS = 4


def _load(root):
    paths = glob.glob(
        str(Path(root) / "**" / "nested_stability_manifest.json"),
        recursive=True,
    )
    manifests = [json.loads(Path(path).read_text()) for path in paths]
    if len(manifests) != 36:
        raise RuntimeError("expected 36 nested stability manifests")
    keys = {
        (row["dataset"], row["model_seed"], row["audit_seed"])
        for row in manifests
    }
    if len(keys) != 36:
        raise RuntimeError("manifest task keys are incomplete or duplicated")
    return manifests


def _variance_components(values):
    seeds = MODEL_SEEDS
    streams = AUDIT_SEEDS
    model_count = len(seeds)
    stream_count = len(streams)
    event_count = REPEATS
    flattened = [
        value
        for model_seed in seeds
        for audit_seed in streams
        for value in values[(model_seed, audit_seed)]
    ]
    grand_mean = statistics.fmean(flattened)
    model_means = {
        model_seed: statistics.fmean(
            value
            for audit_seed in streams
            for value in values[(model_seed, audit_seed)]
        )
        for model_seed in seeds
    }
    stream_means = {
        (model_seed, audit_seed): statistics.fmean(
            values[(model_seed, audit_seed)])
        for model_seed in seeds
        for audit_seed in streams
    }
    ss_model = stream_count * event_count * sum(
        (model_means[model_seed] - grand_mean) ** 2
        for model_seed in seeds
    )
    ss_stream = event_count * sum(
        (
            stream_means[(model_seed, audit_seed)]
            - model_means[model_seed]
        ) ** 2
        for model_seed in seeds
        for audit_seed in streams
    )
    ss_event = sum(
        (
            value - stream_means[(model_seed, audit_seed)]
        ) ** 2
        for model_seed in seeds
        for audit_seed in streams
        for value in values[(model_seed, audit_seed)]
    )
    ms_model = ss_model / (model_count - 1)
    ms_stream = ss_stream / (
        model_count * (stream_count - 1))
    ms_event = ss_event / (
        model_count * stream_count * (event_count - 1))
    variance_event = ms_event
    variance_stream = max(
        (ms_stream - ms_event) / event_count, 0.0)
    variance_model = max(
        (ms_model - ms_stream) / (stream_count * event_count),
        0.0,
    )
    total = variance_model + variance_stream + variance_event
    return {
        "grand_mean": grand_mean,
        "model_means": {
            str(key): value for key, value in model_means.items()},
        "stream_means": {
            "{}::{}".format(*key): value
            for key, value in stream_means.items()
        },
        "variance_model_seed": variance_model,
        "variance_stream": variance_stream,
        "variance_event": variance_event,
        "variance_total": total,
        "model_seed_share": (
            variance_model / total if total else 0.0),
        "sampling_share": (
            (variance_stream + variance_event) / total
            if total else 0.0
        ),
    }


def _ranking_diagnostics(components):
    stream_means = {
        tuple(int(item) for item in key.split("::")): value
        for key, value in components["stream_means"].items()
    }
    rankings = {}
    for audit_seed in AUDIT_SEEDS:
        rankings[str(audit_seed)] = sorted(
            MODEL_SEEDS,
            key=lambda model_seed: stream_means[
                (model_seed, audit_seed)],
            reverse=True,
        )
    pair_disagreements = []
    for left_index, left in enumerate(MODEL_SEEDS):
        for right in MODEL_SEEDS[left_index + 1:]:
            first = (
                stream_means[(left, AUDIT_SEEDS[0])]
                - stream_means[(right, AUDIT_SEEDS[0])]
            )
            second = (
                stream_means[(left, AUDIT_SEEDS[1])]
                - stream_means[(right, AUDIT_SEEDS[1])]
            )
            if first * second < 0.0:
                pair_disagreements.append([left, right])
    return {
        "rankings": rankings,
        "pairwise_disagreements": pair_disagreements,
        "ranking_reversal": bool(pair_disagreements),
    }


def summarize(root):
    manifests = _load(root)
    commits = sorted(set(row["git_commit"] for row in manifests))
    epochs = sorted(set(row["checkpoint_epoch"] for row in manifests))
    protocols = sorted(set(row["sampling_protocol"] for row in manifests))
    event_count = sum(len(row["events"]) for row in manifests)

    protocol_failures = []
    hash_groups = collections.defaultdict(list)
    by_dataset = collections.defaultdict(dict)
    thresholds = collections.defaultdict(list)
    baselines = {}
    for manifest in manifests:
        task_key = (
            int(manifest["model_seed"]),
            int(manifest["audit_seed"]),
        )
        if len(manifest["events"]) != REPEATS:
            protocol_failures.append(
                [manifest["dataset"], *task_key, "repeat_count"])
        if (
            manifest["sampling_protocol"] != "dynamic_random"
            or manifest["fixed_target_panel"]
            or manifest["dedicated_evaluation_generator"]
            or manifest["sampler_rng_restoration"]
        ):
            protocol_failures.append(
                [manifest["dataset"], *task_key, "protocol"])
        values = []
        for event in manifest["events"]:
            if (
                event["validation_loader_iterations"] != 256
                or event["test_loader_iterations"] != 256
            ):
                protocol_failures.append(
                    [manifest["dataset"], *task_key, "iterations"])
            if (
                event["val_unique_edge_rate"] != 1.0
                or event["test_unique_edge_rate"] != 1.0
            ):
                protocol_failures.append(
                    [manifest["dataset"], *task_key, "unique_edges"])
            hash_groups[(
                manifest["dataset"],
                int(manifest["audit_seed"]),
                int(event["repeat"]),
            )].append({
                "model_seed": int(manifest["model_seed"]),
                "val": event["val_edge_id_sha256"],
                "test": event["test_edge_id_sha256"],
            })
            values.append(float(event["test"]["f1"]))
            thresholds[manifest["dataset"]].append(
                float(event["val"]["threshold"]))
        by_dataset[manifest["dataset"]][task_key] = values
        baselines[manifest["dataset"]] = float(
            manifest["initial_a2_val_selected_test_f1"])

    hash_failures = []
    for key, rows in sorted(hash_groups.items()):
        if (
            len(rows) != 3
            or {row["model_seed"] for row in rows} != set(MODEL_SEEDS)
            or len({row["val"] for row in rows}) != 1
            or len({row["test"] for row in rows}) != 1
        ):
            hash_failures.append({
                "group": list(key),
                "rows": rows,
            })

    datasets = {}
    for dataset, values in sorted(by_dataset.items()):
        components = _variance_components(values)
        ranking = _ranking_diagnostics(components)
        flattened = [
            value for rows in values.values() for value in rows]
        baseline = baselines[dataset]
        deltas = [value - baseline for value in flattened]
        checkpoint_inflation = {}
        for model_seed in MODEL_SEEDS:
            seed_values = [
                value
                for audit_seed in AUDIT_SEEDS
                for value in values[(model_seed, audit_seed)]
            ]
            checkpoint_inflation[str(model_seed)] = (
                max(seed_values) - statistics.fmean(seed_values)
            )
        inflation_values = list(checkpoint_inflation.values())
        datasets[dataset] = {
            "event_count": len(flattened),
            "test_f1_mean": statistics.fmean(flattened),
            "test_f1_std": statistics.stdev(flattened),
            "test_f1_min": min(flattened),
            "test_f1_max": max(flattened),
            "val_threshold_std": statistics.stdev(
                thresholds[dataset]),
            "initial_a2_diagnostic": baseline,
            "delta_vs_initial_a2_min": min(deltas),
            "delta_vs_initial_a2_max": max(deltas),
            "delta_sign_reversal": (
                min(deltas) < 0.0 < max(deltas)),
            "checkpoint_raw_max_inflation": checkpoint_inflation,
            "median_raw_max_inflation": statistics.median(
                inflation_values),
            **components,
            **ranking,
        }

    sampling_ge_model = [
        dataset for dataset, row in datasets.items()
        if (
            row["variance_stream"] + row["variance_event"]
            >= row["variance_model_seed"]
        )
    ]
    rank_reversal = [
        dataset for dataset, row in datasets.items()
        if row["ranking_reversal"]
    ]
    sign_reversal = [
        dataset for dataset, row in datasets.items()
        if row["delta_sign_reversal"]
    ]
    raw_inflation = [
        dataset for dataset, row in datasets.items()
        if row["median_raw_max_inflation"] >= 0.01
    ]
    empirical_gate = {
        "sampling_ge_model_variance_at_least_2":
            len(sampling_ge_model) >= 2,
        "seed_rank_reversal_at_least_1":
            len(rank_reversal) >= 1,
        "a2_delta_sign_reversal_at_least_2":
            len(sign_reversal) >= 2,
        "raw_max_inflation_at_least_0.01_on_4":
            len(raw_inflation) >= 4,
    }
    integrity_gate = {
        "manifest_count_36": len(manifests) == 36,
        "event_count_144": event_count == 144,
        "one_commit": len(commits) == 1,
        "epoch_499": epochs == [499],
        "dynamic_random": protocols == ["dynamic_random"],
        "protocol_failures_zero": not protocol_failures,
        "hash_groups_48": len(hash_groups) == 48,
        "hash_failures_zero": not hash_failures,
    }
    ready = all(integrity_gate.values()) and any(empirical_gate.values())
    return {
        "decision": (
            "PROCEED_NEGATIVE_BENCHMARK_PAPER"
            if ready else "STOP_OR_RESCOPE_BENCHMARK"
        ),
        "benchmark_ready": ready,
        "root": str(Path(root).resolve()),
        "commits": commits,
        "epochs": epochs,
        "protocols": protocols,
        "manifest_count": len(manifests),
        "event_count": event_count,
        "integrity_gate": integrity_gate,
        "empirical_gate": empirical_gate,
        "sampling_ge_model_variance_datasets": sampling_ge_model,
        "seed_rank_reversal_datasets": rank_reversal,
        "a2_delta_sign_reversal_datasets": sign_reversal,
        "raw_max_inflation_datasets": raw_inflation,
        "protocol_failures": protocol_failures,
        "hash_failures": hash_failures,
        "datasets": datasets,
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
        "integrity_gate": result["integrity_gate"],
        "empirical_gate": result["empirical_gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
