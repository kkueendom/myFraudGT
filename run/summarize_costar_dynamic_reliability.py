#!/usr/bin/env python3
"""Validate and aggregate fixed-COSTAR dynamic reliability manifests."""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.summarize_a2_dynamic_stability import distribution
from run.summarize_tier_dynamic_reliability import classify_mechanism


DATASET_ORDER = ("Small-LI", "Large-LI")


def load_manifests(root):
    paths = sorted(root.glob("*/costar_reliability_manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    if len(manifests) != 4:
        raise ValueError(f"expected 4 manifests, found {len(manifests)}")
    return manifests, paths


def validate_manifests(manifests, paths):
    grouped = {}
    for manifest, path in zip(manifests, paths):
        if manifest["experiment"] != (
            "costar_fixed_checkpoint_dynamic_reliability"
        ):
            raise ValueError("unexpected experiment type")
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
        if manifest["repeats"] != 8 or len(manifest["events"]) != 8:
            raise ValueError("formal manifest does not contain eight events")
        if manifest["validation_loader_iterations"] != 8 * 256:
            raise ValueError("validation iteration count differs")
        if manifest["test_loader_iterations"] != 8 * 256:
            raise ValueError("test iteration count differs")
        if manifest["git_commit"] != "8d95c17a":
            raise ValueError("unexpected audit commit")
        if manifest["model_git_commit"] != "4abe58c6":
            raise ValueError("unexpected COSTAR model commit")
        if manifest["checkpoint_epoch"] != 499:
            raise ValueError("COSTAR audit checkpoint is not epoch 499")
        if any(row["shuffle"] is not True
               for row in manifest["loader_audit"]):
            raise ValueError("one loader does not use shuffle=True")
        if any(
            row["loader_generator"] is not None
            or row["sampler_generator"] is not None
            for row in manifest["loader_audit"]
        ):
            raise ValueError("one loader uses a dedicated generator")

        trajectory = path.parent / "reliability_trajectory.jsonl"
        trajectory_rows = [
            json.loads(line)
            for line in trajectory.read_text().splitlines()
            if line.strip()
        ]
        if trajectory_rows != manifest["events"]:
            raise ValueError("trajectory and manifest events differ")

        for event in manifest["events"]:
            if event["sampling_protocol"] != "dynamic_random":
                raise ValueError("non-dynamic event detected")
            if event["validation_loader_iterations"] != 256:
                raise ValueError("event validation iterations differ")
            if event["test_loader_iterations"] != 256:
                raise ValueError("event test iterations differ")
            samples = {
                event["test"][condition]["samples"]
                for condition in ("normal", "shuffled", "off")
            }
            samples.add(event["a2_anchor_test"]["samples"])
            samples.add(
                event["interventions"]["normal"]["samples"]
            )
            if len(samples) != 1:
                raise ValueError("same-batch condition samples differ")

        grouped.setdefault(manifest["dataset"], []).append(manifest)

    if set(grouped) != set(DATASET_ORDER):
        raise ValueError("dataset set differs")
    for dataset, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"{dataset}: expected two streams")
        if len({row["audit_seed"] for row in rows}) != 2:
            raise ValueError(f"{dataset}: audit seeds are not distinct")
        if len({row["checkpoint"] for row in rows}) != 1:
            raise ValueError(f"{dataset}: streams use different checkpoints")
        if len({row["model_seed"] for row in rows}) != 1:
            raise ValueError(f"{dataset}: streams use different model seeds")
    return grouped


def aggregate_dataset(dataset, manifests):
    events = [
        event
        for manifest in manifests
        for event in manifest["events"]
    ]
    normal_f1 = [
        event["test"]["normal"]["f1"] for event in events
    ]
    shuffled_f1 = [
        event["test"]["shuffled"]["f1"] for event in events
    ]
    off_f1 = [
        event["test"]["off"]["f1"] for event in events
    ]
    anchor_f1 = [
        event["a2_anchor_test"]["f1"] for event in events
    ]
    anchor_delta = [
        event["same_batch_delta_vs_a2_anchor"]
        for event in events
    ]
    historical_delta = [
        event["delta_vs_historical_initial_a2"]
        for event in events
    ]
    historical_values = [
        normal - delta
        for normal, delta in zip(normal_f1, historical_delta)
    ]
    if max(historical_values) - min(historical_values) > 1e-12:
        raise ValueError("historical A2 reference changes across events")
    shuffle_gap = [
        event["normal_minus_shuffled_f1"] for event in events
    ]
    off_gap = [
        event["normal_minus_off_f1"] for event in events
    ]
    changed = [
        event["interventions"]["normal"]["changed_predictions"]
        for event in events
    ]
    corrected = [
        event["interventions"]["normal"]["corrected_predictions"]
        for event in events
    ]
    broken = [
        event["interventions"]["normal"]["broken_predictions"]
        for event in events
    ]
    corrected_minus_broken = [
        event["interventions"]["normal"]["corrected_minus_broken"]
        for event in events
    ]
    corrected_gt_broken = sum(
        left > right for left, right in zip(corrected, broken)
    )
    corrected_le_broken = len(events) - corrected_gt_broken
    classification = classify_mechanism(
        statistics.fmean(shuffle_gap),
        statistics.fmean(off_gap),
        statistics.fmean(anchor_delta),
        corrected_gt_broken,
        corrected_le_broken,
        statistics.fmean(changed),
        len(events),
    )
    threshold_crossing = (
        min(shuffle_gap) < 0.01 <= max(shuffle_gap)
    )
    utility_sign_crossing = min(anchor_delta) < 0 < max(anchor_delta)
    stream_rows = []
    for manifest in sorted(
        manifests, key=lambda row: row["audit_seed"]
    ):
        stream_events = manifest["events"]
        stream_rows.append({
            "experiment_label": manifest["experiment_label"],
            "audit_seed": manifest["audit_seed"],
            "normal_test_f1_mean": statistics.fmean(
                event["test"]["normal"]["f1"]
                for event in stream_events
            ),
            "same_batch_delta_vs_a2_anchor_mean": statistics.fmean(
                event["same_batch_delta_vs_a2_anchor"]
                for event in stream_events
            ),
            "normal_minus_shuffled_f1_mean": statistics.fmean(
                event["normal_minus_shuffled_f1"]
                for event in stream_events
            ),
        })

    return {
        "dataset": dataset,
        "streams": len(manifests),
        "events": len(events),
        "model_seed": manifests[0]["model_seed"],
        "audit_seeds": sorted(
            manifest["audit_seed"] for manifest in manifests
        ),
        "git_commit": manifests[0]["git_commit"],
        "model_git_commit": manifests[0]["model_git_commit"],
        "checkpoint": manifests[0]["checkpoint"],
        "checkpoint_epoch": manifests[0]["checkpoint_epoch"],
        "normal_test_f1": distribution(normal_f1),
        "shuffled_test_f1": distribution(shuffled_f1),
        "off_test_f1": distribution(off_f1),
        "same_batch_a2_anchor_test_f1": distribution(anchor_f1),
        "same_batch_delta_vs_a2_anchor": distribution(anchor_delta),
        "historical_initial_a2_f1": historical_values[0],
        "delta_vs_historical_initial_a2": distribution(
            historical_delta
        ),
        "normal_minus_shuffled_f1": distribution(shuffle_gap),
        "normal_minus_off_f1": distribution(off_gap),
        "normal_changed": distribution(changed),
        "normal_corrected": distribution(corrected),
        "normal_broken": distribution(broken),
        "normal_corrected_minus_broken": distribution(
            corrected_minus_broken
        ),
        "corrected_gt_broken_events": corrected_gt_broken,
        "corrected_le_broken_events": corrected_le_broken,
        "positive_same_batch_delta_events": sum(
            value > 0 for value in anchor_delta
        ),
        "nonpositive_same_batch_delta_events": sum(
            value <= 0 for value in anchor_delta
        ),
        "normal_shuffled_gap_ge_0_01_events": sum(
            value >= 0.01 for value in shuffle_gap
        ),
        "normal_shuffled_gap_lt_0_01_events": sum(
            value < 0.01 for value in shuffle_gap
        ),
        "normal_off_gap_ge_0_01_events": sum(
            value >= 0.01 for value in off_gap
        ),
        "normal_off_gap_lt_0_01_events": sum(
            value < 0.01 for value in off_gap
        ),
        "sensitivity_threshold_crossing": threshold_crossing,
        "utility_sign_crossing": utility_sign_crossing,
        "single_event_conclusion_reversal": (
            threshold_crossing or utility_sign_crossing
        ),
        "mean_abs_costar_delta": distribution([
            event["contribution"]["mean_abs_costar_delta"]
            for event in events
        ]),
        "median_costar_over_base": distribution([
            event["contribution"]["median_costar_over_base"]
            for event in events
        ]),
        "applied_correction_rate": distribution([
            event["contribution"]["applied_correction_rate"]
            for event in events
        ]),
        "test_unique_edge_rate": distribution([
            event["test_unique_edge_rate"] for event in events
        ]),
        "stream_summaries": stream_rows,
        "mechanism_classification": classification,
    }


def format_number(value):
    return f"{float(value):.5f}"


def render_markdown(aggregate, source_root):
    lines = [
        "# COSTAR Fixed-Checkpoint Dynamic Reliability Audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: ANALYZED",
        "- Sampling protocol: `dynamic_random`",
        "- Training or tuning: none",
        "- Audit commit: `8d95c17a`",
        "- COSTAR model commit: `4abe58c6`",
        "- Streams: 2 per dataset",
        "- Events: 8 per stream, 32 total",
        "- Val/test iterations: 256 per event",
        f"- Source root: `{source_root}`",
        "- Formal baseline: registered historical initial A2",
        "",
        "## Protocol Audit",
        "",
        "All four tasks use `shuffle=True`, "
        "`val.fixed_target_panel=False`, no dedicated evaluation generator, "
        "no sampler RNG restoration, no evaluation step cap, and exactly "
        "2,048 validation plus 2,048 test iterations per stream. Every "
        "normal/shuffled/off/A2-anchor comparison has an identical paired "
        "sample count.",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Normal F1 mean +/- sd | A2 anchor mean +/- sd | Anchor "
        "delta mean | Historical A2 | Historical delta mean | "
        "Normal-shuffled | Normal-off | Changed mean | Corrected-broken mean "
        "| Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASET_ORDER:
        row = aggregate["datasets"][dataset]
        normal = row["normal_test_f1"]
        anchor = row["same_batch_a2_anchor_test_f1"]
        anchor_delta = row["same_batch_delta_vs_a2_anchor"]
        historical_delta = row["delta_vs_historical_initial_a2"]
        shuffle = row["normal_minus_shuffled_f1"]
        off = row["normal_minus_off_f1"]
        changed = row["normal_changed"]
        net = row["normal_corrected_minus_broken"]
        lines.append(
            f"| {dataset} | "
            f"{format_number(normal['mean'])} +/- "
            f"{format_number(normal['std'])} | "
            f"{format_number(anchor['mean'])} +/- "
            f"{format_number(anchor['std'])} | "
            f"{format_number(anchor_delta['mean'])} | "
            f"{format_number(row['historical_initial_a2_f1'])} | "
            f"{format_number(historical_delta['mean'])} | "
            f"{format_number(shuffle['mean'])} | "
            f"{format_number(off['mean'])} | "
            f"{format_number(changed['mean'])} | "
            f"{format_number(net['mean'])} | "
            f"`{row['mechanism_classification']}` |"
        )
    lines.extend([
        "",
        "The A2 anchor is the same-batch pre-COSTAR margin. Comparing normal "
        "to this anchor isolates the newly trained COSTAR correction. The off "
        "condition removes all evidence, including the prototype contribution, "
        "so normal-off alone cannot be attributed to the COSTAR router.",
        "",
        "## Contribution Scale",
        "",
        "| Dataset | Mean absolute COSTAR delta | Median COSTAR/base ratio | "
        "Applied correction rate | Positive anchor delta | Corrected > broken |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_ORDER:
        row = aggregate["datasets"][dataset]
        lines.append(
            f"| {dataset} | "
            f"{format_number(row['mean_abs_costar_delta']['mean'])} | "
            f"{format_number(row['median_costar_over_base']['mean'])} | "
            f"{format_number(row['applied_correction_rate']['mean'])} | "
            f"{row['positive_same_batch_delta_events']}/16 | "
            f"{row['corrected_gt_broken_events']}/16 |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "1. Normal-shuffled/off evaluates total evidence sensitivity; "
        "normal-anchor evaluates the incremental COSTAR correction.",
        "2. A total-evidence gap with negligible normal-anchor change means "
        "the prototype/base path, not COSTAR, carries the observed effect.",
        "3. The 16 events per dataset come from two dynamic streams and one "
        "fixed checkpoint. They are descriptive repeated evaluations, not "
        "independent model seeds.",
        "4. The registered historical A2 remains the formal baseline; the "
        "same-batch anchor is a mechanism diagnostic.",
        "",
        "## Decision Gate",
        "",
        f"- Useful-aligned datasets: {aggregate['classification_counts'].get('useful_aligned_evidence', 0)}/2",
        f"- Sensitive-but-harmful datasets: {aggregate['classification_counts'].get('sensitive_but_harmful', 0)}/2",
        f"- Used-but-unaligned datasets: {aggregate['classification_counts'].get('used_but_unaligned', 0)}/2",
        f"- Inactive datasets: {aggregate['classification_counts'].get('inactive_evidence', 0)}/2",
        f"- Datasets with a single-event conclusion reversal: {aggregate['single_event_reversal_datasets']}/2",
        "",
        "Final paper-level interpretation must combine this fixed-checkpoint "
        "block with A2, CET and TIER rather than selecting one favorable "
        "dataset.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifests, paths = load_manifests(args.input_root)
    grouped = validate_manifests(manifests, paths)
    datasets = {
        dataset: aggregate_dataset(dataset, grouped[dataset])
        for dataset in DATASET_ORDER
    }
    classification_counts = {}
    for row in datasets.values():
        label = row["mechanism_classification"]
        classification_counts[label] = (
            classification_counts.get(label, 0) + 1
        )
    aggregate = {
        "experiment": "costar_fixed_checkpoint_dynamic_reliability",
        "sampling_protocol": "dynamic_random",
        "manifest_count": len(manifests),
        "event_count": sum(row["events"] for row in datasets.values()),
        "dataset_count": len(datasets),
        "manifest_paths": [str(path) for path in paths],
        "datasets": datasets,
        "classification_counts": classification_counts,
        "single_event_reversal_datasets": sum(
            row["single_event_conclusion_reversal"]
            for row in datasets.values()
        ),
    }
    args.output_json.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(
        render_markdown(aggregate, args.input_root) + "\n"
    )
    print(json.dumps({
        "manifest_count": aggregate["manifest_count"],
        "event_count": aggregate["event_count"],
        "classification_counts": classification_counts,
        "single_event_reversal_datasets": (
            aggregate["single_event_reversal_datasets"]
        ),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
