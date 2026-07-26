#!/usr/bin/env python3
"""Validate and aggregate fixed-CET dynamic reliability manifests."""

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


PAIR_ORDER = (
    ("Small-LI", "encoder_only"),
    ("Small-LI", "fusion"),
    ("Large-LI", "encoder_only"),
    ("Large-LI", "fusion"),
)
EXPECTED_STREAMS = {
    ("Small-LI", "encoder_only"): 1,
    ("Small-LI", "fusion"): 2,
    ("Large-LI", "encoder_only"): 1,
    ("Large-LI", "fusion"): 3,
}


def load_manifests(root):
    paths = sorted(root.glob("*/reliability_manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    if len(manifests) != 7:
        raise ValueError(f"expected 7 manifests, found {len(manifests)}")
    return manifests, paths


def validate_manifests(manifests, paths):
    grouped = {}
    for manifest, path in zip(manifests, paths):
        if manifest["model"] != "CET-FraudGT-v1-reliability-audit":
            raise ValueError("unexpected model")
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
        if manifest["repeats"] != 4 or len(manifest["events"]) != 4:
            raise ValueError("formal manifest does not contain four events")
        if manifest["validation_loader_iterations"] != 4 * 256:
            raise ValueError("validation iteration count differs")
        if manifest["test_loader_iterations"] != 4 * 256:
            raise ValueError("test iteration count differs")
        if manifest["git_commit"] != "33aab60f":
            raise ValueError("unexpected audit commit")
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
            samples.add(event["base_test"]["samples"])
            if len(samples) != 1:
                raise ValueError("same-batch condition samples differ")

        key = (manifest["dataset"], manifest["variant"])
        grouped.setdefault(key, []).append(manifest)

    if set(grouped) != set(PAIR_ORDER):
        raise ValueError("dataset/variant pair set differs")
    for key, rows in grouped.items():
        if len(rows) != EXPECTED_STREAMS[key]:
            raise ValueError(f"{key}: unexpected stream count")
        if len({row["audit_seed"] for row in rows}) != len(rows):
            raise ValueError(f"{key}: audit seeds are not distinct")
        if len({row["checkpoint"] for row in rows}) != 1:
            raise ValueError(f"{key}: streams use different checkpoints")
        if len({row["model_seed"] for row in rows}) != 1:
            raise ValueError(f"{key}: streams use different model seeds")
    return grouped


def aggregate_pair(dataset, variant, manifests):
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
    base_f1 = [event["base_test"]["f1"] for event in events]
    paired_delta = [
        event["same_batch_delta_vs_frozen_a2"]
        for event in events
    ]
    historical_delta = [
        event["delta_vs_initial_a2"] for event in events
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
        event["intervention"]["changed_predictions"]
        for event in events
    ]
    corrected = [
        event["intervention"]["corrected_predictions"]
        for event in events
    ]
    broken = [
        event["intervention"]["broken_predictions"]
        for event in events
    ]
    net = [
        event["intervention"]["corrected_minus_broken"]
        for event in events
    ]
    corrected_gt_broken = sum(
        left > right for left, right in zip(corrected, broken)
    )
    corrected_le_broken = len(events) - corrected_gt_broken
    classification = classify_mechanism(
        statistics.fmean(shuffle_gap),
        statistics.fmean(off_gap),
        statistics.fmean(paired_delta),
        corrected_gt_broken,
        corrected_le_broken,
        statistics.fmean(changed),
        len(events),
    )
    threshold_crossing = (
        min(shuffle_gap) < 0.01 <= max(shuffle_gap)
    )
    utility_sign_crossing = min(paired_delta) < 0 < max(paired_delta)
    return {
        "dataset": dataset,
        "variant": variant,
        "streams": len(manifests),
        "events": len(events),
        "model_seed": manifests[0]["model_seed"],
        "audit_seeds": sorted(
            manifest["audit_seed"] for manifest in manifests
        ),
        "git_commit": manifests[0]["git_commit"],
        "checkpoint": manifests[0]["checkpoint"],
        "checkpoint_epoch": manifests[0]["checkpoint_epoch"],
        "normal_test_f1": distribution(normal_f1),
        "shuffled_test_f1": distribution(shuffled_f1),
        "off_test_f1": distribution(off_f1),
        "same_batch_frozen_a2_test_f1": distribution(base_f1),
        "same_batch_delta_vs_frozen_a2": distribution(paired_delta),
        "historical_initial_a2_f1": historical_values[0],
        "delta_vs_historical_initial_a2": distribution(
            historical_delta
        ),
        "normal_minus_shuffled_f1": distribution(shuffle_gap),
        "normal_minus_off_f1": distribution(off_gap),
        "changed_predictions": distribution(changed),
        "corrected_predictions": distribution(corrected),
        "broken_predictions": distribution(broken),
        "corrected_minus_broken": distribution(net),
        "corrected_gt_broken_events": corrected_gt_broken,
        "corrected_le_broken_events": corrected_le_broken,
        "positive_same_batch_delta_events": sum(
            value > 0 for value in paired_delta
        ),
        "normal_shuffled_gap_ge_0_01_events": sum(
            value >= 0.01 for value in shuffle_gap
        ),
        "sensitivity_threshold_crossing": threshold_crossing,
        "utility_sign_crossing": utility_sign_crossing,
        "single_event_conclusion_reversal": (
            threshold_crossing or utility_sign_crossing
        ),
        "coverage_rate": distribution([
            event["diagnostics"]["coverage_rate"]
            for event in events
        ]),
        "evidence_norm_mean": distribution([
            event["diagnostics"]["evidence_norm"]["mean"]
            for event in events
        ]),
        "fusion_gain_norm_mean": distribution([
            event["diagnostics"]["fusion_gain_norm"]["mean"]
            for event in events
        ]),
        "test_unique_edge_rate": distribution([
            event["test_unique_edge_rate"] for event in events
        ]),
        "mechanism_classification": classification,
    }


def format_number(value):
    return f"{float(value):.5f}"


def render_markdown(aggregate, source_root):
    lines = [
        "# CET Fixed-Checkpoint Dynamic Reliability Audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: ANALYZED",
        "- Sampling protocol: `dynamic_random`",
        "- Training or tuning: none",
        "- Audit commit: `33aab60f`",
        "- Events: 28 total",
        "- Val/test iterations: 256 per event",
        f"- Source root: `{source_root}`",
        "- Formal baseline: registered historical initial A2",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Variant | Events | Normal F1 mean +/- sd | Frozen A2 "
        "mean +/- sd | Paired delta mean | Normal-shuffled | Normal-off | "
        "Changed mean | Corrected-broken mean | Classification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in PAIR_ORDER:
        row = aggregate["pairs"]["|".join(key)]
        normal = row["normal_test_f1"]
        base = row["same_batch_frozen_a2_test_f1"]
        paired = row["same_batch_delta_vs_frozen_a2"]
        shuffle = row["normal_minus_shuffled_f1"]
        off = row["normal_minus_off_f1"]
        changed = row["changed_predictions"]
        net = row["corrected_minus_broken"]
        lines.append(
            f"| {key[0]} | {key[1]} | {row['events']} | "
            f"{format_number(normal['mean'])} +/- "
            f"{format_number(normal['std'])} | "
            f"{format_number(base['mean'])} +/- "
            f"{format_number(base['std'])} | "
            f"{format_number(paired['mean'])} | "
            f"{format_number(shuffle['mean'])} | "
            f"{format_number(off['mean'])} | "
            f"{format_number(changed['mean'])} | "
            f"{format_number(net['mean'])} | "
            f"`{row['mechanism_classification']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The fusion and encoder-only checkpoints are separate fixed variants, "
        "not ablation seeds. Normal-shuffled measures evidence alignment; "
        "normal-off measures evidence presence; same-batch delta and "
        "corrected-minus-broken measure utility. Repeated event statistics "
        "are descriptive and do not substitute for independent model seeds.",
        "",
        "## Decision Gate",
        "",
        f"- Useful-aligned pairs: {aggregate['classification_counts'].get('useful_aligned_evidence', 0)}/4",
        f"- Sensitive-but-harmful pairs: {aggregate['classification_counts'].get('sensitive_but_harmful', 0)}/4",
        f"- Used-but-unaligned pairs: {aggregate['classification_counts'].get('used_but_unaligned', 0)}/4",
        f"- Inactive pairs: {aggregate['classification_counts'].get('inactive_evidence', 0)}/4",
        f"- Pairs with a single-event conclusion reversal: {aggregate['single_event_reversal_pairs']}/4",
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
    pairs = {
        "|".join(key): aggregate_pair(
            key[0], key[1], grouped[key]
        )
        for key in PAIR_ORDER
    }
    classification_counts = {}
    for row in pairs.values():
        label = row["mechanism_classification"]
        classification_counts[label] = (
            classification_counts.get(label, 0) + 1
        )
    aggregate = {
        "experiment": "cet_fixed_checkpoint_dynamic_reliability",
        "sampling_protocol": "dynamic_random",
        "manifest_count": len(manifests),
        "event_count": sum(row["events"] for row in pairs.values()),
        "pair_count": len(pairs),
        "manifest_paths": [str(path) for path in paths],
        "pairs": pairs,
        "classification_counts": classification_counts,
        "single_event_reversal_pairs": sum(
            row["single_event_conclusion_reversal"]
            for row in pairs.values()
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
        "single_event_reversal_pairs": (
            aggregate["single_event_reversal_pairs"]
        ),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

