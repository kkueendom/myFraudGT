#!/usr/bin/env python3
"""Validate and aggregate fixed-TIER dynamic reliability manifests."""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.summarize_a2_dynamic_stability import distribution


PAIR_ORDER = (
    ("Small-LI", "recent"),
    ("Small-LI", "role_motif"),
    ("Large-LI", "recent"),
    ("Large-LI", "role_motif"),
)


def load_manifests(root):
    paths = sorted(root.glob("*/tier_reliability_manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    if len(manifests) != 8:
        raise ValueError(f"expected 8 manifests, found {len(manifests)}")
    return manifests, paths


def validate_manifests(manifests, paths):
    grouped = {}
    for manifest, path in zip(manifests, paths):
        if manifest["experiment"] != (
            "tier_fixed_checkpoint_dynamic_reliability"
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
        if manifest["git_commit"] != "32b3aab4":
            raise ValueError("unexpected audit commit")
        if manifest["model"] != "TIER-EvidenceOnly":
            raise ValueError("unexpected model")
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
            samples.add(
                event["interventions"]["normal"]["samples"]
            )
            if len(samples) != 1:
                raise ValueError("same-batch condition samples differ")
            if not 0 < event["test_paired_retention_rate"] <= 1:
                raise ValueError("invalid paired retention rate")

        key = (
            manifest["dataset"],
            manifest["evidence_selection"],
        )
        grouped.setdefault(key, []).append(manifest)

    if set(grouped) != set(PAIR_ORDER):
        raise ValueError("dataset/selection pair set differs")
    for key, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"{key}: expected two streams")
        if len({row["audit_seed"] for row in rows}) != 2:
            raise ValueError(f"{key}: audit seeds are not distinct")
        if len({row["checkpoint"] for row in rows}) != 1:
            raise ValueError(f"{key}: streams use different checkpoints")
        if len({row["model_seed"] for row in rows}) != 1:
            raise ValueError(f"{key}: streams use different model seeds")
    return grouped


def classify_mechanism(
    normal_minus_shuffled_mean,
    normal_minus_off_mean,
    same_batch_delta_mean,
    corrected_gt_broken_events,
    corrected_le_broken_events,
    mean_changed,
    event_count,
):
    if (
        normal_minus_shuffled_mean >= 0.01
        and same_batch_delta_mean > 0
        and corrected_gt_broken_events >= 12
        and mean_changed >= 50
    ):
        return "useful_aligned_evidence"
    if (
        normal_minus_shuffled_mean >= 0.01
        and (
            same_batch_delta_mean <= 0
            or corrected_le_broken_events >= event_count / 2
        )
    ):
        return "sensitive_but_harmful"
    if (
        normal_minus_off_mean >= 0.01
        and normal_minus_shuffled_mean < 0.01
    ):
        return "used_but_unaligned"
    if (
        normal_minus_shuffled_mean < 0.01
        and normal_minus_off_mean < 0.01
    ):
        return "inactive_evidence"
    return "mixed_or_indeterminate"


def aggregate_pair(dataset, selection, manifests):
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
        left - right for left, right in zip(corrected, broken)
    ]
    corrected_gt_broken = sum(
        left > right for left, right in zip(corrected, broken)
    )
    corrected_le_broken = len(events) - corrected_gt_broken
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
            "same_batch_delta_mean": statistics.fmean(
                event["same_batch_delta_vs_frozen_a2"]
                for event in stream_events
            ),
            "normal_minus_shuffled_f1_mean": statistics.fmean(
                event["normal_minus_shuffled_f1"]
                for event in stream_events
            ),
        })

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
        "evidence_selection": selection,
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
        "normal_changed": distribution(changed),
        "normal_corrected": distribution(corrected),
        "normal_broken": distribution(broken),
        "normal_corrected_minus_broken": distribution(
            corrected_minus_broken
        ),
        "corrected_gt_broken_events": corrected_gt_broken,
        "corrected_le_broken_events": corrected_le_broken,
        "positive_same_batch_delta_events": sum(
            value > 0 for value in paired_delta
        ),
        "nonpositive_same_batch_delta_events": sum(
            value <= 0 for value in paired_delta
        ),
        "normal_shuffled_gap_ge_0_01_events": sum(
            value >= 0.01 for value in shuffle_gap
        ),
        "normal_shuffled_gap_lt_0_01_events": sum(
            value < 0.01 for value in shuffle_gap
        ),
        "sensitivity_threshold_crossing": threshold_crossing,
        "utility_sign_crossing": utility_sign_crossing,
        "single_event_conclusion_reversal": (
            threshold_crossing or utility_sign_crossing
        ),
        "coverage_rate": distribution([
            event["coverage_rate"] for event in events
        ]),
        "test_paired_retention_rate": distribution([
            event["test_paired_retention_rate"] for event in events
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
        "# TIER Fixed-Checkpoint Dynamic Reliability Audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: ANALYZED",
        "- Sampling protocol: `dynamic_random`",
        "- Training or tuning: none",
        "- Audit commit: `32b3aab4`",
        "- Streams: 2 per model/dataset pair",
        "- Events: 8 per stream, 64 total",
        "- Val/test iterations: 256 per event",
        f"- Source root: `{source_root}`",
        "- Formal baseline: registered historical initial A2",
        "",
        "## Protocol Audit",
        "",
        "All eight tasks use `shuffle=True`, "
        "`val.fixed_target_panel=False`, no dedicated evaluation generator, "
        "no sampler RNG restoration, no evaluation step cap, and exactly "
        "2,048 validation plus 2,048 test iterations per stream. Every "
        "normal/shuffled/off/frozen-A2 comparison has an identical paired "
        "sample count.",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Evidence | Epoch | Normal F1 mean +/- sd | Frozen A2 "
        "mean +/- sd | Paired delta mean | Historical A2 | Historical delta "
        "mean | Normal-shuffled | Normal-off | Changed mean | "
        "Corrected-broken mean | Classification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in PAIR_ORDER:
        row = aggregate["pairs"]["|".join(key)]
        normal = row["normal_test_f1"]
        base = row["same_batch_frozen_a2_test_f1"]
        paired = row["same_batch_delta_vs_frozen_a2"]
        historical_delta = row["delta_vs_historical_initial_a2"]
        shuffle = row["normal_minus_shuffled_f1"]
        off = row["normal_minus_off_f1"]
        changed = row["normal_changed"]
        net = row["normal_corrected_minus_broken"]
        lines.append(
            f"| {key[0]} | {key[1]} | {row['checkpoint_epoch']} | "
            f"{format_number(normal['mean'])} +/- "
            f"{format_number(normal['std'])} | "
            f"{format_number(base['mean'])} +/- "
            f"{format_number(base['std'])} | "
            f"{format_number(paired['mean'])} | "
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
        "The frozen epoch-499 A2 score is a same-batch diagnostic only. The "
        "registered historical initial A2 remains the formal baseline because "
        "the available checkpoint epoch differs from the historical "
        "Val-selected epoch.",
        "",
        "## Event Consistency",
        "",
        "| Dataset | Evidence | Sensitivity >=0.01 | Positive utility | "
        "Corrected > broken | Sensitivity threshold crossed? | Utility sign "
        "crossed? |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for key in PAIR_ORDER:
        row = aggregate["pairs"]["|".join(key)]
        lines.append(
            f"| {key[0]} | {key[1]} | "
            f"{row['normal_shuffled_gap_ge_0_01_events']}/16 | "
            f"{row['positive_same_batch_delta_events']}/16 | "
            f"{row['corrected_gt_broken_events']}/16 | "
            f"{str(row['sensitivity_threshold_crossing']).lower()} | "
            f"{str(row['utility_sign_crossing']).lower()} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "1. A large normal-shuffled gap establishes that the fixed evidence "
        "classifier uses transaction-history evidence; it does not establish "
        "that replacing A2 is beneficial.",
        "2. Corrective utility requires a positive same-batch F1 delta and "
        "more corrected than broken predictions. These quantities are "
        "reported separately from sensitivity.",
        "3. The 16 events are repeated dynamic evaluations from two streams, "
        "not 16 independent model seeds. Mean and standard deviation are "
        "descriptive; no IID p-value is claimed.",
        "4. A threshold or sign crossing means a one-event conclusion can "
        "reverse under dynamic sampling. It is not itself proof of a model "
        "effect.",
        "5. Large-LI paired retention is reported explicitly because repeated "
        "loader requests can collapse to fewer unique A2-scored target edges. "
        "All retained targets remain strictly aligned across conditions.",
        "",
        "## Decision Gate",
        "",
        f"- Useful-aligned pairs: {aggregate['classification_counts'].get('useful_aligned_evidence', 0)}/4",
        f"- Sensitive-but-harmful pairs: {aggregate['classification_counts'].get('sensitive_but_harmful', 0)}/4",
        f"- Used-but-unaligned pairs: {aggregate['classification_counts'].get('used_but_unaligned', 0)}/4",
        f"- Inactive pairs: {aggregate['classification_counts'].get('inactive_evidence', 0)}/4",
        f"- Pairs with a single-event conclusion reversal: {aggregate['single_event_reversal_pairs']}/4",
        "",
        "Paper-level advancement remains conditional on the preregistered "
        "multi-model gate after combining A2, CET, TIER and COSTAR evidence.",
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
        "experiment": "tier_fixed_checkpoint_dynamic_reliability",
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
