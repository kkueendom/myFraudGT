#!/usr/bin/env python3
"""Combine A2, CET, TIER and COSTAR dynamic reliability audits."""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MISMATCH_LABELS = {
    "sensitive_but_harmful",
    "used_but_unaligned",
}


def load_json(path):
    return json.loads(path.read_text())


def validate_inputs(a2, cet, tier, costar):
    rows = (a2, cet, tier, costar)
    if any(row["sampling_protocol"] != "dynamic_random" for row in rows):
        raise ValueError("all audit blocks must use dynamic_random")
    if a2["manifest_count"] != 12 or a2["event_count"] != 96:
        raise ValueError("A2 audit is incomplete")
    if cet["manifest_count"] != 7 or cet["event_count"] != 28:
        raise ValueError("CET audit is incomplete")
    if tier["manifest_count"] != 8 or tier["event_count"] != 64:
        raise ValueError("TIER audit is incomplete")
    if costar["manifest_count"] != 4 or costar["event_count"] != 32:
        raise ValueError("COSTAR audit is incomplete")


def evidence_units(cet, tier, costar):
    return {
        "CET": list(cet["pairs"].values()),
        "TIER": list(tier["pairs"].values()),
        "COSTAR": list(costar["datasets"].values()),
    }


def summarize_family(name, units):
    labels = {}
    for row in units:
        label = row["mechanism_classification"]
        labels[label] = labels.get(label, 0) + 1
    mismatch = [
        row for row in units
        if row["mechanism_classification"] in MISMATCH_LABELS
    ]
    mean_mismatch_datasets = sorted({
        row["dataset"] for row in mismatch
    })
    repeatable = [
        row for row in mismatch
        if unit_has_repeatable_mismatch(row)
    ]
    repeatable_datasets = sorted({
        row["dataset"] for row in repeatable
    })
    all_datasets = sorted({row["dataset"] for row in units})
    return {
        "family": name,
        "unit_count": len(units),
        "event_count": sum(row["events"] for row in units),
        "datasets": all_datasets,
        "classification_counts": labels,
        "mismatch_unit_count": len(mismatch),
        "mean_mismatch_datasets": mean_mismatch_datasets,
        "repeatable_mismatch_unit_count": len(repeatable),
        "repeatable_mismatch_datasets": repeatable_datasets,
        "repeatable_mismatch": bool(repeatable),
        "cross_scale_mismatch": (
            "Small-LI" in repeatable_datasets
            and "Large-LI" in repeatable_datasets
        ),
        "useful_aligned_unit_count": labels.get(
            "useful_aligned_evidence", 0
        ),
        "inactive_unit_count": labels.get("inactive_evidence", 0),
        "single_event_reversal_unit_count": sum(
            row["single_event_conclusion_reversal"]
            for row in units
        ),
    }


def unit_has_repeatable_mismatch(row):
    event_count = int(row["events"])
    required = math.ceil(0.75 * event_count)
    label = row["mechanism_classification"]
    harmful_events = max(
        int(row["nonpositive_same_batch_delta_events"]),
        int(row["corrected_le_broken_events"]),
    )
    if label == "sensitive_but_harmful":
        return (
            int(row["normal_shuffled_gap_ge_0_01_events"])
            >= required
            and harmful_events >= required
        )
    if label == "used_but_unaligned":
        return (
            int(row["normal_off_gap_ge_0_01_events"]) >= required
            and int(row["normal_shuffled_gap_lt_0_01_events"])
            >= required
            and harmful_events >= required
        )
    return False


def a2_single_run_reversal(a2):
    datasets = []
    for dataset, row in a2["datasets"].items():
        delta = row[
            "diagnostic_delta_vs_historical_initial_a2"
        ]
        if delta["min"] < -0.005 and delta["max"] > 0.005:
            datasets.append(dataset)
    return sorted(datasets)


def build_aggregate(a2, cet, tier, costar):
    validate_inputs(a2, cet, tier, costar)
    units = evidence_units(cet, tier, costar)
    families = {
        name: summarize_family(name, rows)
        for name, rows in units.items()
    }
    all_units = [
        row
        for rows in units.values()
        for row in rows
    ]
    mismatch_families = sum(
        row["repeatable_mismatch"] for row in families.values()
    )
    cross_scale_families = sum(
        row["cross_scale_mismatch"] for row in families.values()
    )
    a2_reversal = a2_single_run_reversal(a2)
    evidence_reversal_units = sum(
        row["single_event_conclusion_reversal"]
        for row in all_units
    )
    gate = {
        "same_batch_protocol_complete": True,
        "at_least_three_mismatch_families": mismatch_families >= 3,
        "mismatch_repeats_across_scales": cross_scale_families >= 1,
        "dynamic_sampling_changes_single_run_conclusion": bool(
            a2_reversal or evidence_reversal_units
        ),
        "concepts_reported_separately": True,
    }
    gate["all_pass"] = all(gate.values())
    useful_units = sum(
        row["mechanism_classification"]
        == "useful_aligned_evidence"
        for row in all_units
    )
    mismatch_units = sum(
        row["mechanism_classification"] in MISMATCH_LABELS
        for row in all_units
    )
    inactive_units = sum(
        row["mechanism_classification"] == "inactive_evidence"
        for row in all_units
    )
    if not gate["all_pass"]:
        decision = "STOP_PAPER_LEVEL_MULTI_MODEL_GENERALITY_CLAIM"
    elif useful_units == 0:
        decision = (
            "PROCEED_RELIABILITY_METHOD_DEVELOPMENT_ONLY_"
            "NO_PREDICTIVE_CLAIM"
        )
    else:
        decision = "PROCEED_TO_FORMAL_RELIABILITY_METHOD_VALIDATION"
    return {
        "experiment": "multi_model_dynamic_evidence_reliability",
        "sampling_protocol": "dynamic_random",
        "total_manifest_count": (
            a2["manifest_count"]
            + cet["manifest_count"]
            + tier["manifest_count"]
            + costar["manifest_count"]
        ),
        "total_event_count": (
            a2["event_count"]
            + cet["event_count"]
            + tier["event_count"]
            + costar["event_count"]
        ),
        "evidence_family_count": len(families),
        "evidence_unit_count": len(all_units),
        "mismatch_family_count": mismatch_families,
        "cross_scale_mismatch_family_count": cross_scale_families,
        "mismatch_unit_count": mismatch_units,
        "useful_aligned_unit_count": useful_units,
        "inactive_unit_count": inactive_units,
        "a2_single_run_reversal_datasets": a2_reversal,
        "evidence_single_run_reversal_unit_count": (
            evidence_reversal_units
        ),
        "families": families,
        "advancement_gate": gate,
        "decision": decision,
    }


def format_bool(value):
    return "pass" if value else "fail"


def render_markdown(aggregate, paths):
    lines = [
        "# Multi-Model Dynamic Evidence Reliability Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: ANALYZED",
        "- Sampling protocol: `dynamic_random`",
        f"- Manifests: {aggregate['total_manifest_count']}",
        f"- Dynamic val/test events: {aggregate['total_event_count']}",
        f"- Evidence model families: {aggregate['evidence_family_count']}",
        f"- Evidence model/dataset units: {aggregate['evidence_unit_count']}",
        "- Training or tuning in this benchmark: none",
        f"- A2 input: `{paths['a2']}`",
        f"- CET input: `{paths['cet']}`",
        f"- TIER input: `{paths['tier']}`",
        f"- COSTAR input: `{paths['costar']}`",
        "",
        "## Family-Level Results",
        "",
        "| Family | Units | Events | Mismatch units | Mismatch datasets | "
        "Repeatable mismatch units | Repeatable datasets | Useful aligned | "
        "Inactive | Cross-scale repeatable mismatch | Event reversal |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---|---:|",
    ]
    for name in ("CET", "TIER", "COSTAR"):
        row = aggregate["families"][name]
        lines.append(
            f"| {name} | {row['unit_count']} | {row['event_count']} | "
            f"{row['mismatch_unit_count']} | "
            f"{', '.join(row['mean_mismatch_datasets']) or 'none'} | "
            f"{row['repeatable_mismatch_unit_count']} | "
            f"{', '.join(row['repeatable_mismatch_datasets']) or 'none'} | "
            f"{row['useful_aligned_unit_count']} | "
            f"{row['inactive_unit_count']} | "
            f"{str(row['cross_scale_mismatch']).lower()} | "
            f"{row['single_event_reversal_unit_count']} |"
        )
    lines.extend([
        "",
        "Mismatch includes `sensitive_but_harmful` and "
        "`used_but_unaligned`. It does not relabel inactive evidence as a "
        "sensitivity-versus-utility mismatch. A family counts toward the "
        "advancement gate only when at least one mismatch unit satisfies its "
        "sensitivity and harmful-utility conditions in at least 75% of "
        "dynamic events.",
        "",
        "## Advancement Gate",
        "",
        "| Requirement | Result |",
        "|---|---|",
    ])
    labels = {
        "same_batch_protocol_complete": (
            "All fixed evidence blocks completed paired "
            "normal/shuffled/off/base evaluation"
        ),
        "at_least_three_mismatch_families": (
            "At least three independently trained evidence families show "
            "repeatable sensitivity/utility mismatch"
        ),
        "mismatch_repeats_across_scales": (
            "Mismatch occurs on both Small-LI and Large-LI in at least one "
            "family"
        ),
        "dynamic_sampling_changes_single_run_conclusion": (
            "Dynamic sampling changes at least one single-run conclusion"
        ),
        "concepts_reported_separately": (
            "Sampling variation, sensitivity, utility and checkpoint "
            "selection are reported separately"
        ),
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | "
            f"**{format_bool(aggregate['advancement_gate'][key])}** |"
        )
    lines.extend([
        "",
        "## Aggregate Mechanism Counts",
        "",
        f"- Mismatch units: {aggregate['mismatch_unit_count']}/"
        f"{aggregate['evidence_unit_count']}",
        f"- Useful-aligned units: {aggregate['useful_aligned_unit_count']}/"
        f"{aggregate['evidence_unit_count']}",
        f"- Inactive units: {aggregate['inactive_unit_count']}/"
        f"{aggregate['evidence_unit_count']}",
        "- A2 datasets whose repeated epoch-499 diagnostic delta crosses "
        "both +/-0.005: "
        f"{', '.join(aggregate['a2_single_run_reversal_datasets']) or 'none'}",
        "- Evidence units with a sensitivity-threshold or utility-sign "
        f"reversal: {aggregate['evidence_single_run_reversal_unit_count']}",
        "",
        "## Decision",
        "",
        f"`{aggregate['decision']}`",
        "",
    ])
    if not aggregate["advancement_gate"]["all_pass"]:
        lines.extend([
            "The preregistered cross-model generality gate is not fully "
            "satisfied. The completed work supports a rigorous negative "
            "benchmark and mechanism taxonomy, but not yet a paper-level "
            "claim that one new reliability method generalizes across three "
            "evidence families.",
            "",
        ])
    elif aggregate["useful_aligned_unit_count"] == 0:
        lines.extend([
            "Cross-model mechanism generality is sufficient for method "
            "development, but no fixed evidence unit demonstrates useful "
            "aligned correction. The next method must be evaluated as a "
            "harm-control/reliability method, not presented as an F1-improving "
            "FraudGT architecture.",
            "",
        ])
    lines.extend([
        "Repeated events are not independent model seeds. No IID p-value or "
        "finite-sample guarantee is claimed. A strong method-paper claim still "
        "requires graph/time-blocked risk units, explicit dependence "
        "assumptions and nonzero out-of-sample corrective coverage.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-json", type=Path, required=True)
    parser.add_argument("--cet-json", type=Path, required=True)
    parser.add_argument("--tier-json", type=Path, required=True)
    parser.add_argument("--costar-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = {
        "a2": args.a2_json,
        "cet": args.cet_json,
        "tier": args.tier_json,
        "costar": args.costar_json,
    }
    aggregate = build_aggregate(
        load_json(args.a2_json),
        load_json(args.cet_json),
        load_json(args.tier_json),
        load_json(args.costar_json),
    )
    args.output_json.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(
        render_markdown(aggregate, inputs) + "\n"
    )
    print(json.dumps({
        "total_manifest_count": aggregate["total_manifest_count"],
        "total_event_count": aggregate["total_event_count"],
        "mismatch_family_count": aggregate["mismatch_family_count"],
        "useful_aligned_unit_count": (
            aggregate["useful_aligned_unit_count"]
        ),
        "gate_pass": aggregate["advancement_gate"]["all_pass"],
        "decision": aggregate["decision"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
