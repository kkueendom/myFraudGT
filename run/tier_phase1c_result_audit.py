#!/usr/bin/env python3
"""Audit the complete TIER Phase 1c intervention diagnostic matrix."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DATASETS = ("Small-LI", "Large-LI")
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")
BASELINE = {
    "Small-LI": {
        "val_selected_test_f1": 0.46247,
        "raw_best_test_f1": 0.50667,
    },
    "Large-LI": {
        "val_selected_test_f1": 0.30108,
        "raw_best_test_f1": 0.44720,
    },
}
GATE = {
    "min_correction_rate_on_a2_errors": 0.10,
    "min_corrected_to_broken_ratio": 1.5,
    "min_changed_predictions": 50,
    "min_changed_fraction_of_a2_errors": 0.10,
}


def format_number(value):
    return "n/a" if value is None else f"{value:.5f}"


def recompute_gate(stats, require_f1_gain):
    corrected = int(stats["corrected_predictions"])
    broken = int(stats["broken_predictions"])
    ratio = stats["corrected_to_broken_ratio"]
    no_breaks = corrected > 0 and broken == 0
    ratio_passed = no_breaks or (
        ratio is not None
        and float(ratio) >= GATE["min_corrected_to_broken_ratio"]
    )
    required_changed = max(
        GATE["min_changed_predictions"],
        math.ceil(
            GATE["min_changed_fraction_of_a2_errors"]
            * int(stats["a2_errors"])
        ),
    )
    passed = (
        float(stats["correction_rate_on_a2_errors"])
        >= GATE["min_correction_rate_on_a2_errors"]
        and ratio_passed
        and corrected > broken
        and int(stats["changed_predictions"]) >= required_changed
    )
    if require_f1_gain:
        passed &= float(stats["delta_paired_f1"]) > 0
    return passed, required_changed


def audit_loader(rows):
    if [row["split"] for row in rows] != ["train", "val", "test"]:
        raise ValueError("loader audit does not contain train/val/test")
    for row in rows:
        if row.get("shuffle") is not True:
            raise ValueError("one loader is not shuffled")
        if (
            row.get("loader_generator") is not None
            or row.get("sampler_generator") is not None
        ):
            raise ValueError("dedicated evaluation generator detected")


def audit_manifest(path, expected_commit, parent_commit):
    item = json.loads(path.read_text())
    if item.get("sampling_protocol") != "dynamic_random":
        raise ValueError(f"{path}: protocol is not dynamic_random")
    if not str(item.get("git_commit", "")).startswith(expected_commit):
        raise ValueError(f"{path}: unexpected experiment commit")
    if not str(item.get("parent_evidence_commit", "")).startswith(
        parent_commit
    ):
        raise ValueError(f"{path}: unexpected parent evidence commit")
    audit_loader(item["loader_audit"])
    dataset = item["dataset"]
    if dataset not in BASELINE:
        raise ValueError(f"{path}: unexpected dataset")
    expected_baseline = BASELINE[dataset]
    if (
        item["initial_a2_val_selected_test_f1"]
        != expected_baseline["val_selected_test_f1"]
        or item["initial_a2_raw_best_test_f1"]
        != expected_baseline["raw_best_test_f1"]
    ):
        raise ValueError(f"{path}: initial A2 reference differs")
    if item.get("raw_best_test_f1") is not None:
        raise ValueError(f"{path}: diagnostic must not report raw-best")
    validation_passed, validation_required = recompute_gate(
        item["validation_statistics"], require_f1_gain=False)
    test_passed, test_required = recompute_gate(
        item["test_statistics"], require_f1_gain=True)
    decision = "pass" if validation_passed and test_passed else "fail"
    if item.get("validation_policy_passed") is not validation_passed:
        raise ValueError(f"{path}: stored validation gate is inconsistent")
    if item.get("qualification_decision") != decision:
        raise ValueError(f"{path}: stored qualification is inconsistent")
    if (
        int(item["validation_required_changed_predictions"])
        != validation_required
        or int(item["test_required_changed_predictions"]) != test_required
    ):
        raise ValueError(f"{path}: changed-prediction gate differs")
    if (
        int(item["validation_eligible_policies"]) == 0
        and validation_passed
    ):
        raise ValueError(f"{path}: selected pass without eligible policy")
    stats = item["test_statistics"]
    return {
        "dataset": dataset,
        "family": item["evidence_family"],
        "selection": item["evidence_selection"],
        "seed": int(item["seed"]),
        "commit": item["git_commit"],
        "parent_commit": item["parent_evidence_commit"],
        "policy_direction": item["selected_policy"]["direction"],
        "validation_eligible_policies": int(
            item["validation_eligible_policies"]),
        "validation_passed": validation_passed,
        "validation_corrected": int(
            item["validation_statistics"]["corrected_predictions"]),
        "validation_broken": int(
            item["validation_statistics"]["broken_predictions"]),
        "validation_changed": int(
            item["validation_statistics"]["changed_predictions"]),
        "validation_required_changed": validation_required,
        "test_passed": test_passed,
        "qualification": decision,
        "corrected": int(stats["corrected_predictions"]),
        "broken": int(stats["broken_predictions"]),
        "corrected_minus_broken": int(stats["corrected_minus_broken"]),
        "corrected_to_broken": stats["corrected_to_broken_ratio"],
        "correction_rate": float(
            stats["correction_rate_on_a2_errors"]),
        "changed": int(stats["changed_predictions"]),
        "required_changed": test_required,
        "a2_paired_f1": float(stats["a2_same_batch_f1"]),
        "routed_paired_f1": float(stats["routed_same_batch_f1"]),
        "paired_delta": float(stats["delta_paired_f1"]),
        "initial_a2_reference": expected_baseline[
            "val_selected_test_f1"
        ],
        "delta_vs_initial_a2": float(
            item["delta_vs_initial_a2_val_selected_f1"]
        ),
        "scope": item["result_scope"],
    }


def family_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    summaries = {}
    for family in FAMILIES:
        items = grouped[family]
        small_passes = sum(
            row["qualification"] == "pass"
            for row in items
            if row["dataset"] == "Small-LI"
        )
        large_passes = sum(
            row["qualification"] == "pass"
            for row in items
            if row["dataset"] == "Large-LI"
        )
        qualified = sum(
            row["qualification"] == "pass" for row in items)
        if small_passes and large_passes:
            decision = "cross_scale_pass"
        elif qualified:
            decision = "scale_dependent"
        elif any(row["validation_passed"] for row in items):
            decision = "validation_only"
        else:
            decision = "not_separable_at_registered_coverage"
        ratios = [
            float(row["corrected_to_broken"])
            for row in items
            if row["corrected_to_broken"] is not None
        ]
        summaries[family] = {
            "tasks": len(items),
            "validation_passes": sum(
                row["validation_passed"] for row in items),
            "test_gate_passes": sum(
                row["test_passed"] for row in items),
            "qualified_tasks": qualified,
            "small_li_passes": small_passes,
            "large_li_passes": large_passes,
            "mean_paired_f1_delta": sum(
                row["paired_delta"] for row in items
            ) / len(items),
            "best_test_corrected_to_broken": max(ratios, default=None),
            "max_test_changed": max(row["changed"] for row in items),
            "decision": decision,
        }
    return summaries


def render_markdown(payload):
    lines = [
        "# TIER Phase 1c High-Precision Intervention Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: EXECUTED manifests audited",
        "- Sampling Protocol: `dynamic_random`",
        f"- Expected Commit: `{payload['expected_commit']}`",
        f"- Parent Evidence Commit: `{payload['parent_commit']}`",
        "- Fixed-panel A2: excluded",
        "- Scope: paired same-batch mechanism diagnostic",
        "",
        "## Per-Task Results",
        "",
        "| Dataset | Family | Selection | Direction | Val eligible | "
        "Val corrected/broken | Val changed/required | Test corrected/broken "
        "| Test ratio | Test changed/required | A2 paired F1 | Routed paired "
        "F1 | Paired delta | Gate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {dataset} | {family} | {selection} | {direction} | "
            "{eligible} | {val_corrected}/{val_broken} | "
            "{val_changed}/{val_required} | {corrected}/{broken} | "
            "{ratio} | {changed}/{required} | {a2} | {routed} | "
            "{delta} | {gate} |".format(
                dataset=row["dataset"],
                family=row["family"],
                selection=row["selection"],
                direction=row["policy_direction"],
                eligible=row["validation_eligible_policies"],
                val_corrected=row["validation_corrected"],
                val_broken=row["validation_broken"],
                val_changed=row["validation_changed"],
                val_required=row["validation_required_changed"],
                corrected=row["corrected"],
                broken=row["broken"],
                ratio=format_number(row["corrected_to_broken"]),
                changed=row["changed"],
                required=row["required_changed"],
                a2=format_number(row["a2_paired_f1"]),
                routed=format_number(row["routed_paired_f1"]),
                delta=format_number(row["paired_delta"]),
                gate=row["qualification"],
            )
        )
    lines.extend([
        "",
        "## Family Summary",
        "",
        "| Family | Tasks | Validation passes | Test-gate passes | "
        "Qualified | Small passes | Large passes | Mean paired delta | "
        "Best test ratio | Max test changed | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for family in FAMILIES:
        item = payload["family_summaries"][family]
        lines.append(
            f"| {family} | {item['tasks']} | "
            f"{item['validation_passes']} | {item['test_gate_passes']} | "
            f"{item['qualified_tasks']} | {item['small_li_passes']} | "
            f"{item['large_li_passes']} | "
            f"{format_number(item['mean_paired_f1_delta'])} | "
            f"{format_number(item['best_test_corrected_to_broken'])} | "
            f"{item['max_test_changed']} | {item['decision']} |"
        )
    lines.extend([
        "",
        "## Gate Conclusion",
        "",
        payload["conclusion"],
        "",
        "The initial A2 delta is retained in JSON for experiment bookkeeping. "
        "Because this phase evaluates only the A2-scored paired subset, it "
        "must not replace the full-model headline table.",
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=12)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.root.glob("*/phase1c_manifest.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(
            f"expected {args.expected_tasks} manifests, found {len(paths)}")
    rows = [
        audit_manifest(path, args.expected_commit, args.parent_commit)
        for path in paths
    ]
    keys = {
        (row["dataset"], row["family"], row["selection"])
        for row in rows
    }
    expected_keys = {
        (dataset, family, selection)
        for dataset in DATASETS
        for family in FAMILIES
        for selection in SELECTIONS
    }
    if keys != expected_keys:
        raise RuntimeError("manifest matrix is incomplete or duplicated")
    rows.sort(key=lambda row: (
        row["family"], row["dataset"], row["selection"]))
    summaries = family_summaries(rows)
    cross_scale = [
        family
        for family, item in summaries.items()
        if item["decision"] == "cross_scale_pass"
    ]
    conclusion = (
        "Cross-scale routing qualification passed for: "
        + ", ".join(cross_scale)
        if cross_scale
        else (
            "No evidence family produced a validation-eligible intervention "
            "policy that passed the locked test gate on both scales. "
            "Learned ErrorRouter and CrossFusion are not authorized by "
            "Phase 1c."
        )
    )
    payload = {
        "sampling_protocol": "dynamic_random",
        "expected_commit": args.expected_commit,
        "parent_commit": args.parent_commit,
        "scope": "paired_same_batch_mechanism_diagnostic",
        "rows": rows,
        "family_summaries": summaries,
        "conclusion": conclusion,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(payload))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
