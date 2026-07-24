#!/usr/bin/env python3
"""Audit the TIER Phase 2a-P0 directional utility probe matrix."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DATASETS = ("Small-LI", "Large-LI")
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")
BASELINE = {
    "Small-LI": (0.46247, 0.50667),
    "Large-LI": (0.30108, 0.44720),
}
CALIBRATION_GATE = {
    "min_correction_rate_on_a2_errors": 0.10,
    "min_corrected_to_broken_ratio": 1.5,
    "min_changed_predictions": 10,
    "min_changed_fraction_of_a2_errors": 0.05,
}
EVALUATION_GATE = {
    "min_correction_rate_on_a2_errors": 0.10,
    "min_corrected_to_broken_ratio": 1.5,
    "min_changed_predictions": 50,
    "min_changed_fraction_of_a2_errors": 0.10,
}


def number(value):
    return "n/a" if value is None else f"{value:.5f}"


def recompute(stats, gate):
    corrected = int(stats["corrected_predictions"])
    broken = int(stats["broken_predictions"])
    ratio = stats["corrected_to_broken_ratio"]
    ratio_passed = (
        corrected > 0
        and broken == 0
    ) or (
        ratio is not None
        and float(ratio) >= gate["min_corrected_to_broken_ratio"]
    )
    required = max(
        int(gate["min_changed_predictions"]),
        math.ceil(
            float(gate["min_changed_fraction_of_a2_errors"])
            * int(stats["a2_errors"])
        ),
    )
    passed = (
        float(stats["correction_rate_on_a2_errors"])
        >= gate["min_correction_rate_on_a2_errors"]
        and ratio_passed
        and corrected > broken
        and int(stats["changed_predictions"]) >= required
        and float(stats["delta_paired_f1"]) > 0
    )
    return passed, required


def audit_loader(rows):
    if [row["split"] for row in rows] != ["train", "val", "test"]:
        raise ValueError("loader audit split order differs")
    for row in rows:
        if row.get("shuffle") is not True:
            raise ValueError("one loader is not shuffled")
        if (
            row.get("loader_generator") is not None
            or row.get("sampler_generator") is not None
        ):
            raise ValueError("dedicated loader generator detected")


def audit_manifest(path, expected_commit, parent_commit):
    item = json.loads(path.read_text())
    if item.get("sampling_protocol") != "dynamic_random":
        raise ValueError(f"{path}: protocol differs")
    if not str(item.get("git_commit", "")).startswith(expected_commit):
        raise ValueError(f"{path}: commit differs")
    if not str(item.get("parent_evidence_commit", "")).startswith(
        parent_commit
    ):
        raise ValueError(f"{path}: parent evidence commit differs")
    audit_loader(item["loader_audit"])
    dataset = item["dataset"]
    baseline = BASELINE[dataset]
    if (
        item["initial_a2_val_selected_test_f1"] != baseline[0]
        or item["initial_a2_raw_best_test_f1"] != baseline[1]
    ):
        raise ValueError(f"{path}: initial A2 reference differs")
    if item.get("raw_best_test_f1") is not None:
        raise ValueError(f"{path}: P0 must not report raw-best")
    calibration_passed, calibration_required = recompute(
        item["calibration_statistics"], CALIBRATION_GATE)
    validation_passed, validation_required = recompute(
        item["validation_statistics"], EVALUATION_GATE)
    test_passed, test_required = recompute(
        item["test_statistics"], EVALUATION_GATE)
    decision = (
        "pass"
        if calibration_passed and validation_passed and test_passed
        else "fail"
    )
    stored = (
        item["calibration_policy_passed"],
        item["validation_policy_passed"],
        item["test_policy_passed"],
        item["qualification_decision"],
    )
    recomputed = (
        calibration_passed,
        validation_passed,
        test_passed,
        decision,
    )
    if stored != recomputed:
        raise ValueError(f"{path}: stored gates are inconsistent")
    required = (
        int(item["calibration_required_changed_predictions"]),
        int(item["validation_required_changed_predictions"]),
        int(item["test_required_changed_predictions"]),
    )
    if required != (
        calibration_required,
        validation_required,
        test_required,
    ):
        raise ValueError(f"{path}: changed-prediction gate differs")
    if calibration_passed and not int(
        item["calibration_eligible_policies"]
    ):
        raise ValueError(f"{path}: calibration pass has no eligible policy")

    def stage(prefix, stats, passed, required_changed):
        return {
            f"{prefix}_passed": passed,
            f"{prefix}_corrected": int(stats["corrected_predictions"]),
            f"{prefix}_broken": int(stats["broken_predictions"]),
            f"{prefix}_ratio": stats["corrected_to_broken_ratio"],
            f"{prefix}_changed": int(stats["changed_predictions"]),
            f"{prefix}_required_changed": required_changed,
            f"{prefix}_paired_delta": float(stats["delta_paired_f1"]),
            f"{prefix}_add_interventions": int(
                stats.get("add_interventions", 0)),
            f"{prefix}_remove_interventions": int(
                stats.get("remove_interventions", 0)),
        }

    row = {
        "dataset": dataset,
        "family": item["evidence_family"],
        "selection": item["evidence_selection"],
        "seed": int(item["seed"]),
        "commit": item["git_commit"],
        "parent_commit": item["parent_evidence_commit"],
        "add_probe_trained": bool(item["add_probe_fit"]["trained"]),
        "remove_probe_trained": bool(
            item["remove_probe_fit"]["trained"]),
        "calibration_eligible_policies": int(
            item["calibration_eligible_policies"]),
        "qualification": decision,
        "scope": item["result_scope"],
    }
    row.update(stage(
        "calibration",
        item["calibration_statistics"],
        calibration_passed,
        calibration_required,
    ))
    row.update(stage(
        "validation",
        item["validation_statistics"],
        validation_passed,
        validation_required,
    ))
    row.update(stage(
        "test",
        item["test_statistics"],
        test_passed,
        test_required,
    ))
    return row


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    output = {}
    for family in FAMILIES:
        items = grouped[family]
        small = sum(
            row["qualification"] == "pass"
            for row in items
            if row["dataset"] == "Small-LI"
        )
        large = sum(
            row["qualification"] == "pass"
            for row in items
            if row["dataset"] == "Large-LI"
        )
        qualified = sum(
            row["qualification"] == "pass" for row in items)
        positive_both = sum(
            row["validation_paired_delta"] > 0
            and row["test_paired_delta"] > 0
            for row in items
        )
        decision = (
            "cross_scale_pass"
            if small and large
            else (
                "scale_dependent"
                if qualified
                else (
                    "optimistic_signal_insufficient"
                    if positive_both
                    else "no_utility_signal"
                )
            )
        )
        output[family] = {
            "tasks": len(items),
            "calibration_passes": sum(
                row["calibration_passed"] for row in items),
            "validation_passes": sum(
                row["validation_passed"] for row in items),
            "test_passes": sum(row["test_passed"] for row in items),
            "qualified_tasks": qualified,
            "small_li_passes": small,
            "large_li_passes": large,
            "positive_validation_and_test_delta": positive_both,
            "mean_validation_paired_delta": sum(
                row["validation_paired_delta"] for row in items
            ) / len(items),
            "mean_test_paired_delta": sum(
                row["test_paired_delta"] for row in items
            ) / len(items),
            "decision": decision,
        }
    return output


def render(payload):
    lines = [
        "# TIER Phase 2a-P0 Directional Utility Probe Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: EXECUTED manifests audited",
        "- Sampling Protocol: `dynamic_random`",
        f"- Expected Commit: `{payload['expected_commit']}`",
        f"- Parent Evidence Commit: `{payload['parent_commit']}`",
        "- Scope: optimistic in-sample-teacher feasibility probe",
        "- Fixed-panel A2: excluded",
        "",
        "## Per-Task Results",
        "",
        "| Dataset | Family | Selection | Cal eligible | Cal pass | "
        "Val corrected/broken | Val changed/required | Val delta | Val pass | "
        "Test corrected/broken | Test changed/required | Test delta | "
        "Add/remove | Test pass | Final |",
        "|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {dataset} | {family} | {selection} | {eligible} | "
            "{cal_pass} | {vc}/{vb} | {vchanged}/{vrequired} | "
            "{vdelta} | {vpass} | {tc}/{tb} | "
            "{tchanged}/{trequired} | {tdelta} | {add}/{remove} | "
            "{tpass} | {final} |".format(
                dataset=row["dataset"],
                family=row["family"],
                selection=row["selection"],
                eligible=row["calibration_eligible_policies"],
                cal_pass=str(row["calibration_passed"]).lower(),
                vc=row["validation_corrected"],
                vb=row["validation_broken"],
                vchanged=row["validation_changed"],
                vrequired=row["validation_required_changed"],
                vdelta=number(row["validation_paired_delta"]),
                vpass=str(row["validation_passed"]).lower(),
                tc=row["test_corrected"],
                tb=row["test_broken"],
                tchanged=row["test_changed"],
                trequired=row["test_required_changed"],
                tdelta=number(row["test_paired_delta"]),
                add=row["test_add_interventions"],
                remove=row["test_remove_interventions"],
                tpass=str(row["test_passed"]).lower(),
                final=row["qualification"],
            )
        )
    lines.extend([
        "",
        "## Family Summary",
        "",
        "| Family | Tasks | Cal passes | Val passes | Test passes | "
        "Qualified | Positive val+test delta | Mean val delta | "
        "Mean test delta | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for family in FAMILIES:
        item = payload["family_summaries"][family]
        lines.append(
            f"| {family} | {item['tasks']} | "
            f"{item['calibration_passes']} | "
            f"{item['validation_passes']} | {item['test_passes']} | "
            f"{item['qualified_tasks']} | "
            f"{item['positive_validation_and_test_delta']} | "
            f"{number(item['mean_validation_paired_delta'])} | "
            f"{number(item['mean_test_paired_delta'])} | "
            f"{item['decision']} |"
        )
    lines.extend([
        "",
        "## Gate Conclusion",
        "",
        payload["conclusion"],
        "",
        "This P0 uses in-sample A2 training errors and is an optimistic upper "
        "bound. It cannot support a paper result even if a task passes. "
        "Raw-best is not defined for this locked diagnostic.",
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
    paths = sorted(args.root.glob("*/phase2a_p0_manifest.json"))
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
    expected = {
        (dataset, family, selection)
        for dataset in DATASETS
        for family in FAMILIES
        for selection in SELECTIONS
    }
    if keys != expected:
        raise RuntimeError("manifest matrix is incomplete or duplicated")
    rows.sort(key=lambda row: (
        row["family"], row["dataset"], row["selection"]))
    summaries = summarize(rows)
    passing = [
        family
        for family, item in summaries.items()
        if item["decision"] == "cross_scale_pass"
    ]
    conclusion = (
        "P0 authorizes proper target-edge cross-fitting for: "
        + ", ".join(passing)
        if passing
        else (
            "No evidence family passed calibration, frozen validation, and "
            "locked test on both scales. Because even the optimistic "
            "in-sample-teacher upper bound failed, expensive cross-fitted "
            "teacher training and full CrossFusion are not authorized."
        )
    )
    payload = {
        "sampling_protocol": "dynamic_random",
        "expected_commit": args.expected_commit,
        "parent_commit": args.parent_commit,
        "scope": "optimistic_in_sample_teacher_feasibility_probe",
        "rows": rows,
        "family_summaries": summaries,
        "conclusion": conclusion,
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(payload))
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
