#!/usr/bin/env python3
"""Audit TIER Phase 1b family decomposition manifests."""

import argparse
import json
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "run" / "dynamic_random_a2_baseline.json"
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")
DATASETS = ("Small-LI", "Large-LI")
EXPECTED_SEEDS = {"Small-LI": 42, "Large-LI": 44}
GATE = {
    "normal_minus_shuffled": 0.010,
    "normal_minus_off": 0.005,
    "correction_rate": 0.10,
    "corrected_to_broken": 1.5,
    "coverage": 0.80,
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def format_number(value, digits=5):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def recompute_qualification(record):
    selected = record["selected_event"]["test"]
    sampled = record["same_batch_a2_diagnostic"]["sampled_instances"]
    checks = record["qualification_checks"]
    corrected = int(sampled["corrected_predictions"])
    broken = int(sampled["broken_predictions"])
    required_changed = max(50, 0.10 * int(sampled["a2_errors"]))
    ratio = sampled["corrected_to_broken_ratio"]
    ratio_pass = (
        corrected > 0
        if broken == 0
        else ratio is not None and float(ratio) >= GATE["corrected_to_broken"]
    )
    passed = (
        float(selected["normal"]["f1"])
        - float(selected["shuffled"]["f1"])
        >= GATE["normal_minus_shuffled"]
        and float(selected["normal"]["f1"])
        - float(selected["off"]["f1"])
        >= GATE["normal_minus_off"]
        and float(sampled["correction_rate_on_a2_errors"])
        >= GATE["correction_rate"]
        and ratio_pass
        and corrected > broken
        and int(sampled["changed_predictions"]) >= required_changed
        and float(checks["overall_coverage"]) >= GATE["coverage"]
        and all(
            float(value) >= GATE["coverage"]
            for value in checks["class_coverage"].values()
        )
    )
    return "pass" if passed else "fail"


def audit_manifest(path, baseline, expected_commit):
    record = load_json(path)
    dataset = record.get("dataset")
    family = record.get("evidence_family")
    selection = record.get("evidence_selection")
    require(
        record.get("sampling_protocol") == "dynamic_random",
        f"{path}: sampling_protocol must be dynamic_random",
    )
    require(dataset in DATASETS, f"{path}: unexpected dataset {dataset!r}")
    require(family in FAMILIES, f"{path}: unexpected family {family!r}")
    require(
        selection in SELECTIONS,
        f"{path}: unexpected selection {selection!r}",
    )
    require(
        record.get("model") == "TIER-EvidenceOnly-FamilyDecomposition",
        f"{path}: unexpected model",
    )
    require(
        record.get("variant") == f"{family}_{selection}",
        f"{path}: variant does not match family and selection",
    )
    require(
        int(record.get("seed")) == EXPECTED_SEEDS[dataset],
        f"{path}: unexpected seed",
    )
    if expected_commit:
        require(
            record.get("git_commit", "").startswith(expected_commit),
            f"{path}: git commit differs from {expected_commit}",
        )

    reference = baseline["datasets"][dataset]
    for metric in ("val_selected_test_f1", "raw_best_test_f1"):
        baseline_key = f"initial_a2_{metric}"
        delta_key = (
            "delta_val_selected_f1"
            if metric == "val_selected_test_f1"
            else "delta_raw_best_f1"
        )
        require(metric in record, f"{path}: missing {metric}")
        require(
            abs(float(record[baseline_key]) - float(reference[metric])) < 1e-9,
            f"{path}: {baseline_key} differs from initial A2",
        )
        expected_delta = float(record[metric]) - float(reference[metric])
        require(
            abs(float(record[delta_key]) - expected_delta) < 1e-9,
            f"{path}: {delta_key} crosses or miscomputes metrics",
        )

    selected = record.get("selected_event", {}).get("test", {})
    require(
        {"normal", "shuffled", "off"}.issubset(selected),
        f"{path}: selected event lacks normal/shuffled/off",
    )
    diagnostic = record.get("same_batch_a2_diagnostic", {})
    require(
        {"sampled_instances", "unique_edges"}.issubset(diagnostic),
        f"{path}: paired A2 diagnostics are incomplete",
    )
    recomputed = recompute_qualification(record)
    require(
        record.get("qualification_decision") == recomputed,
        f"{path}: stored qualification decision is inconsistent",
    )
    return record


def row_from_record(record):
    selected = record["selected_event"]["test"]
    sampled = record["same_batch_a2_diagnostic"]["sampled_instances"]
    normal = float(selected["normal"]["f1"])
    corrected = int(sampled["corrected_predictions"])
    broken = int(sampled["broken_predictions"])
    return {
        "dataset": record["dataset"],
        "family": record["evidence_family"],
        "selection": record["evidence_selection"],
        "seed": int(record["seed"]),
        "commit": record["git_commit"],
        "selected": float(record["val_selected_test_f1"]),
        "selected_delta": float(record["delta_val_selected_f1"]),
        "raw": float(record["raw_best_test_f1"]),
        "raw_delta": float(record["delta_raw_best_f1"]),
        "normal_minus_shuffled": (
            normal - float(selected["shuffled"]["f1"])
        ),
        "normal_minus_off": normal - float(selected["off"]["f1"]),
        "corrected": corrected,
        "broken": broken,
        "corrected_minus_broken": corrected - broken,
        "corrected_to_broken": sampled["corrected_to_broken_ratio"],
        "correction_rate": float(
            sampled["correction_rate_on_a2_errors"]
        ),
        "changed": int(sampled["changed_predictions"]),
        "qualification": record["qualification_decision"],
    }


def summarize(rows):
    summaries = {}
    for family in FAMILIES:
        items = [row for row in rows if row["family"] == family]
        qualified_datasets = {
            row["dataset"]
            for row in items
            if row["qualification"] == "pass"
        }
        active = [
            row
            for row in items
            if (
                row["normal_minus_shuffled"]
                >= GATE["normal_minus_shuffled"]
                and row["normal_minus_off"] >= GATE["normal_minus_off"]
            )
        ]
        if set(DATASETS).issubset(qualified_datasets):
            decision = "cross_scale_pass"
        elif qualified_datasets:
            decision = "scale_specific_only"
        elif active:
            decision = "informative_but_unsafe"
        else:
            decision = "no_context_signal"
        summaries[family] = {
            "tasks": len(items),
            "counterfactually_active_tasks": len(active),
            "qualified_tasks": sum(
                row["qualification"] == "pass" for row in items
            ),
            "small_li_passes": sum(
                row["dataset"] == "Small-LI"
                and row["qualification"] == "pass"
                for row in items
            ),
            "large_li_passes": sum(
                row["dataset"] == "Large-LI"
                and row["qualification"] == "pass"
                for row in items
            ),
            "mean_val_selected_delta": sum(
                row["selected_delta"] for row in items
            ) / len(items),
            "mean_raw_delta": sum(
                row["raw_delta"] for row in items
            ) / len(items),
            "best_corrected_to_broken": max(
                (
                    float(row["corrected_to_broken"])
                    for row in items
                    if row["corrected_to_broken"] is not None
                ),
                default=None,
            ),
            "decision": decision,
        }
    return summaries


def markdown(rows, summaries, baseline_source, expected_commit):
    lines = [
        "# TIER Phase 1b Evidence Family Decomposition Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: EXECUTED manifests audited",
        "- Sampling Protocol: `dynamic_random`",
        f"- Historical Baseline: `{baseline_source}`",
        f"- Expected Commit: `{expected_commit or 'not constrained'}`",
        "- Fixed-panel A2: excluded",
        "",
        "## Per-Task Results",
        "",
        "| Dataset | Family | Selection | Seed | Val-selected Test F1 | Delta vs A2 | Raw-best Test F1 | Delta vs A2 | Normal-shuffled | Normal-off | Corrected | Broken | Ratio | Correction rate | Changed | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["family"],
            item["dataset"],
            item["selection"],
        ),
    ):
        lines.append(
            "| {dataset} | {family} | {selection} | {seed} | {selected} | "
            "{selected_delta} | {raw} | {raw_delta} | {shuffle} | {off} | "
            "{corrected} | {broken} | {ratio} | {correction} | {changed} | "
            "{qualification} |".format(
                dataset=row["dataset"],
                family=row["family"],
                selection=row["selection"],
                seed=row["seed"],
                selected=format_number(row["selected"]),
                selected_delta=format_number(row["selected_delta"]),
                raw=format_number(row["raw"]),
                raw_delta=format_number(row["raw_delta"]),
                shuffle=format_number(row["normal_minus_shuffled"]),
                off=format_number(row["normal_minus_off"]),
                corrected=row["corrected"],
                broken=row["broken"],
                ratio=format_number(row["corrected_to_broken"]),
                correction=format_number(row["correction_rate"]),
                changed=row["changed"],
                qualification=row["qualification"],
            )
        )
    lines.extend([
        "",
        "## Family Summary",
        "",
        "| Family | Tasks | Counterfactually active | Qualified | Small passes | Large passes | Mean Val-selected Delta | Mean Raw-best Delta | Best corrected/broken | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for family in FAMILIES:
        item = summaries[family]
        lines.append(
            f"| {family} | {item['tasks']} | "
            f"{item['counterfactually_active_tasks']} | "
            f"{item['qualified_tasks']} | {item['small_li_passes']} | "
            f"{item['large_li_passes']} | "
            f"{format_number(item['mean_val_selected_delta'])} | "
            f"{format_number(item['mean_raw_delta'])} | "
            f"{format_number(item['best_corrected_to_broken'])} | "
            f"{item['decision']} |"
        )
    cross_scale = [
        family
        for family, item in summaries.items()
        if item["decision"] == "cross_scale_pass"
    ]
    lines.extend([
        "",
        "## Gate Conclusion",
        "",
        (
            "Cross-scale qualified families: "
            + ", ".join(f"`{family}`" for family in cross_scale)
            if cross_scale
            else (
                "No evidence family passed the preregistered qualification "
                "gate on both Small-LI and Large-LI. Phase 2 is not "
                "authorized by these results."
            )
        ),
        "",
        "Qualification is recomputed from each manifest. It is not inferred",
        "from headline F1, Raw-best F1, or the stored gate label alone.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=12)
    parser.add_argument("--expected-commit")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(BASELINE_PATH)
    require(
        baseline.get("sampling_protocol") == "dynamic_random",
        "canonical A2 baseline is not dynamic_random",
    )
    manifests = sorted(args.root.glob("*/experiment_manifest.json"))
    require(
        len(manifests) == args.expected_tasks,
        f"expected {args.expected_tasks} manifests, found {len(manifests)}",
    )
    records = [
        audit_manifest(path, baseline, args.expected_commit)
        for path in manifests
    ]
    keys = {
        (
            record["dataset"],
            record["evidence_family"],
            record["evidence_selection"],
        )
        for record in records
    }
    expected_keys = set(product(DATASETS, FAMILIES, SELECTIONS))
    require(keys == expected_keys, "manifest matrix is incomplete or duplicated")

    rows = [row_from_record(record) for record in records]
    summaries = summarize(rows)
    payload = {
        "sampling_protocol": "dynamic_random",
        "baseline_source": baseline["source"],
        "expected_commit": args.expected_commit,
        "rows": rows,
        "family_summaries": summaries,
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        markdown(
            rows,
            summaries,
            baseline["source"],
            args.expected_commit,
        )
    )
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
