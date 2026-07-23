#!/usr/bin/env python3
"""Audit TIER Phase 1 manifests against the canonical dynamic A2 table."""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "run" / "dynamic_random_a2_baseline.json"


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def format_number(value, digits=5):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_manifest(manifest_path, baseline):
    record = load_json(manifest_path)
    dataset = record.get("dataset")
    require(
        record.get("sampling_protocol") == "dynamic_random",
        f"{manifest_path}: sampling_protocol must be dynamic_random",
    )
    require(
        dataset in baseline["datasets"],
        f"{manifest_path}: unknown dataset {dataset!r}",
    )
    reference = baseline["datasets"][dataset]
    for metric in ("val_selected_test_f1", "raw_best_test_f1"):
        require(metric in record, f"{manifest_path}: missing {metric}")
        baseline_key = f"initial_a2_{metric}"
        delta_key = (
            "delta_val_selected_f1"
            if metric == "val_selected_test_f1"
            else "delta_raw_best_f1"
        )
        require(
            abs(float(record[baseline_key]) - float(reference[metric])) < 1e-9,
            f"{manifest_path}: {baseline_key} differs from initial A2",
        )
        expected_delta = float(record[metric]) - float(reference[metric])
        require(
            abs(float(record[delta_key]) - expected_delta) < 1e-9,
            f"{manifest_path}: {delta_key} crosses or miscomputes metrics",
        )
    selected = record.get("selected_event", {})
    selected_test = selected.get("test", {})
    require(
        {"normal", "shuffled", "off"}.issubset(selected_test),
        f"{manifest_path}: selected event lacks normal/shuffled/off results",
    )
    diagnostic = record.get("same_batch_a2_diagnostic")
    require(
        diagnostic and "sampled_instances" in diagnostic,
        f"{manifest_path}: missing same-batch A2 diagnostic",
    )
    return record


def row_from_record(record):
    selected = record["selected_event"]["test"]
    paired = record["same_batch_a2_diagnostic"]["sampled_instances"]
    return {
        "dataset": record["dataset"],
        "selection": record["evidence_selection"],
        "seed": int(record["seed"]),
        "commit": record["git_commit"],
        "selected": float(record["val_selected_test_f1"]),
        "selected_delta": float(record["delta_val_selected_f1"]),
        "raw": float(record["raw_best_test_f1"]),
        "raw_delta": float(record["delta_raw_best_f1"]),
        "normal_minus_shuffled": (
            float(selected["normal"]["f1"])
            - float(selected["shuffled"]["f1"])
        ),
        "normal_minus_off": (
            float(selected["normal"]["f1"]) - float(selected["off"]["f1"])
        ),
        "correction_rate": float(paired["correction_rate_on_a2_errors"]),
        "corrected_to_broken": paired["corrected_to_broken_ratio"],
        "changed": int(paired["changed_predictions"]),
        "qualification": record["qualification_decision"],
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["selection"], []).append(row)
    output = {}
    for selection, items in sorted(groups.items()):
        output[selection] = {
            "tasks": len(items),
            "mean_val_selected_delta": sum(
                item["selected_delta"] for item in items
            ) / len(items),
            "mean_raw_delta": sum(
                item["raw_delta"] for item in items
            ) / len(items),
            "qualified_tasks": sum(
                item["qualification"] == "pass" for item in items
            ),
            "failed_tasks": sum(
                item["qualification"] != "pass" for item in items
            ),
        }
    return output


def markdown(rows, aggregates, baseline_source):
    lines = [
        "# TIER Phase 1 Evidence Qualification Results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: EXECUTED manifests audited",
        "- Sampling Protocol: `dynamic_random`",
        f"- Historical Baseline: `{baseline_source}`",
        "- Fixed-panel A2: excluded",
        "",
        "## Per-Task Results",
        "",
        "| Dataset | Selection | Seed | Val-selected Test F1 | Delta vs initial A2 | Raw-best Test F1 | Delta vs initial A2 | Normal-shuffled | Normal-off | Corrected/broken | Correction rate | Changed | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], item["selection"])):
        lines.append(
            "| {dataset} | {selection} | {seed} | {selected} | {selected_delta} | "
            "{raw} | {raw_delta} | {shuffle} | {off} | {ratio} | {correction} | "
            "{changed} | {qualification} |".format(
                dataset=row["dataset"],
                selection=row["selection"],
                seed=row["seed"],
                selected=format_number(row["selected"]),
                selected_delta=format_number(row["selected_delta"]),
                raw=format_number(row["raw"]),
                raw_delta=format_number(row["raw_delta"]),
                shuffle=format_number(row["normal_minus_shuffled"]),
                off=format_number(row["normal_minus_off"]),
                ratio=format_number(row["corrected_to_broken"]),
                correction=format_number(row["correction_rate"]),
                changed=row["changed"],
                qualification=row["qualification"],
            )
        )
    lines.extend([
        "",
        "## Selection Summary",
        "",
        "| Selection | Tasks | Mean Val-selected Delta | Mean Raw-best Delta | Qualified | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for selection, item in aggregates.items():
        lines.append(
            f"| {selection} | {item['tasks']} | "
            f"{format_number(item['mean_val_selected_delta'])} | "
            f"{format_number(item['mean_raw_delta'])} | "
            f"{item['qualified_tasks']} | {item['failed_tasks']} |"
        )
    lines.extend([
        "",
        "A result with an absolute historical-A2 delta below `0.005` is within",
        "the configured dynamic-sampling caution range. Qualification uses the",
        "counterfactual evidence and paired A2 diagnostics in each manifest;",
        "it is not inferred from headline F1 alone.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(BASELINE_PATH)
    require(
        baseline.get("sampling_protocol") == "dynamic_random",
        "canonical A2 baseline is not dynamic_random",
    )
    manifests = sorted(args.root.glob("*/experiment_manifest.json"))
    if args.expected_tasks is not None:
        require(
            len(manifests) == args.expected_tasks,
            f"expected {args.expected_tasks} manifests, found {len(manifests)}",
        )
    require(manifests, "no Phase 1 manifests found")
    records = [audit_manifest(path, baseline) for path in manifests]
    rows = [row_from_record(record) for record in records]
    aggregates = summarize(rows)
    payload = {
        "sampling_protocol": "dynamic_random",
        "baseline_source": baseline["source"],
        "rows": rows,
        "aggregates": aggregates,
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(rows, aggregates, baseline["source"]))
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
