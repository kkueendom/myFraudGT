#!/usr/bin/env python3
"""Audit CET evidence reliability under the dynamic-random sampling protocol."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

import fraudGT  # noqa: F401 - register GraphGym components
from fraudGT.evidence.cet_model import CETFusionClassifier
from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.cet_phaseb_screen import (
    CONDITIONS,
    audit_cet_protocol,
    evaluate,
    intervention_statistics,
    task_spec,
)
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    TASK,
    best_f1_threshold,
    binary_f1,
    configure_fraudgt,
    loader_audit,
    load_a2_model,
    metric_row,
    seed_process,
    verify_repository,
)


SUPPORT_BINS = (
    ("zero", 0, 0),
    ("low_1_7", 1, 7),
    ("medium_8_23", 8, 23),
    ("high_24_47", 24, 47),
    ("max_48", 48, None),
)


def scalar_distribution(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def masked_mean(values, mask):
    if not mask.any():
        return None
    return float(values[mask].float().mean())


def support_conditioned_audit(test, model_threshold, base_threshold):
    labels = test["labels"].bool()
    support = test["support"][:, 0].long()
    normal_scores = test["normal"].float()
    shuffled_scores = test["shuffled"].float()
    off_scores = test["off"].float()
    base_scores = test["base"].float()
    model_predictions = normal_scores >= float(model_threshold)
    base_predictions = base_scores >= float(base_threshold)
    changed = model_predictions != base_predictions
    corrected = changed & (model_predictions == labels) & (
        base_predictions != labels)
    broken = changed & (model_predictions != labels) & (
        base_predictions == labels)
    rows = []
    for name, lower, upper in SUPPORT_BINS:
        mask = support >= lower
        if upper is not None:
            mask &= support <= upper
        count = int(mask.sum())
        if not count:
            rows.append({
                "bin": name,
                "samples": 0,
                "positives": 0,
            })
            continue
        rows.append({
            "bin": name,
            "samples": count,
            "positives": int(labels[mask].sum()),
            "base_f1": binary_f1(
                labels[mask], base_predictions[mask]),
            "normal_f1": binary_f1(
                labels[mask], model_predictions[mask]),
            "changed": int(changed[mask].sum()),
            "corrected": int(corrected[mask].sum()),
            "broken": int(broken[mask].sum()),
            "mean_abs_normal_minus_shuffled": masked_mean(
                (normal_scores - shuffled_scores).abs(), mask),
            "mean_abs_normal_minus_off": masked_mean(
                (normal_scores - off_scores).abs(), mask),
        })
    return rows


def repeat_event(model, a2_model, loaders, index, spec, task, device,
                 repeat_index, eval_step_cap):
    val = evaluate(
        model,
        a2_model,
        loaders[1],
        index,
        spec,
        task,
        device,
        "val",
        eval_step_cap,
        ("normal",),
    )
    model_threshold, val_f1 = best_f1_threshold(
        val["labels"], val["normal"])
    base_threshold, base_val_f1 = best_f1_threshold(
        val["labels"], val["base"])
    test = evaluate(
        model,
        a2_model,
        loaders[2],
        index,
        spec,
        task,
        device,
        "test",
        eval_step_cap,
        CONDITIONS,
    )
    metrics = {
        condition: metric_row(
            test["labels"], test[condition], model_threshold)
        for condition in CONDITIONS
    }
    base_metric = metric_row(
        test["labels"], test["base"], base_threshold)
    intervention = intervention_statistics(
        test["labels"],
        test["base"],
        test["normal"],
        base_threshold,
        model_threshold,
    )
    unique_edges = int(torch.unique(test["edge_ids"]).numel())
    baseline = INITIAL_A2[task["dataset"]]
    return {
        "repeat": repeat_index,
        "sampling_protocol": "dynamic_random",
        "val": metric_row(
            val["labels"], val["normal"], model_threshold),
        "base_val": metric_row(
            val["labels"], val["base"], base_threshold),
        "val_f1": val_f1,
        "base_val_f1": base_val_f1,
        "model_threshold": model_threshold,
        "base_threshold": base_threshold,
        "test": metrics,
        "base_test": base_metric,
        "delta_vs_initial_a2": (
            metrics["normal"]["f1"]
            - baseline["val_selected_test_f1"]
        ),
        "same_batch_delta_vs_frozen_a2": (
            metrics["normal"]["f1"] - base_metric["f1"]
        ),
        "normal_minus_shuffled_f1": (
            metrics["normal"]["f1"] - metrics["shuffled"]["f1"]
        ),
        "normal_minus_off_f1": (
            metrics["normal"]["f1"] - metrics["off"]["f1"]
        ),
        "intervention": intervention,
        "support_conditioned": support_conditioned_audit(
            test, model_threshold, base_threshold),
        "diagnostics": test["diagnostics"],
        "validation_loader_iterations": val["steps"],
        "test_loader_iterations": test["steps"],
        "test_samples": int(test["labels"].numel()),
        "test_unique_edges": unique_edges,
        "test_unique_edge_rate": (
            unique_edges / max(int(test["labels"].numel()), 1)
        ),
        "val_query_seconds": val["query_seconds"],
        "test_query_seconds": test["query_seconds"],
        "val_elapsed_seconds": val["elapsed_seconds"],
        "test_elapsed_seconds": test["elapsed_seconds"],
    }


def summarize_events(events):
    fields = {
        "normal_test_f1": [
            row["test"]["normal"]["f1"] for row in events],
        "shuffled_test_f1": [
            row["test"]["shuffled"]["f1"] for row in events],
        "off_test_f1": [
            row["test"]["off"]["f1"] for row in events],
        "same_batch_base_test_f1": [
            row["base_test"]["f1"] for row in events],
        "delta_vs_initial_a2": [
            row["delta_vs_initial_a2"] for row in events],
        "same_batch_delta_vs_frozen_a2": [
            row["same_batch_delta_vs_frozen_a2"] for row in events],
        "normal_minus_shuffled_f1": [
            row["normal_minus_shuffled_f1"] for row in events],
        "normal_minus_off_f1": [
            row["normal_minus_off_f1"] for row in events],
        "changed_predictions": [
            row["intervention"]["changed_predictions"] for row in events],
        "corrected_predictions": [
            row["intervention"]["corrected_predictions"] for row in events],
        "broken_predictions": [
            row["intervention"]["broken_predictions"] for row in events],
        "corrected_minus_broken": [
            row["intervention"]["corrected_minus_broken"] for row in events],
        "test_unique_edge_rate": [
            row["test_unique_edge_rate"] for row in events],
    }
    return {
        key: scalar_distribution(values)
        for key, values in fields.items()
    }


def write_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_audit(spec, task, args):
    commit = verify_repository(spec)
    resolved_spec = task_spec(spec, task)
    config_relative = resolved_spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_cet_protocol(resolved_spec, task, config)
    seed_process(int(task["audit_seed"]))
    configure_fraudgt(config_path, task["audit_seed"], args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    device = torch.device(args.device)
    full_store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=full_store.edge_index,
        timestamps=full_store.timestamps,
        raw_edge_attr=full_store.raw_edge_attr,
    )
    a2_model = load_a2_model(
        dataset, task["a2_checkpoint"], device).to(device)
    for parameter in a2_model.parameters():
        parameter.requires_grad_(False)
    model = CETFusionClassifier(
        base_feature_dim=int(cfg.gt.dim_hidden) * 3,
        num_currencies=int(full_store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(
            full_store.raw_edge_attr[:, 3].max()) + 1,
        variant=task["variant"],
        hidden_dim=int(resolved_spec["hidden_dim"]),
        num_heads=int(resolved_spec["num_heads"]),
        dropout=float(resolved_spec["dropout"]),
    ).to(device)
    checkpoint = torch.load(
        task["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_dir = args.output_dir / (
        f"{task['dataset']}_{task['experiment_label']}_"
        f"auditseed{task['audit_seed']}_{commit}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "reliability_trajectory.jsonl"
    events = []
    started = time.monotonic()
    repeats = (
        int(args.repeats)
        if args.repeats is not None
        else int(task["repeats"])
    )
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if args.eval_step_cap is not None and args.eval_step_cap < 1:
        raise ValueError("eval step cap must be positive")
    for repeat_index in range(repeats):
        event = repeat_event(
            model,
            a2_model,
            loaders,
            index,
            resolved_spec,
            task,
            device,
            repeat_index,
            args.eval_step_cap,
        )
        write_jsonl(trajectory, event)
        events.append(event)
    manifest = {
        "dataset": task["dataset"],
        "model": "CET-FraudGT-v1-reliability-audit",
        "variant": task["variant"],
        "experiment_label": task["experiment_label"],
        "model_seed": int(task["model_seed"]),
        "audit_seed": int(task["audit_seed"]),
        "git_commit": commit,
        "model_checkpoint_commit": checkpoint.get("git_commit"),
        "config": config_relative,
        "experiment_spec": str(args.spec),
        "checkpoint": task["checkpoint"],
        "a2_checkpoint": task["a2_checkpoint"],
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "fixed_target_panel": False,
        "dedicated_evaluation_generator": False,
        "sampler_rng_restoration": False,
        "registered_repeats": int(task["repeats"]),
        "repeats": repeats,
        "eval_step_cap": args.eval_step_cap,
        "validation_loader_iterations": sum(
            row["validation_loader_iterations"] for row in events),
        "test_loader_iterations": sum(
            row["test_loader_iterations"] for row in events),
        "events": events,
        "summary": summarize_events(events),
        "raw_best_repeated_test_f1": max(
            row["test"]["normal"]["f1"] for row in events),
        "sampling_variation_note": (
            "descriptive_repeated_dynamic_sampling_audit"
        ),
        "route_diagnostics": "not_applicable_no_router",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = output_dir / "reliability_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": task["dataset"],
        "experiment_label": task["experiment_label"],
        "audit_seed": task["audit_seed"],
        "normal_test_f1": manifest["summary"]["normal_test_f1"],
        "normal_minus_shuffled_f1": (
            manifest["summary"]["normal_minus_shuffled_f1"]),
        "corrected_minus_broken": (
            manifest["summary"]["corrected_minus_broken"]),
        "output": str(output_dir),
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--eval-step-cap", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    task = spec["tasks"][args.task_index]
    run_audit(spec, task, args)


if __name__ == "__main__":
    main()
