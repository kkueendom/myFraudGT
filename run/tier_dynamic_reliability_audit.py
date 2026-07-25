#!/usr/bin/env python3
"""Repeated same-batch reliability audit for fixed TIER checkpoints."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import torch
import yaml

import fraudGT  # noqa: F401 - register GraphGym components
from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.evidence.tier_model import TransactionEvidenceEncoder
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    TASK,
    audit_protocol,
    best_f1_threshold,
    configure_fraudgt,
    correction_counts,
    loader_audit,
    load_a2_model,
    metric_row,
    query_batch,
    seed_process,
    select_values_by_edge_id,
    verify_repository,
)


CONDITIONS = ("normal", "shuffled", "off")


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


def audit_reliability_protocol(spec, task, config):
    audit_protocol(spec, task, config)
    source = Path(__file__).read_text()
    forbidden = (
        "torch." + "Generator(",
        "get_" + "rng_state(",
        "set_" + "rng_state(",
        "fixed_target_panel" + "=True",
    )
    if any(token in source for token in forbidden):
        raise RuntimeError(
            "fixed evaluation RNG or target panel detected in audit")


@torch.no_grad()
def evaluate_paired(
    evidence_model,
    a2_model,
    loader,
    index,
    spec,
    selection,
    device,
    split,
    conditions,
    step_cap,
):
    evidence_model.eval()
    a2_model.eval()
    edge_ids_all = []
    labels_all = []
    condition_scores = {condition: [] for condition in conditions}
    base_scores_all = []
    support_all = []
    token_counts_all = []
    requested_samples = 0
    query_seconds = 0.0
    started = time.monotonic()
    steps = 0
    for batch in loader:
        (
            target_ids,
            labels,
            evidence,
            target_raw,
            query_time,
            evidence_cpu,
        ) = query_batch(index, batch, spec, selection, device)
        raw_scores = {}
        if "normal" in conditions:
            logits, _ = evidence_model(evidence, target_raw)
            raw_scores["normal"] = torch.sigmoid(
                logits).detach().cpu()
        if "shuffled" in conditions:
            permutation = torch.randperm(
                evidence.tokens.size(0), device=device)
            logits, _ = evidence_model(
                evidence.shuffled(permutation), target_raw)
            raw_scores["shuffled"] = torch.sigmoid(
                logits).detach().cpu()
        if "off" in conditions:
            logits, _ = evidence_model(evidence.off(), target_raw)
            raw_scores["off"] = torch.sigmoid(
                logits).detach().cpu()

        batch.split = split
        batch.to(device)
        a2_logits, a2_labels = a2_model(batch)
        a2_logits = a2_logits.squeeze(-1).detach().cpu()
        a2_labels = a2_labels.detach().cpu().long().view(-1)
        target_mask = torch.isin(
            batch[TASK].e_id, target_ids.to(device))
        paired_ids = batch[TASK].e_id[
            target_mask].detach().cpu()
        if not (
            paired_ids.numel()
            == a2_logits.numel()
            == a2_labels.numel()
        ):
            raise AssertionError(
                "A2 outputs are not aligned to target edge IDs")

        token_counts = evidence_cpu.mask.sum(dim=1)
        selected = select_values_by_edge_id(
            paired_ids,
            target_ids,
            labels,
            evidence_cpu.support,
            token_counts,
            *[raw_scores[condition] for condition in conditions],
        )
        paired_labels, support, token_counts, *paired_conditions = (
            selected)
        if not torch.equal(paired_labels.long(), a2_labels):
            raise AssertionError(
                "A2 and evidence labels are not aligned")
        edge_ids_all.append(paired_ids)
        labels_all.append(a2_labels)
        base_scores_all.append(torch.sigmoid(a2_logits))
        support_all.append(support)
        token_counts_all.append(token_counts)
        for condition, scores in zip(conditions, paired_conditions):
            condition_scores[condition].append(scores)
        requested_samples += int(target_ids.numel())
        query_seconds += query_time
        steps += 1
        if step_cap is not None and steps >= int(step_cap):
            break
    edge_ids = torch.cat(edge_ids_all)
    labels = torch.cat(labels_all)
    base_scores = torch.cat(base_scores_all)
    support = torch.cat(support_all)
    token_counts = torch.cat(token_counts_all)
    unique_edges = int(torch.unique(edge_ids).numel())
    return {
        "edge_ids": edge_ids,
        "labels": labels,
        "base": base_scores,
        "conditions": {
            condition: torch.cat(parts)
            for condition, parts in condition_scores.items()
        },
        "support": support,
        "token_counts": token_counts,
        "coverage_rate": float((token_counts > 0).float().mean()),
        "support_mean": support.float().mean(dim=0).tolist(),
        "requested_samples": requested_samples,
        "paired_samples": int(labels.numel()),
        "paired_retention_rate": (
            int(labels.numel()) / max(requested_samples, 1)),
        "steps": steps,
        "unique_edges": unique_edges,
        "unique_edge_rate": unique_edges / max(int(labels.numel()), 1),
        "query_seconds": query_seconds,
        "elapsed_seconds": time.monotonic() - started,
    }


def repeat_event(
    evidence_model,
    a2_model,
    loaders,
    index,
    spec,
    task,
    device,
    repeat_index,
    eval_step_cap,
):
    val = evaluate_paired(
        evidence_model,
        a2_model,
        loaders[1],
        index,
        spec,
        task["selection"],
        device,
        "val",
        ("normal",),
        eval_step_cap,
    )
    evidence_threshold, _ = best_f1_threshold(
        val["labels"], val["conditions"]["normal"])
    base_threshold, _ = best_f1_threshold(
        val["labels"], val["base"])
    test = evaluate_paired(
        evidence_model,
        a2_model,
        loaders[2],
        index,
        spec,
        task["selection"],
        device,
        "test",
        CONDITIONS,
        eval_step_cap,
    )
    metrics = {
        condition: metric_row(
            test["labels"],
            test["conditions"][condition],
            evidence_threshold,
        )
        for condition in CONDITIONS
    }
    base_metric = metric_row(
        test["labels"], test["base"], base_threshold)
    interventions = {
        condition: correction_counts(
            test["labels"],
            test["conditions"][condition],
            test["base"],
            evidence_threshold,
            base_threshold,
        )
        for condition in CONDITIONS
    }
    historical = INITIAL_A2[task["dataset"]][
        "val_selected_test_f1"]
    return {
        "repeat": repeat_index,
        "sampling_protocol": "dynamic_random",
        "evidence_threshold": evidence_threshold,
        "base_threshold": base_threshold,
        "val": metric_row(
            val["labels"],
            val["conditions"]["normal"],
            evidence_threshold,
        ),
        "base_val": metric_row(
            val["labels"], val["base"], base_threshold),
        "test": metrics,
        "base_test": base_metric,
        "delta_vs_historical_initial_a2": (
            metrics["normal"]["f1"] - historical),
        "same_batch_delta_vs_frozen_a2": (
            metrics["normal"]["f1"] - base_metric["f1"]),
        "normal_minus_shuffled_f1": (
            metrics["normal"]["f1"]
            - metrics["shuffled"]["f1"]),
        "normal_minus_off_f1": (
            metrics["normal"]["f1"] - metrics["off"]["f1"]),
        "mean_abs_normal_minus_shuffled_score": float(
            (
                test["conditions"]["normal"]
                - test["conditions"]["shuffled"]
            ).abs().mean()
        ),
        "mean_abs_normal_minus_off_score": float(
            (
                test["conditions"]["normal"]
                - test["conditions"]["off"]
            ).abs().mean()
        ),
        "interventions": interventions,
        "coverage_rate": test["coverage_rate"],
        "support_mean": test["support_mean"],
        "test_samples": test["paired_samples"],
        "test_unique_edges": test["unique_edges"],
        "test_unique_edge_rate": test["unique_edge_rate"],
        "val_paired_retention_rate": val["paired_retention_rate"],
        "test_paired_retention_rate": test["paired_retention_rate"],
        "validation_loader_iterations": val["steps"],
        "test_loader_iterations": test["steps"],
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
        "same_batch_delta_vs_frozen_a2": [
            row["same_batch_delta_vs_frozen_a2"] for row in events],
        "normal_minus_shuffled_f1": [
            row["normal_minus_shuffled_f1"] for row in events],
        "normal_minus_off_f1": [
            row["normal_minus_off_f1"] for row in events],
        "mean_abs_normal_minus_shuffled_score": [
            row["mean_abs_normal_minus_shuffled_score"]
            for row in events
        ],
        "mean_abs_normal_minus_off_score": [
            row["mean_abs_normal_minus_off_score"]
            for row in events
        ],
        "normal_changed": [
            row["interventions"]["normal"]["changed_predictions"]
            for row in events
        ],
        "normal_corrected": [
            row["interventions"]["normal"]["corrected_predictions"]
            for row in events
        ],
        "normal_broken": [
            row["interventions"]["normal"]["broken_predictions"]
            for row in events
        ],
        "normal_corrected_minus_broken": [
            (
                row["interventions"]["normal"]["corrected_predictions"]
                - row["interventions"]["normal"]["broken_predictions"]
            )
            for row in events
        ],
        "coverage_rate": [
            row["coverage_rate"] for row in events],
        "test_unique_edge_rate": [
            row["test_unique_edge_rate"] for row in events],
        "evidence_threshold": [
            row["evidence_threshold"] for row in events],
        "base_threshold": [
            row["base_threshold"] for row in events],
    }
    return {
        name: scalar_distribution(values)
        for name, values in fields.items()
    }


def append_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_audit(spec, task, args):
    commit = verify_repository(spec)
    relative_config = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / relative_config
    config = yaml.safe_load(config_path.read_text())
    audit_reliability_protocol(spec, task, config)
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
    evidence_model = TransactionEvidenceEncoder(
        num_currencies=int(full_store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(
            full_store.raw_edge_attr[:, 3].max()) + 1,
        family=spec["family"],
        hidden_dim=int(spec["hidden_dim"]),
        num_heads=int(spec["num_heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    checkpoint = torch.load(
        task["checkpoint"], map_location=device)
    evidence_model.load_state_dict(
        checkpoint["model_state"], strict=True)
    evidence_model.eval()
    a2_model = load_a2_model(
        dataset, task["a2_checkpoint"], device).to(device)
    for model in (evidence_model, a2_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    output_dir = args.output_dir / (
        f"{task['dataset']}_{task['selection']}_"
        f"{task['experiment_label']}_auditseed"
        f"{task['audit_seed']}_{commit}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "reliability_trajectory.jsonl"
    repeats = (
        int(args.repeats)
        if args.repeats is not None
        else int(task["repeats"])
    )
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if args.eval_step_cap is not None and args.eval_step_cap < 1:
        raise ValueError("eval step cap must be positive")
    events = []
    started = time.monotonic()
    for repeat_index in range(repeats):
        event = repeat_event(
            evidence_model,
            a2_model,
            loaders,
            index,
            spec,
            task,
            device,
            repeat_index,
            args.eval_step_cap,
        )
        append_jsonl(trajectory, event)
        events.append(event)
    manifest = {
        "experiment": "tier_fixed_checkpoint_dynamic_reliability",
        "dataset": task["dataset"],
        "model": "TIER-EvidenceOnly",
        "evidence_family": spec["family"],
        "evidence_selection": task["selection"],
        "experiment_label": task["experiment_label"],
        "model_seed": int(task["model_seed"]),
        "audit_seed": int(task["audit_seed"]),
        "git_commit": commit,
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config": relative_config,
        "checkpoint": task["checkpoint"],
        "a2_checkpoint": task["a2_checkpoint"],
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "fixed_target_panel": False,
        "dedicated_evaluation_generator": False,
        "sampler_rng_restoration": False,
        "registered_repeats": int(task["repeats"]),
        "repeats": repeats,
        "eval_step_cap": args.eval_step_cap,
        "events": events,
        "summary": summarize_events(events),
        "validation_loader_iterations": sum(
            row["validation_loader_iterations"] for row in events),
        "test_loader_iterations": sum(
            row["test_loader_iterations"] for row in events),
        "elapsed_seconds": time.monotonic() - started,
        "sampling_variation_note": (
            "descriptive_repeated_dynamic_sampling_audit"
        ),
    }
    manifest_path = output_dir / "tier_reliability_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": task["dataset"],
        "selection": task["selection"],
        "experiment_label": task["experiment_label"],
        "audit_seed": task["audit_seed"],
        "normal_test_f1": manifest["summary"]["normal_test_f1"],
        "normal_minus_shuffled_f1": (
            manifest["summary"]["normal_minus_shuffled_f1"]),
        "same_batch_delta_vs_frozen_a2": (
            manifest["summary"][
                "same_batch_delta_vs_frozen_a2"]),
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
