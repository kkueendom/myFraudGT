#!/usr/bin/env python3
"""Paired nested-prefix audit of dynamic evaluation budget convergence."""

import argparse
import hashlib
import json
import math
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
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.a2_dynamic_stability_audit import (
    audit_protocol,
    configure_a2_fraudgt,
    loader_audit,
    metric_row,
    verify_repository,
)
from run.nested_dynamic_stability_audit import expand_tasks
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    TASK,
    best_f1_threshold,
    load_a2_model,
    seed_process,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-budget", type=int)
    return parser.parse_args()


def _sha256(values):
    return hashlib.sha256(
        values.contiguous().numpy().tobytes()
    ).hexdigest()


@torch.no_grad()
def evaluate_prefixes(model, loader, split, device, budgets):
    model.eval()
    budgets = tuple(sorted(int(value) for value in budgets))
    maximum = budgets[-1]
    edge_parts = []
    label_parts = []
    score_parts = []
    snapshots = {}
    started = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        batch.split = split
        batch.to(device)
        mask = model.post_gt._edge_mask(batch)
        edge_ids = batch[TASK].e_id[mask].detach().cpu()
        logits, labels = model(batch)
        logits = logits.squeeze(-1).detach().cpu()
        labels = labels.detach().cpu().long().view(-1)
        if not (
            edge_ids.numel() == logits.numel() == labels.numel()
        ):
            raise AssertionError(
                "model outputs are not aligned to target edge IDs")
        edge_parts.append(edge_ids)
        label_parts.append(labels)
        score_parts.append(torch.sigmoid(logits))
        if step in budgets:
            all_edges = torch.cat(edge_parts)
            all_labels = torch.cat(label_parts)
            all_scores = torch.cat(score_parts)
            unique_edges = int(torch.unique(all_edges).numel())
            snapshots[step] = {
                "edge_ids": all_edges.clone(),
                "labels": all_labels.clone(),
                "scores": all_scores.clone(),
                "steps": step,
                "samples": int(all_labels.numel()),
                "unique_edges": unique_edges,
                "unique_edge_rate": (
                    unique_edges / max(int(all_labels.numel()), 1)
                ),
                "edge_id_sha256": _sha256(all_edges),
            }
        if step >= maximum:
            break
    if set(snapshots) != set(budgets):
        raise RuntimeError(
            "loader ended before every registered budget was observed")
    elapsed = time.monotonic() - started
    return snapshots, elapsed


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_budget_rows(val_prefixes, test_prefixes, historical_f1):
    rows = {}
    for budget in sorted(val_prefixes):
        val = val_prefixes[budget]
        test = test_prefixes[budget]
        threshold, _ = best_f1_threshold(
            val["labels"], val["scores"])
        val_metric = metric_row(
            val["labels"], val["scores"], threshold)
        test_metric = metric_row(
            test["labels"], test["scores"], threshold)
        delta = test_metric["f1"] - historical_f1
        rows[str(budget)] = {
            "budget": budget,
            "val": val_metric,
            "test": test_metric,
            "delta_vs_initial_a2": delta,
            "delta_sign_vs_initial_a2": _sign(delta),
            "validation_loader_iterations": val["steps"],
            "test_loader_iterations": test["steps"],
            "val_unique_edges": val["unique_edges"],
            "test_unique_edges": test["unique_edges"],
            "val_unique_edge_rate": val["unique_edge_rate"],
            "test_unique_edge_rate": test["unique_edge_rate"],
            "val_edge_id_sha256": val["edge_id_sha256"],
            "test_edge_id_sha256": test["edge_id_sha256"],
        }
    reference = rows[str(max(val_prefixes))]
    for row in rows.values():
        signed_error = (
            row["test"]["f1"] - reference["test"]["f1"])
        row["paired_f1_error_vs_reference"] = signed_error
        row["paired_absolute_f1_error_vs_reference"] = abs(
            signed_error)
        row["paired_threshold_error_vs_reference"] = (
            row["val"]["threshold"]
            - reference["val"]["threshold"]
        )
        row["delta_sign_agrees_with_reference"] = (
            row["delta_sign_vs_initial_a2"]
            == reference["delta_sign_vs_initial_a2"]
        )
    return rows


def verify_nested_prefixes(prefixes):
    maximum = prefixes[max(prefixes)]["edge_ids"]
    for budget, row in prefixes.items():
        prefix = row["edge_ids"]
        if not torch.equal(prefix, maximum[:prefix.numel()]):
            raise AssertionError(
                f"budget {budget} is not a prefix of the full sequence")
    return True


def append_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _check_finite(value, path="root"):
    if isinstance(value, dict):
        for key, nested in value.items():
            _check_finite(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _check_finite(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")


def run_task(spec, task, args):
    commit = verify_repository(spec)
    config_path = Path(task["config"])
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    budgets = tuple(int(value) for value in spec["budgets"])
    if budgets != tuple(sorted(set(budgets))):
        raise ValueError("budgets must be unique and increasing")
    if int(spec["reference_budget"]) != budgets[-1]:
        raise ValueError("reference budget must equal maximum budget")
    maximum = (
        int(args.max_budget)
        if args.max_budget is not None
        else budgets[-1]
    )
    selected_budgets = tuple(
        value for value in budgets if value <= maximum)
    if not selected_budgets or selected_budgets[-1] != maximum:
        raise ValueError("max budget must be a registered budget")

    seed_process(int(task["audit_seed"]))
    dropped, normalized = configure_a2_fraudgt(
        config, task["audit_seed"], args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    device = torch.device(args.device)
    model = load_a2_model(
        dataset, task["checkpoint"], device).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(task["checkpoint"], map_location="cpu")

    output_dir = args.output_dir / (
        f"{task['dataset']}_{task['experiment_label']}_"
        f"auditseed{task['audit_seed']}_{commit}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "budget_trajectory.jsonl"
    repeats = (
        int(args.repeats)
        if args.repeats is not None
        else int(task["repeats"])
    )
    if repeats < 1:
        raise ValueError("repeats must be positive")

    historical_f1 = INITIAL_A2[task["dataset"]][
        "val_selected_test_f1"]
    events = []
    started = time.monotonic()
    for repeat_index in range(repeats):
        val_prefixes, val_elapsed = evaluate_prefixes(
            model, loaders[1], "val", device, selected_budgets)
        test_prefixes, test_elapsed = evaluate_prefixes(
            model, loaders[2], "test", device, selected_budgets)
        nested_prefix_verified = (
            verify_nested_prefixes(val_prefixes)
            and verify_nested_prefixes(test_prefixes)
        )
        event = {
            "repeat": repeat_index,
            "sampling_protocol": "dynamic_random",
            "nested_prefix_verified": nested_prefix_verified,
            "budgets": build_budget_rows(
                val_prefixes, test_prefixes, historical_f1),
            "val_elapsed_seconds": val_elapsed,
            "test_elapsed_seconds": test_elapsed,
        }
        _check_finite(event)
        append_jsonl(trajectory, event)
        events.append(event)

    manifest = {
        "experiment": spec["experiment"],
        "dataset": task["dataset"],
        "model": task["model"],
        "variant": task["variant"],
        "experiment_label": task["experiment_label"],
        "model_seed": int(task["model_seed"]),
        "audit_seed": int(task["audit_seed"]),
        "git_commit": commit,
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
            if checkpoint.get("epoch") is not None
            else int(Path(task["checkpoint"]).stem)
        ),
        "config": str(config_path),
        "checkpoint": task["checkpoint"],
        "dropped_runtime_config_keys": dropped,
        "normalized_archived_config_values": normalized,
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "fixed_target_panel": False,
        "dedicated_evaluation_generator": False,
        "sampler_rng_restoration": False,
        "nested_prefix_construction": (
            "cumulative_prefixes_from_one_256_batch_trajectory"),
        "registered_budgets": list(budgets),
        "budgets": list(selected_budgets),
        "reference_budget": maximum,
        "registered_repeats": int(task["repeats"]),
        "repeats": repeats,
        "initial_a2_val_selected_test_f1": historical_f1,
        "events": events,
        "validation_loader_iterations": (
            repeats * maximum),
        "test_loader_iterations": repeats * maximum,
        "elapsed_seconds": time.monotonic() - started,
        "comparison_note": spec["comparison_note"],
    }
    _check_finite(manifest)
    manifest_path = output_dir / spec["manifest_filename"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": task["dataset"],
        "model_seed": task["model_seed"],
        "audit_seed": task["audit_seed"],
        "repeats": repeats,
        "budgets": list(selected_budgets),
        "output": str(output_dir),
    }, sort_keys=True))


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    tasks = expand_tasks(spec)
    if len(tasks) != int(spec["task_count"]):
        raise RuntimeError(
            "expanded task count differs from specification")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError("task index is outside registered task list")
    task = tasks[args.task_index]
    for path in (task["config"], task["checkpoint"]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    run_task(spec, task, args)


if __name__ == "__main__":
    main()
