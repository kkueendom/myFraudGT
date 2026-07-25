#!/usr/bin/env python3
"""Repeated dynamic-random stability audit for fixed initial-A2 checkpoints."""

import argparse
import json
import statistics
import subprocess
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
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    TASK,
    best_f1_threshold,
    configure_fraudgt,
    loader_audit,
    load_a2_model,
    seed_process,
)


def git_output(*args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def verify_repository(spec):
    if git_output("status", "--porcelain"):
        raise RuntimeError("refusing to audit from a dirty worktree")
    branch = git_output("branch", "--show-current")
    expected = spec.get("expected_branch")
    if expected and branch and branch != expected:
        raise RuntimeError(f"expected branch {expected}, found {branch}")
    return git_output("rev-parse", "--short=8", "HEAD")


def audit_protocol(spec, task, config):
    if spec.get("sampling_protocol") != "dynamic_random":
        raise ValueError("sampling_protocol must be dynamic_random")
    if config.get("val", {}).get("fixed_target_panel") is True:
        raise ValueError("val.fixed_target_panel must not be true")
    if int(config["train"]["batch_size"]) != int(
        task["expected_batch_size"]
    ):
        raise ValueError("train.batch_size differs from registered value")
    if int(config["val"]["iter_per_epoch"]) != int(
        task["expected_val_iter_per_epoch"]
    ):
        raise ValueError("val.iter_per_epoch differs from registered value")
    sampler_source = (
        REPO_ROOT / "fraudGT/sampler/custom_sampler.py"
    ).read_text()
    loader_source = (
        REPO_ROOT / "fraudGT/graphgym/loader.py"
    ).read_text()
    runner_source = Path(__file__).read_text()
    forbidden_sampler = (
        "_fixed_target_panel",
        "reset_generator",
        "generator=reset_generator",
    )
    forbidden_runner = (
        "torch." + "Generator(",
        "get_" + "rng_state(",
        "set_" + "rng_state(",
    )
    if any(token in sampler_source for token in forbidden_sampler):
        raise RuntimeError("fixed-panel or RNG-reset sampler logic detected")
    if any(token in runner_source for token in forbidden_runner):
        raise RuntimeError("independent or restored evaluation RNG detected")
    if "shuffle=shuffle" not in sampler_source:
        raise RuntimeError("LinkNeighborLoader no longer forwards shuffle")
    if "def create_loader(dataset = None, shuffle = True" not in loader_source:
        raise RuntimeError("create_loader no longer defaults to shuffle=True")


def metric_row(labels, scores, threshold):
    predictions = scores >= float(threshold)
    labels = labels.bool()
    tp = int((predictions & labels).sum())
    fp = int((predictions & (~labels)).sum())
    fn = int(((~predictions) & labels).sum())
    tn = int(((~predictions) & (~labels)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denominator = 2 * tp + fp + fn
    return {
        "f1": 2 * tp / denominator if denominator else 0.0,
        "precision": precision,
        "recall": recall,
        "threshold": float(threshold),
        "samples": int(labels.numel()),
        "positives": int(labels.sum()),
        "prevalence": float(labels.float().mean()),
        "predicted_positives": int(predictions.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate(model, loader, split, device, step_cap=None):
    model.eval()
    edge_ids_all = []
    labels_all = []
    scores_all = []
    started = time.monotonic()
    steps = 0
    for batch in loader:
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
                "A2 outputs are not aligned to target edge IDs")
        edge_ids_all.append(edge_ids)
        labels_all.append(labels)
        scores_all.append(torch.sigmoid(logits))
        steps += 1
        if step_cap is not None and steps >= int(step_cap):
            break
    edge_ids = torch.cat(edge_ids_all)
    labels = torch.cat(labels_all)
    scores = torch.cat(scores_all)
    unique_edges = int(torch.unique(edge_ids).numel())
    return {
        "edge_ids": edge_ids,
        "labels": labels,
        "scores": scores,
        "steps": steps,
        "samples": int(labels.numel()),
        "unique_edges": unique_edges,
        "unique_edge_rate": unique_edges / max(int(labels.numel()), 1),
        "elapsed_seconds": time.monotonic() - started,
    }


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


def summarize(events, historical_f1):
    fields = {
        "val_f1": [row["val"]["f1"] for row in events],
        "test_f1": [row["test"]["f1"] for row in events],
        "delta_vs_initial_a2": [
            row["test"]["f1"] - historical_f1 for row in events],
        "val_threshold": [
            row["val"]["threshold"] for row in events],
        "test_precision": [
            row["test"]["precision"] for row in events],
        "test_recall": [
            row["test"]["recall"] for row in events],
        "test_prevalence": [
            row["test"]["prevalence"] for row in events],
        "test_positive_count": [
            row["test"]["positives"] for row in events],
        "test_unique_edge_rate": [
            row["test_unique_edge_rate"] for row in events],
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
    config_path = Path(task["config"])
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    seed_process(int(task["audit_seed"]))
    configure_fraudgt(config_path, task["audit_seed"], args.device)
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
    trajectory = output_dir / "stability_trajectory.jsonl"
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
    historical_f1 = INITIAL_A2[task["dataset"]][
        "val_selected_test_f1"]
    for repeat_index in range(repeats):
        val = evaluate(
            model,
            loaders[1],
            "val",
            device,
            args.eval_step_cap,
        )
        threshold, _ = best_f1_threshold(
            val["labels"], val["scores"])
        test = evaluate(
            model,
            loaders[2],
            "test",
            device,
            args.eval_step_cap,
        )
        event = {
            "repeat": repeat_index,
            "sampling_protocol": "dynamic_random",
            "val": metric_row(
                val["labels"], val["scores"], threshold),
            "test": metric_row(
                test["labels"], test["scores"], threshold),
            "delta_vs_initial_a2": (
                metric_row(
                    test["labels"], test["scores"], threshold
                )["f1"] - historical_f1
            ),
            "validation_loader_iterations": val["steps"],
            "test_loader_iterations": test["steps"],
            "val_unique_edges": val["unique_edges"],
            "test_unique_edges": test["unique_edges"],
            "val_unique_edge_rate": val["unique_edge_rate"],
            "test_unique_edge_rate": test["unique_edge_rate"],
            "val_elapsed_seconds": val["elapsed_seconds"],
            "test_elapsed_seconds": test["elapsed_seconds"],
        }
        append_jsonl(trajectory, event)
        events.append(event)
    manifest = {
        "experiment": "initial_a2_dynamic_sampling_stability",
        "dataset": task["dataset"],
        "model": "initial-A2-fixed-checkpoint",
        "variant": "a2_multi",
        "experiment_label": task["experiment_label"],
        "model_seed": int(task["model_seed"]),
        "audit_seed": int(task["audit_seed"]),
        "git_commit": commit,
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": str(config_path),
        "checkpoint": task["checkpoint"],
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "fixed_target_panel": False,
        "dedicated_evaluation_generator": False,
        "sampler_rng_restoration": False,
        "registered_repeats": int(task["repeats"]),
        "repeats": repeats,
        "eval_step_cap": args.eval_step_cap,
        "initial_a2_val_selected_test_f1": historical_f1,
        "events": events,
        "summary": summarize(events, historical_f1),
        "validation_loader_iterations": sum(
            row["validation_loader_iterations"] for row in events),
        "test_loader_iterations": sum(
            row["test_loader_iterations"] for row in events),
        "raw_best_repeated_test_f1": max(
            row["test"]["f1"] for row in events),
        "elapsed_seconds": time.monotonic() - started,
        "sampling_variation_note": (
            "descriptive_repeated_dynamic_sampling_audit_not_a_new_baseline"
        ),
    }
    manifest_path = output_dir / "a2_stability_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": task["dataset"],
        "experiment_label": task["experiment_label"],
        "audit_seed": task["audit_seed"],
        "test_f1": manifest["summary"]["test_f1"],
        "delta_vs_initial_a2": (
            manifest["summary"]["delta_vs_initial_a2"]),
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
