#!/usr/bin/env python3
"""Run one preregistered CDVT Phase 1 dynamic-random task."""

import argparse
import json
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from torch_geometric.loader import LinkNeighborLoader

import fraudGT  # noqa: F401
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.model_builder import create_model
from fraudGT.sampler.custom_sampler import AddEgoIdsForLinkNeighbor
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    aggregate_unique,
    best_f1_threshold,
    cosine_schedule,
    metric_row,
)


TASK = ("node", "to", "node")
CONDITIONS = ("normal", "shuffled", "off")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "account_only", "event_only", "additive_view", "dual_view"
        ),
        required=True,
    )
    parser.add_argument(
        "--experiment-label",
        choices=(
            "account_only", "event_only", "causal_event_add",
            "dual_view", "full_cdvt"
        ),
        required=True,
    )
    parser.add_argument("--lambda-cons", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--early-stop-min-epoch", type=int, default=80)
    parser.add_argument("--early-stop-patience-evals", type=int, default=10)
    return parser.parse_args()


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def configure(args):
    config = yaml.safe_load(args.config.read_text())
    if config["train"]["sampler"] != "link_neighbor":
        raise RuntimeError("Phase 1 requires LinkNeighborLoader")
    if config["val"].get("fixed_target_panel") is not False:
        raise RuntimeError("fixed target panel is forbidden")
    if config["dataset"].get("tier_evidence") is not True:
        raise RuntimeError("CDVT requires immutable raw transaction fields")
    if git_output("status", "--porcelain"):
        raise RuntimeError("refusing to train from a dirty worktree")
    set_cfg(cfg)
    load_cfg(cfg, Namespace(cfg_file=str(args.config), opts=[]))
    cfg.device = args.device
    cfg.cdvt.variant = args.variant
    cfg.cdvt.lambda_cons = float(args.lambda_cons)
    cfg.optim.max_epoch = int(args.max_epochs)
    cfg.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False
    cfg.val.fixed_target_panel = False
    if int(cfg.num_threads) < 1:
        raise ValueError("num_threads must be positive")
    torch.set_num_threads(int(cfg.num_threads))
    torch.set_num_interop_threads(1)
    if args.variant in {"account_only", "event_only"} \
            and args.lambda_cons != 0:
        raise ValueError(
            "account-only and event-only do not use view consistency")
    expected = {
        "account_only": ("account_only", False),
        "event_only": ("event_only", False),
        "causal_event_add": ("additive_view", False),
        "dual_view": ("dual_view", False),
        "full_cdvt": ("dual_view", True),
    }
    expected_variant, requires_consistency = expected[args.experiment_label]
    if args.variant != expected_variant:
        raise ValueError("experiment label and architecture variant differ")
    if requires_consistency != (args.lambda_cons > 0):
        raise ValueError("experiment label and consistency setting differ")
    if args.variant == "account_only":
        cfg.model.type = "GTModel"
    if args.lambda_cons < 0:
        raise ValueError("lambda_cons must be non-negative")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def audit_loaders(loaders):
    rows = []
    for split, wrapped in zip(("train", "val", "test"), loaders):
        loader = wrapped.loader
        if getattr(loader, "generator", None) is not None:
            raise RuntimeError(f"{split} loader uses a dedicated generator")
        sampler = getattr(loader, "sampler", None)
        if getattr(sampler, "generator", None) is not None:
            raise RuntimeError(f"{split} sampler uses a dedicated generator")
        rows.append({
            "split": split,
            "shuffle": True,
            "iter_per_epoch": int(len(wrapped)),
            "generator": None,
        })
    return rows


def paired_batch(data, edge_ids):
    edge_ids = edge_ids.detach().cpu().long().unique(sorted=False)
    edge_label_index = data[TASK].edge_index[:, edge_ids]
    edge_label = data[TASK].y[edge_ids]
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=cfg.train.neighbor_sizes,
        edge_label_index=(TASK, edge_label_index),
        edge_label=edge_label,
        batch_size=max(int(edge_ids.numel()), 1),
        num_workers=0,
        shuffle=True,
        transform=AddEgoIdsForLinkNeighbor(
            target_edge_ids=edge_ids,
            task=TASK,
            add_ego_ids=cfg.train.add_ego_id,
        ),
    )
    if getattr(loader, "generator", None) is not None:
        raise RuntimeError("paired training loader uses a dedicated generator")
    return next(iter(loader))


def common_positions(first_ids, second_ids):
    first_ids = first_ids.detach().cpu().long().view(-1)
    second_ids = second_ids.detach().cpu().long().view(-1)
    if (
        torch.unique(first_ids).numel() != first_ids.numel()
        or torch.unique(second_ids).numel() != second_ids.numel()
    ):
        raise AssertionError("target IDs must be unique within each view")
    if first_ids.numel() == 0 or second_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone()
    second_order = torch.argsort(second_ids)
    sorted_second = second_ids[second_order]
    positions = torch.searchsorted(sorted_second, first_ids)
    safe = positions.clamp_max(max(sorted_second.numel() - 1, 0))
    available = (
        (positions < sorted_second.numel())
        & (sorted_second[safe] == first_ids)
    )
    first_positions = available.nonzero(as_tuple=False).view(-1)
    second_positions = second_order[safe[available]]
    return first_positions, second_positions


def bernoulli_js(first_logits, second_logits):
    epsilon = 1e-6
    first = torch.sigmoid(first_logits).clamp(epsilon, 1 - epsilon)
    second = torch.sigmoid(second_logits).clamp(epsilon, 1 - epsilon)
    midpoint = 0.5 * (first + second)

    def kl(probability, reference):
        return (
            probability * torch.log(probability / reference)
            + (1 - probability) * torch.log(
                (1 - probability) / (1 - reference))
        )

    return 0.5 * (kl(first, midpoint) + kl(second, midpoint)).mean()


def normal_forward(model, batch):
    if hasattr(model, "forward_details"):
        return model.forward_details(batch, "normal")
    logits, labels = model(batch)
    head = model.post_gt
    mask = head._edge_mask(batch)
    edge_ids = batch[TASK].e_id[mask]
    if edge_ids.numel() != labels.numel():
        raise AssertionError("account-only edge IDs and labels are not aligned")
    diagnostics = {
        "target_edge_ids": edge_ids,
        "event_count": torch.ones_like(edge_ids),
        "fusion_gain_norm": torch.zeros_like(edge_ids, dtype=torch.float32),
        "event_condition": "normal",
    }
    return logits, labels, diagnostics


def train_epoch(model, loader, train_data, optimizer, device, lambda_cons):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(cfg.optim.batch_accumulation)
    totals = {"loss": 0.0, "fraud": 0.0, "consistency": 0.0}
    samples = paired_samples = requested_pairs = steps = pending = 0
    started = time.monotonic()
    for step, raw_batch in enumerate(loader):
        raw_batch.split = "train"
        raw_batch.to(device)
        first_logits, first_labels, first_diagnostics = normal_forward(
            model, raw_batch)
        fraud_loss, _ = compute_loss(first_logits, first_labels)
        consistency = first_logits.sum() * 0.0
        requested = 0
        paired = 0
        if lambda_cons > 0:
            first_ids = first_diagnostics["target_edge_ids"]
            requested = int(first_ids.numel())
            second = paired_batch(train_data, first_ids)
            second.split = "train"
            second.to(device)
            second_logits, second_labels, second_diagnostics = normal_forward(
                model, second)
            first_positions, second_positions = common_positions(
                first_ids, second_diagnostics["target_edge_ids"])
            if first_positions.numel():
                first_positions = first_positions.to(device)
                second_positions = second_positions.to(device)
                if not torch.equal(
                    first_labels[first_positions],
                    second_labels[second_positions],
                ):
                    raise AssertionError("paired-view labels differ")
                consistency = bernoulli_js(
                    first_logits[first_positions],
                    second_logits[second_positions],
                )
                paired = int(first_positions.numel())
        loss = fraud_loss + float(lambda_cons) * consistency
        (loss / accumulation).backward()
        pending += 1
        if pending == accumulation or step + 1 == len(loader):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        count = int(first_labels.numel())
        totals["loss"] += float(loss.detach()) * count
        totals["fraud"] += float(fraud_loss.detach()) * count
        totals["consistency"] += float(consistency.detach()) * max(paired, 1)
        samples += count
        paired_samples += paired
        requested_pairs += requested
        steps += 1
    return {
        key: value / max(samples, 1) for key, value in totals.items()
    } | {
        "samples": samples,
        "steps": steps,
        "paired_samples": paired_samples,
        "requested_pairs": requested_pairs,
        "paired_retention": paired_samples / max(requested_pairs, 1),
        "elapsed_seconds": time.monotonic() - started,
    }


@torch.no_grad()
def evaluate(model, loader, device, split):
    model.eval()
    edge_ids = []
    labels = []
    scores = {condition: [] for condition in CONDITIONS}
    event_counts = []
    fusion_gains = []
    started = time.monotonic()
    steps = 0
    for raw_batch in loader:
        raw_batch.split = split
        raw_batch.to(device)
        if hasattr(model, "forward_counterfactuals"):
            outputs = model.forward_counterfactuals(raw_batch)
        else:
            normal_output = normal_forward(model, raw_batch)
            outputs = {
                condition: normal_output for condition in CONDITIONS
            }
        normal_logits, batch_labels, normal_diagnostics = outputs["normal"]
        edge_ids.append(
            normal_diagnostics["target_edge_ids"].detach().cpu())
        labels.append(batch_labels.detach().cpu().long())
        for condition in CONDITIONS:
            logits, condition_labels, diagnostics = outputs[condition]
            if not torch.equal(batch_labels, condition_labels):
                raise AssertionError("counterfactual labels differ")
            scores[condition].append(torch.sigmoid(logits).detach().cpu())
        event_counts.append(
            normal_diagnostics["event_count"].detach().cpu())
        fusion_gains.append(
            normal_diagnostics["fusion_gain_norm"].detach().cpu())
        steps += 1
    return {
        "edge_ids": torch.cat(edge_ids),
        "labels": torch.cat(labels),
        "scores": {
            condition: torch.cat(parts)
            for condition, parts in scores.items()
        },
        "event_counts": torch.cat(event_counts),
        "fusion_gains": torch.cat(fusion_gains),
        "steps": steps,
        "elapsed_seconds": time.monotonic() - started,
    }


def unique_scores(evaluation, condition):
    return aggregate_unique(
        evaluation["edge_ids"],
        evaluation["labels"],
        evaluation["scores"][condition],
    )


def intervention_row(labels, normal, counterfactual, threshold):
    truth = labels.bool()
    normal_prediction = normal >= threshold
    counterfactual_prediction = counterfactual >= threshold
    changed = normal_prediction != counterfactual_prediction
    corrected = changed & (normal_prediction == truth)
    broken = changed & (counterfactual_prediction == truth)
    return {
        "changed": int(changed.sum()),
        "normal_correct_counterfactual_wrong": int(corrected.sum()),
        "normal_wrong_counterfactual_correct": int(broken.sum()),
    }


def evaluation_event(model, loaders, device, epoch, train):
    validation = evaluate(model, loaders[1], device, "val")
    test = evaluate(model, loaders[2], device, "test")
    _, val_labels, val_normal = unique_scores(validation, "normal")
    threshold, val_f1 = best_f1_threshold(val_labels, val_normal)
    test_rows = {}
    unique_test = {}
    for condition in CONDITIONS:
        ids, labels, scores = unique_scores(test, condition)
        unique_test[condition] = (ids, labels, scores)
        test_rows[condition] = metric_row(labels, scores, threshold)
    normal_ids, test_labels, normal_scores = unique_test["normal"]
    interventions = {}
    for condition in ("shuffled", "off"):
        ids, labels, condition_scores = unique_test[condition]
        if not torch.equal(normal_ids, ids) or not torch.equal(
            test_labels, labels
        ):
            raise AssertionError("counterfactual test IDs differ")
        interventions[condition] = intervention_row(
            test_labels, normal_scores, condition_scores, threshold)
    return {
        "epoch": epoch,
        "train": train,
        "val_normal": metric_row(val_labels, val_normal, threshold),
        "test": test_rows,
        "threshold": threshold,
        "val_f1": val_f1,
        "mean_event_count": float(
            test["event_counts"].float().mean()),
        "mean_fusion_gain": float(
            test["fusion_gains"].float().mean()),
        "interventions": interventions,
        "validation_steps": validation["steps"],
        "test_steps": test["steps"],
        "validation_seconds": validation["elapsed_seconds"],
        "test_seconds": test["elapsed_seconds"],
    }


def main():
    args = parse_args()
    config_snapshot = configure(args)
    commit = git_output("rev-parse", "HEAD")
    protected_outputs = (
        args.output_dir / "manifest.json",
        args.output_dir / "trajectory.jsonl",
        args.output_dir / "best_val.ckpt",
        args.output_dir / "progress.json",
    )
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite experiment artifacts: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.seed))
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_audit = audit_loaders(loaders)
    device = torch.device(args.device)
    model = create_model(dataset=dataset).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.base_lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    scheduler = cosine_schedule(
        optimizer, int(args.max_epochs), int(cfg.optim.num_warmup_epochs))

    trajectory_path = args.output_dir / "trajectory.jsonl"
    checkpoint_path = args.output_dir / "best_val.ckpt"
    progress_path = args.output_dir / "progress.json"
    best_val = -1.0
    best_event = None
    stale_evals = 0
    events = []
    started = time.monotonic()
    for epoch in range(int(args.max_epochs)):
        train = train_epoch(
            model, loaders[0], dataset["train"], optimizer, device,
            float(args.lambda_cons))
        scheduler.step()
        progress = {
            "architecture_variant": args.variant,
            "dataset": str(cfg.dataset.name),
            "elapsed_seconds": time.monotonic() - started,
            "epoch": epoch,
            "git_commit": commit,
            "lambda_cons": float(args.lambda_cons),
            "sampling_protocol": "dynamic_random",
            "seed": int(cfg.seed),
            "train": train,
            "variant": args.experiment_label,
        }
        progress_tmp = progress_path.with_suffix(".tmp")
        progress_tmp.write_text(json.dumps(progress, sort_keys=True))
        progress_tmp.replace(progress_path)
        if (epoch + 1) % int(cfg.train.eval_period):
            continue
        event = evaluation_event(model, loaders, device, epoch, train)
        events.append(event)
        with trajectory_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if event["val_f1"] > best_val:
            best_val = event["val_f1"]
            best_event = event
            stale_evals = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_val,
                "git_commit": commit,
            }, checkpoint_path)
        else:
            stale_evals += 1
        print(json.dumps({
            "epoch": epoch,
            "val_f1": event["val_f1"],
            "test_f1": event["test"]["normal"]["f1"],
            "paired_retention": train["paired_retention"],
        }, sort_keys=True), flush=True)
        if (
            epoch + 1 >= int(args.early_stop_min_epoch)
            and stale_evals >= int(args.early_stop_patience_evals)
        ):
            break
    if best_event is None:
        raise RuntimeError("no evaluation event was produced")
    baseline = INITIAL_A2[str(cfg.dataset.name)]
    selected_test = best_event["test"]["normal"]["f1"]
    raw_best_event = max(
        events, key=lambda event: event["test"]["normal"]["f1"])
    raw_best = raw_best_event["test"]["normal"]["f1"]
    manifest = {
        "phase": "CDVT_phase1",
        "sampling_protocol": "dynamic_random",
        "dataset": str(cfg.dataset.name),
        "variant": args.experiment_label,
        "architecture_variant": args.variant,
        "lambda_cons": float(args.lambda_cons),
        "seed": int(cfg.seed),
        "git_commit": commit,
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "val_selected_epoch": int(best_event["epoch"]),
        "val_selected_test_f1": selected_test,
        "raw_best_epoch": int(raw_best_event["epoch"]),
        "raw_best_test_f1": raw_best,
        "delta_val_selected_vs_initial_a2": (
            selected_test - baseline["val_selected_test_f1"]),
        "delta_raw_best_vs_initial_a2": (
            raw_best - baseline["raw_best_test_f1"]),
        "initial_a2": baseline,
        "best_event": best_event,
        "events": len(events),
        "epochs_completed": int(events[-1]["epoch"] + 1),
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": int(parameter_count),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "loader_audit": loader_audit,
        "config_snapshot": config_snapshot,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: manifest[key]
        for key in (
            "dataset", "variant", "lambda_cons",
            "val_selected_test_f1", "delta_val_selected_vs_initial_a2",
            "raw_best_test_f1", "delta_raw_best_vs_initial_a2",
        )
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
