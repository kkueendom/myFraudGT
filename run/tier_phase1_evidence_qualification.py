#!/usr/bin/env python3
"""Train and audit an independent TIER evidence classifier."""

import argparse
import json
import math
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
import torch.nn.functional as F
import yaml

import fraudGT  # noqa: F401 - registers FraudGT components
from fraudGT.evidence.tier import EvidenceBatch, TemporalIncidentIndex
from fraudGT.evidence.tier_model import TransactionEvidenceEncoder
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from fraudGT.graphgym.model_builder import create_model


TASK = ("node", "to", "node")
ROLE_NAMES = (
    "dst_to_src",
    "src_from_src",
    "dst_to_dst",
    "src_from_dst",
    "reverse",
)
MOTIF_NAMES = ("reciprocal", "relay", "cycle")
INITIAL_A2 = {
    "Small-LI": {
        "val_selected_test_f1": 0.46247,
        "raw_best_test_f1": 0.50667,
    },
    "Small-HI": {
        "val_selected_test_f1": 0.77984,
        "raw_best_test_f1": 0.79497,
    },
    "Medium-LI": {
        "val_selected_test_f1": 0.51163,
        "raw_best_test_f1": 0.59031,
    },
    "Medium-HI": {
        "val_selected_test_f1": 0.77574,
        "raw_best_test_f1": 0.78940,
    },
    "Large-LI": {
        "val_selected_test_f1": 0.30108,
        "raw_best_test_f1": 0.44720,
    },
    "Large-HI": {
        "val_selected_test_f1": 0.72897,
        "raw_best_test_f1": 0.76223,
    },
}


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def verify_repository(spec):
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError("refusing to train from a dirty worktree")
    branch = git_output("branch", "--show-current")
    expected = spec.get("expected_branch")
    if expected and branch and branch != expected:
        raise RuntimeError(f"expected branch {expected}, found {branch}")
    return git_output("rev-parse", "--short=8", "HEAD")


def load_task_config(spec, task):
    relative = spec["config_template"].format(dataset=task["dataset"])
    path = REPO_ROOT / relative
    return relative, path, yaml.safe_load(path.read_text())


def audit_protocol(spec, task, config):
    if spec.get("sampling_protocol") != "dynamic_random":
        raise ValueError("sampling_protocol must be dynamic_random")
    if config["dataset"].get("tier_evidence") is not True:
        raise ValueError("dataset.tier_evidence must be true")
    if config["val"].get("fixed_target_panel") is not False:
        raise ValueError("val.fixed_target_panel must be false")
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


def configure_fraudgt(config_path, seed, device):
    set_cfg(cfg)
    load_cfg(cfg, Namespace(cfg_file=str(config_path), opts=[]))
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False
    cfg.val.fixed_target_panel = False
    torch.set_num_threads(int(cfg.num_threads))


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_audit(loaders):
    rows = []
    for split, wrapped in zip(("train", "val", "test"), loaders):
        base = wrapped.loader
        generator = getattr(base, "generator", None)
        sampler = getattr(base, "sampler", None)
        sampler_generator = getattr(sampler, "generator", None)
        if generator is not None or sampler_generator is not None:
            raise RuntimeError(
                f"{split} loader uses a dedicated random generator"
            )
        rows.append(
            {
                "split": split,
                "shuffle": True,
                "loader_generator": None,
                "sampler_generator": None,
                "configured_steps": int(len(wrapped)),
                "batch_size": int(base.batch_size),
            }
        )
    return rows


def move_evidence(evidence, device):
    return EvidenceBatch(
        tokens=evidence.tokens.to(device, non_blocking=True),
        mask=evidence.mask.to(device, non_blocking=True),
        context_edge_ids=evidence.context_edge_ids.to(
            device, non_blocking=True
        ),
        support=evidence.support.to(device, non_blocking=True),
    )


def query_batch(index, batch, spec, selection, device):
    store = batch[TASK]
    target_ids = store.target_edge_id.detach().cpu().long()
    labels = store.edge_label.detach().cpu().long().view(-1)
    started = time.monotonic()
    evidence_cpu = index.query(
        target_ids,
        max_tokens=int(spec["max_tokens"]),
        selection=selection,
        selection_pool_factor=int(spec["selection_pool_factor"]),
    )
    query_seconds = time.monotonic() - started
    target_raw = index.raw_edge_attr[target_ids]
    return (
        target_ids,
        labels,
        move_evidence(evidence_cpu, device),
        target_raw.to(device, non_blocking=True),
        query_seconds,
        evidence_cpu,
    )


def binary_f1(labels, predictions):
    labels = labels.long().view(-1)
    predictions = predictions.long().view(-1)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def best_f1_threshold(labels, scores):
    labels = labels.long().view(-1)
    scores = scores.float().view(-1)
    if not labels.numel():
        return 0.5, 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = sorted_labels.cumsum(0).float()
    fp = torch.arange(1, labels.numel() + 1).float() - tp
    positives = float(sorted_labels.sum())
    fn = positives - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    endpoints = torch.ones_like(sorted_scores, dtype=torch.bool)
    endpoints[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    f1 = f1.masked_fill(~endpoints, -1)
    best = int(torch.argmax(f1))
    return float(sorted_scores[best]), float(f1[best])


def average_precision(labels, scores):
    labels = labels.long().view(-1)
    scores = scores.float().view(-1)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].float()
    precision = ranked.cumsum(0) / torch.arange(
        1, ranked.numel() + 1, dtype=torch.float32
    )
    return float((precision * ranked).sum() / positives)


def metric_row(labels, scores, threshold):
    predictions = scores >= float(threshold)
    positives = int(labels.sum())
    predicted_positives = int(predictions.sum())
    return {
        "f1": binary_f1(labels, predictions),
        "auprc": average_precision(labels, scores),
        "threshold": float(threshold),
        "samples": int(labels.numel()),
        "positives": positives,
        "predicted_positives": predicted_positives,
    }


def aggregate_unique(target_ids, labels, scores):
    target_ids = target_ids.long().view(-1)
    labels = labels.long().view(-1)
    scores = scores.float().view(-1)
    unique_ids, inverse = torch.unique(
        target_ids, sorted=True, return_inverse=True
    )
    score_sums = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    counts = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    label_sums = torch.zeros(unique_ids.numel(), dtype=torch.long)
    score_sums.scatter_add_(0, inverse, scores)
    counts.scatter_add_(0, inverse, torch.ones_like(scores))
    label_sums.scatter_add_(0, inverse, labels)
    if not torch.equal(
        label_sums, (label_sums > 0).long() * counts.long()
    ):
        raise AssertionError("one edge ID has inconsistent labels")
    return unique_ids, (label_sums > 0).long(), score_sums / counts


def tensor_distribution(values):
    values = values.float().view(-1)
    if not values.numel():
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def summarize_evidence_diagnostics(
    labels,
    token_counts,
    supports,
    role_activations,
    motif_activations,
):
    covered = token_counts > 0
    class_rows = {}
    for class_id in (0, 1):
        class_mask = labels == class_id
        count = int(class_mask.sum())
        class_rows[str(class_id)] = {
            "samples": count,
            "coverage_rate": (
                float(covered[class_mask].float().mean())
                if count
                else None
            ),
            "token_count": tensor_distribution(token_counts[class_mask]),
            "support_mean": (
                supports[class_mask].float().mean(0).tolist()
                if count
                else [0.0] * 6
            ),
        }
    valid_tokens = max(int(token_counts.sum()), 1)
    return {
        "coverage_rate": float(covered.float().mean()),
        "token_count": tensor_distribution(token_counts),
        "class": class_rows,
        "role_token_activation_rate": {
            name: float(role_activations[position] / valid_tokens)
            for position, name in enumerate(ROLE_NAMES)
        },
        "motif_token_activation_rate": {
            name: float(motif_activations[position] / valid_tokens)
            for position, name in enumerate(MOTIF_NAMES)
        },
    }


def train_epoch(
    model,
    loader,
    index,
    optimizer,
    spec,
    selection,
    device,
    step_cap,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(cfg.optim.batch_accumulation)
    loss_total = 0.0
    samples = 0
    query_seconds = 0.0
    steps = 0
    pending = 0
    for batch in loader:
        _, labels, evidence, target_raw, query_time, _ = query_batch(
            index, batch, spec, selection, device
        )
        labels_device = labels.float().to(device, non_blocking=True)
        logits, _ = model(evidence, target_raw)
        weights = torch.where(
            labels_device > 0,
            torch.full_like(labels_device, float(spec["positive_weight"])),
            torch.ones_like(labels_device),
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, labels_device, weight=weights
        )
        (loss / accumulation).backward()
        pending += 1
        if pending == accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        count = int(labels.numel())
        loss_total += float(loss.detach()) * count
        samples += count
        query_seconds += query_time
        steps += 1
        if step_cap and steps >= step_cap:
            break
    if pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_total / max(samples, 1),
        "samples": samples,
        "steps": steps,
        "query_seconds": query_seconds,
    }


@torch.no_grad()
def evaluate_evidence(
    model,
    loader,
    index,
    spec,
    selection,
    device,
    step_cap,
    perturbations,
):
    model.eval()
    target_ids_all = []
    labels_all = []
    scores = {name: [] for name in perturbations}
    token_counts_all = []
    supports_all = []
    role_activations = torch.zeros(len(ROLE_NAMES), dtype=torch.float64)
    motif_activations = torch.zeros(len(MOTIF_NAMES), dtype=torch.float64)
    query_seconds = 0.0
    steps = 0
    for batch in loader:
        (
            target_ids,
            labels,
            evidence,
            target_raw,
            query_time,
            evidence_cpu,
        ) = query_batch(
            index, batch, spec, selection, device
        )
        target_ids_all.append(target_ids)
        labels_all.append(labels)
        query_seconds += query_time
        token_counts_all.append(evidence_cpu.mask.sum(dim=1))
        supports_all.append(evidence_cpu.support)
        valid_tokens = evidence_cpu.tokens[evidence_cpu.mask]
        if valid_tokens.numel():
            role_start = index.raw_edge_attr.size(1) + 1
            motif_start = role_start + len(ROLE_NAMES)
            role_activations += valid_tokens[
                :, role_start:motif_start
            ].double().sum(0)
            motif_activations += valid_tokens[
                :, motif_start:motif_start + len(MOTIF_NAMES)
            ].double().sum(0)
        if "normal" in scores:
            logits, _ = model(evidence, target_raw)
            scores["normal"].append(torch.sigmoid(logits).cpu())
        if "shuffled" in scores:
            permutation = torch.randperm(
                evidence.tokens.size(0), device=device
            )
            logits, _ = model(evidence.shuffled(permutation), target_raw)
            scores["shuffled"].append(torch.sigmoid(logits).cpu())
        if "off" in scores:
            logits, _ = model(evidence.off(), target_raw)
            scores["off"].append(torch.sigmoid(logits).cpu())
        steps += 1
        if step_cap and steps >= step_cap:
            break
    labels = torch.cat(labels_all) if labels_all else torch.empty(0)
    token_counts = (
        torch.cat(token_counts_all)
        if token_counts_all
        else torch.empty(0)
    )
    supports = (
        torch.cat(supports_all)
        if supports_all
        else torch.empty((0, index.SUPPORT_DIM))
    )
    return {
        "target_edge_ids": (
            torch.cat(target_ids_all)
            if target_ids_all
            else torch.empty(0, dtype=torch.long)
        ),
        "labels": labels,
        "scores": {
            name: torch.cat(parts) if parts else torch.empty(0)
            for name, parts in scores.items()
        },
        "evidence_diagnostics": summarize_evidence_diagnostics(
            labels,
            token_counts,
            supports,
            role_activations,
            motif_activations,
        ),
        "steps": steps,
        "query_seconds": query_seconds,
    }


def cosine_schedule(optimizer, max_epochs, warmup_epochs):
    def multiplier(epoch):
        position = epoch + 1
        if position <= warmup_epochs:
            return position / max(warmup_epochs, 1)
        progress = (
            (position - warmup_epochs)
            / max(max_epochs - warmup_epochs, 1)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def append_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_a2_model(dataset, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"A2 diagnostic checkpoint not found: {checkpoint_path}"
        )
    model = create_model(dataset=dataset)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def paired_scores(
    evidence_model,
    a2_model,
    loader,
    index,
    spec,
    selection,
    device,
    step_cap,
    split,
):
    evidence_model.eval()
    a2_model.eval()
    target_ids_all = []
    labels_all = []
    evidence_all = []
    a2_all = []
    for batch in loader:
        target_ids, labels, evidence, target_raw, _, _ = query_batch(
            index, batch, spec, selection, device
        )
        evidence_logits, _ = evidence_model(evidence, target_raw)
        batch.split = split
        batch.to(device)
        a2_logits, a2_labels = a2_model(batch)
        a2_logits = a2_logits.squeeze(-1)
        a2_labels = a2_labels.detach().cpu().long().view(-1)
        if not torch.equal(labels, a2_labels):
            raise AssertionError("A2 and evidence labels are not aligned")
        target_ids_all.append(target_ids)
        labels_all.append(labels)
        evidence_all.append(torch.sigmoid(evidence_logits).cpu())
        a2_all.append(torch.sigmoid(a2_logits).cpu())
        if step_cap and len(labels_all) >= step_cap:
            break
    return {
        "target_edge_ids": torch.cat(target_ids_all),
        "labels": torch.cat(labels_all),
        "evidence": torch.cat(evidence_all),
        "a2": torch.cat(a2_all),
    }


def correction_counts(labels, evidence_scores, a2_scores,
                      evidence_threshold, a2_threshold):
    labels = labels.bool()
    evidence_predictions = evidence_scores >= evidence_threshold
    a2_predictions = a2_scores >= a2_threshold
    evidence_correct = evidence_predictions == labels
    a2_correct = a2_predictions == labels
    corrected = (~a2_correct) & evidence_correct
    broken = a2_correct & (~evidence_correct)
    changed = evidence_predictions != a2_predictions
    a2_errors = int((~a2_correct).sum())
    a2_correct_count = int(a2_correct.sum())
    corrected_count = int(corrected.sum())
    broken_count = int(broken.sum())
    return {
        "samples": int(labels.numel()),
        "positives": int(labels.sum()),
        "evidence_test_f1": binary_f1(labels, evidence_predictions),
        "a2_same_batch_test_f1": binary_f1(labels, a2_predictions),
        "a2_errors": a2_errors,
        "changed_predictions": int(changed.sum()),
        "changed_rate": float(changed.float().mean()),
        "corrected_predictions": corrected_count,
        "broken_predictions": broken_count,
        "correction_rate_on_a2_errors": (
            corrected_count / a2_errors if a2_errors else 0.0
        ),
        "break_rate_on_a2_correct": (
            broken_count / a2_correct_count if a2_correct_count else 0.0
        ),
        "corrected_to_broken_ratio": (
            corrected_count / broken_count
            if broken_count
            else None
        ),
        "corrected_without_breaks": (
            corrected_count > 0 and broken_count == 0
        ),
        "both_correct": int((a2_correct & evidence_correct).sum()),
        "both_wrong": int(((~a2_correct) & (~evidence_correct)).sum()),
        "a2_only_correct": broken_count,
        "evidence_only_correct": corrected_count,
    }


def correction_diagnostic(
    evidence_model,
    a2_model,
    val_loader,
    test_loader,
    index,
    spec,
    selection,
    device,
    step_cap,
):
    val = paired_scores(
        evidence_model,
        a2_model,
        val_loader,
        index,
        spec,
        selection,
        device,
        step_cap,
        "val",
    )
    evidence_threshold, _ = best_f1_threshold(
        val["labels"], val["evidence"]
    )
    a2_threshold, _ = best_f1_threshold(val["labels"], val["a2"])
    test = paired_scores(
        evidence_model,
        a2_model,
        test_loader,
        index,
        spec,
        selection,
        device,
        step_cap,
        "test",
    )
    sampled = correction_counts(
        test["labels"],
        test["evidence"],
        test["a2"],
        evidence_threshold,
        a2_threshold,
    )
    unique_ids, unique_labels, unique_evidence = aggregate_unique(
        test["target_edge_ids"], test["labels"], test["evidence"]
    )
    unique_a2_ids, unique_a2_labels, unique_a2 = aggregate_unique(
        test["target_edge_ids"], test["labels"], test["a2"]
    )
    if (
        not torch.equal(unique_ids, unique_a2_ids)
        or not torch.equal(unique_labels, unique_a2_labels)
    ):
        raise AssertionError("unique paired A2 rows are not aligned")
    unique = correction_counts(
        unique_labels,
        unique_evidence,
        unique_a2,
        evidence_threshold,
        a2_threshold,
    )
    return {
        "reference": "terminal_frozen_a2_same_dynamic_batches",
        "a2_checkpoint_epoch": 499,
        "evidence_val_threshold": evidence_threshold,
        "a2_val_threshold": a2_threshold,
        "sampled_instances": sampled,
        "unique_edges": unique,
    }


def qualification_decision(selected, diagnostic, gate):
    normal = selected["test"]["normal"]["f1"]
    shuffled = selected["test"]["shuffled"]["f1"]
    off = selected["test"]["off"]["f1"]
    sampled = (
        diagnostic["sampled_instances"]
        if diagnostic is not None
        else None
    )
    evidence_diagnostics = selected["evidence_diagnostics"]
    class_coverage = {
        class_id: values["coverage_rate"]
        for class_id, values in evidence_diagnostics["class"].items()
    }
    changed_minimum = (
        max(
            int(gate["min_changed_predictions"]),
            float(gate["min_changed_fraction_of_a2_errors"])
            * sampled["a2_errors"],
        )
        if sampled is not None
        else None
    )
    checks = {
        "normal_minus_shuffled": normal - shuffled,
        "normal_minus_off": normal - off,
        "correction_rate_on_a2_errors": (
            sampled["correction_rate_on_a2_errors"]
            if sampled is not None
            else None
        ),
        "corrected_to_broken_ratio": (
            sampled["corrected_to_broken_ratio"]
            if sampled is not None
            else None
        ),
        "changed_predictions": (
            sampled["changed_predictions"]
            if sampled is not None
            else None
        ),
        "required_changed_predictions": changed_minimum,
        "corrected_without_breaks": (
            sampled["corrected_without_breaks"]
            if sampled is not None
            else False
        ),
        "overall_coverage": evidence_diagnostics["coverage_rate"],
        "class_coverage": class_coverage,
    }
    passed = (
        checks["normal_minus_shuffled"]
        >= float(gate["min_normal_minus_shuffled_test_f1"])
        and checks["normal_minus_off"]
        >= float(gate["min_normal_minus_off_test_f1"])
        and sampled is not None
        and checks["correction_rate_on_a2_errors"]
        >= float(gate["min_correction_rate_on_a2_errors"])
        and (
            checks["corrected_without_breaks"]
            or (
                checks["corrected_to_broken_ratio"] is not None
                and checks["corrected_to_broken_ratio"]
                >= float(gate["min_corrected_to_broken_ratio"])
            )
        )
        and checks["changed_predictions"] >= changed_minimum
        and checks["overall_coverage"]
        >= float(gate["min_overall_coverage"])
        and all(
            value is not None
            and value >= float(gate["min_each_class_coverage"])
            for value in checks["class_coverage"].values()
        )
    )
    return "pass" if passed else "fail", checks


def train_task(
    spec,
    task,
    relative_config,
    config_path,
    commit,
    args,
):
    seed = int(task["seed"])
    seed_process(seed)
    configure_fraudgt(config_path, seed, args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)

    full_store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=full_store.edge_index,
        timestamps=full_store.timestamps,
        raw_edge_attr=full_store.raw_edge_attr,
    )
    num_currencies = int(full_store.raw_edge_attr[:, 2].max()) + 1
    num_payment_formats = int(full_store.raw_edge_attr[:, 3].max()) + 1
    model = TransactionEvidenceEncoder(
        num_currencies=num_currencies,
        num_payment_formats=num_payment_formats,
        family=spec["family"],
        hidden_dim=int(spec["hidden_dim"]),
        num_heads=int(spec["num_heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    max_epochs = (
        int(args.max_epochs)
        if args.max_epochs is not None
        else int(spec["max_epochs"])
    )
    scheduler = cosine_schedule(
        optimizer,
        max_epochs=max_epochs,
        warmup_epochs=min(5, max_epochs),
    )

    variant = f"{spec['family']}_{task['selection']}"
    task_dir = (
        args.output_dir
        / f"{task['dataset']}_{variant}_seed{seed}_{commit}"
    )
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = task_dir / "trajectory.jsonl"
    checkpoint_path = task_dir / "best_val.ckpt"
    started = time.monotonic()
    events = []
    best_val = -1.0
    stale_evaluations = 0
    stopped_early = False

    for epoch in range(max_epochs):
        train = train_epoch(
            model,
            train_loader,
            index,
            optimizer,
            spec,
            task["selection"],
            device,
            args.train_step_cap,
        )
        scheduler.step()
        if (epoch + 1) % int(spec["eval_period"]) != 0:
            continue

        val = evaluate_evidence(
            model,
            val_loader,
            index,
            spec,
            task["selection"],
            device,
            args.eval_step_cap,
            ("normal",),
        )
        threshold, val_best_f1 = best_f1_threshold(
            val["labels"], val["scores"]["normal"]
        )
        test = evaluate_evidence(
            model,
            test_loader,
            index,
            spec,
            task["selection"],
            device,
            args.eval_step_cap,
            ("normal", "shuffled", "off"),
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train,
            "val": metric_row(
                val["labels"], val["scores"]["normal"], threshold
            ),
            "test": {
                name: metric_row(test["labels"], scores, threshold)
                for name, scores in test["scores"].items()
            },
            "test_unique": {},
            "evidence_diagnostics": test["evidence_diagnostics"],
            "sampling_protocol": "dynamic_random",
        }
        for name, scores in test["scores"].items():
            _, unique_labels, unique_scores = aggregate_unique(
                test["target_edge_ids"], test["labels"], scores
            )
            row["test_unique"][name] = metric_row(
                unique_labels, unique_scores, threshold
            )
        row["val"]["f1"] = val_best_f1
        append_jsonl(trajectory_path, row)
        events.append(row)

        previous_best = best_val
        if val_best_f1 > previous_best:
            best_val = val_best_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_threshold": threshold,
                    "val_f1": val_best_f1,
                    "git_commit": commit,
                    "sampling_protocol": "dynamic_random",
                },
                checkpoint_path,
            )
        if (
            val_best_f1
            > previous_best + float(spec["early_stop_delta"])
        ):
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if (
            epoch + 1 >= int(spec["min_epochs"])
            and stale_evaluations
            >= int(spec["early_stop_evaluations"])
        ):
            stopped_early = True
            break

    if not events:
        raise RuntimeError("no evaluation event was produced")
    selected = max(events, key=lambda row: row["val"]["f1"])
    raw_best = max(events, key=lambda row: row["test"]["normal"]["f1"])
    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state"], strict=True)

    diagnostic = None
    if not args.skip_a2_diagnostic:
        a2_model = load_a2_model(dataset, task["a2_checkpoint"], device)
        diagnostic = correction_diagnostic(
            model,
            a2_model,
            val_loader,
            test_loader,
            index,
            spec,
            task["selection"],
            device,
            args.eval_step_cap,
        )
        del a2_model
    decision, checks = qualification_decision(
        selected,
        diagnostic,
        spec["qualification_gate"],
    )

    baseline = INITIAL_A2[task["dataset"]]
    spec_baseline = spec["initial_a2"][task["dataset"]]
    if baseline != spec_baseline:
        raise AssertionError("spec baseline differs from canonical initial A2")
    val_selected_test_f1 = selected["test"]["normal"]["f1"]
    raw_best_test_f1 = raw_best["test"]["normal"]["f1"]
    manifest = {
        "dataset": task["dataset"],
        "model": spec["model"],
        "variant": variant,
        "evidence_family": spec["family"],
        "evidence_selection": task["selection"],
        "seed": seed,
        "git_commit": commit,
        "config": relative_config,
        "checkpoint": str(checkpoint_path),
        "a2_diagnostic_checkpoint": task["a2_checkpoint"],
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "val_selected_epoch": selected["epoch"],
        "val_selected_val_f1": selected["val"]["f1"],
        "val_selected_test_f1": val_selected_test_f1,
        "initial_a2_val_selected_test_f1": baseline[
            "val_selected_test_f1"
        ],
        "delta_val_selected_f1": (
            val_selected_test_f1 - baseline["val_selected_test_f1"]
        ),
        "raw_best_epoch": raw_best["epoch"],
        "raw_best_test_f1": raw_best_test_f1,
        "initial_a2_raw_best_test_f1": baseline["raw_best_test_f1"],
        "delta_raw_best_f1": (
            raw_best_test_f1 - baseline["raw_best_test_f1"]
        ),
        "sampling_variation_note": (
            "possible_sampling_variation"
            if abs(
                val_selected_test_f1
                - baseline["val_selected_test_f1"]
            )
            < 0.005
            else None
        ),
        "selected_event": selected,
        "raw_best_event": raw_best,
        "same_batch_a2_diagnostic": diagnostic,
        "qualification_checks": checks,
        "qualification_decision": decision,
        "max_epochs_requested": max_epochs,
        "epochs_completed": int(events[-1]["epoch"]) + 1,
        "stopped_early": stopped_early,
        "train_step_cap": args.train_step_cap or None,
        "eval_step_cap": args.eval_step_cap or None,
        "elapsed_seconds": time.monotonic() - started,
    }
    (task_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "run/tier_phase1_spec.json",
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--train-step-cap", type=int, default=0)
    parser.add_argument("--eval-step-cap", type=int, default=0)
    parser.add_argument("--skip-a2-diagnostic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    if not 0 <= args.task_index < len(spec["tasks"]):
        raise IndexError("task-index is outside the spec task list")
    task = spec["tasks"][args.task_index]
    relative_config, config_path, config = load_task_config(spec, task)
    audit_protocol(spec, task, config)
    commit = verify_repository(spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = train_task(
        spec,
        task,
        relative_config,
        config_path,
        commit,
        args,
    )
    print(json.dumps({
        key: manifest[key]
        for key in (
            "dataset",
            "variant",
            "seed",
            "val_selected_test_f1",
            "delta_val_selected_f1",
            "raw_best_test_f1",
            "delta_raw_best_f1",
            "qualification_decision",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
