#!/usr/bin/env python3
"""Train fold-excluded A2 and evidence teachers and emit OOF scores."""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml

from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.evidence.tier_model import TransactionEvidenceEncoder
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from fraudGT.graphgym.model_builder import create_model
from run.tier_phase1_evidence_qualification import (
    TASK,
    audit_protocol,
    best_f1_threshold,
    configure_fraudgt,
    cosine_schedule,
    loader_audit,
    query_batch,
    seed_process,
    select_values_by_edge_id,
    verify_repository,
)


def fold_ids(edge_ids, seed, num_folds):
    return (
        edge_ids.long() * 1103515247 + int(seed) * 12347
    ).remainder(int(num_folds))


def fold_mask(edge_ids, seed, num_folds, fold, held_out):
    assigned = fold_ids(edge_ids, seed, num_folds)
    return assigned == int(fold) if held_out else assigned != int(fold)


def weighted_binary_loss(logits, labels, positive_weight):
    labels = labels.float().view(-1)
    logits = logits.view(-1)
    weights = torch.where(
        labels > 0,
        torch.full_like(labels, float(positive_weight)),
        torch.ones_like(labels),
    )
    return F.binary_cross_entropy_with_logits(
        logits, labels, weight=weights)


def a2_output_edge_ids(model, batch):
    head = model.post_gt
    if not hasattr(head, "_edge_mask"):
        raise TypeError("A2 head does not expose edge alignment mask")
    mask = head._edge_mask(batch)
    return batch[TASK].e_id[mask]


def capped(loader, step_cap):
    for step, batch in enumerate(loader):
        if step_cap and step >= step_cap:
            break
        yield batch


def finish_accumulation(model, optimizer, pending):
    if pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def train_a2(
    model,
    loader,
    optimizer,
    scheduler,
    task,
    spec,
    device,
    max_epochs,
    step_cap,
    trajectory,
):
    accumulation = int(cfg.optim.batch_accumulation)
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        loss_total = 0.0
        samples = 0
        held_out_seen = 0
        for batch in capped(loader, step_cap):
            batch.split = "train"
            batch.to(device)
            output_ids = a2_output_edge_ids(model, batch)
            pred, labels = model(batch)
            keep = fold_mask(
                output_ids,
                task["seed"],
                spec["num_folds"],
                task["fold"],
                held_out=False,
            )
            held_out_seen += int((~keep).sum())
            if not keep.any():
                continue
            loss = weighted_binary_loss(
                pred[keep],
                labels[keep],
                spec["positive_weight"],
            )
            (loss / accumulation).backward()
            pending += 1
            if pending == accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            count = int(keep.sum())
            loss_total += float(loss.detach()) * count
            samples += count
        finish_accumulation(model, optimizer, pending)
        scheduler.step()
        trajectory.append({
            "stage": "a2",
            "epoch": epoch,
            "loss": loss_total / max(samples, 1),
            "supervised_samples": samples,
            "held_out_outputs_seen": held_out_seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })


def train_evidence(
    model,
    loader,
    index,
    optimizer,
    scheduler,
    task,
    spec,
    device,
    max_epochs,
    step_cap,
    trajectory,
):
    accumulation = int(cfg.optim.batch_accumulation)
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        loss_total = 0.0
        samples = 0
        held_out_seen = 0
        for batch in capped(loader, step_cap):
            target_ids, labels, evidence, target_raw, _, _ = query_batch(
                index, batch, spec, spec["selection"], device)
            keep = fold_mask(
                target_ids,
                task["seed"],
                spec["num_folds"],
                task["fold"],
                held_out=False,
            )
            held_out_seen += int((~keep).sum())
            if not keep.any():
                continue
            keep_device = keep.to(device)
            logits, _ = model(evidence, target_raw)
            loss = weighted_binary_loss(
                logits[keep_device],
                labels.to(device)[keep_device],
                spec["positive_weight"],
            )
            (loss / accumulation).backward()
            pending += 1
            if pending == accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            count = int(keep.sum())
            loss_total += float(loss.detach()) * count
            samples += count
        finish_accumulation(model, optimizer, pending)
        scheduler.step()
        trajectory.append({
            "stage": "evidence",
            "epoch": epoch,
            "loss": loss_total / max(samples, 1),
            "supervised_samples": samples,
            "held_out_outputs_seen": held_out_seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })


def append_parts(parts, key, value):
    parts.setdefault(key, []).append(value.detach().cpu())


@torch.no_grad()
def collect_fold_scores(
    a2_model,
    evidence_model,
    loader,
    index,
    task,
    spec,
    device,
    step_cap,
):
    a2_model.eval()
    evidence_model.eval()
    parts = {}
    requested = 0
    paired = 0
    for batch in capped(loader, step_cap):
        (
            target_ids,
            labels,
            evidence,
            target_raw,
            _,
            evidence_cpu,
        ) = query_batch(index, batch, spec, spec["selection"], device)
        normal_logits, _ = evidence_model(evidence, target_raw)
        permutation = torch.randperm(evidence.tokens.size(0), device=device)
        shuffled_logits, _ = evidence_model(
            evidence.shuffled(permutation), target_raw)
        off_logits, _ = evidence_model(evidence.off(), target_raw)

        batch.split = "train"
        batch.to(device)
        output_ids = a2_output_edge_ids(a2_model, batch)
        a2_logits, a2_labels = a2_model(batch)
        (
            normal_logits,
            shuffled_logits,
            off_logits,
            labels,
            support,
        ) = select_values_by_edge_id(
            output_ids.detach().cpu(),
            target_ids,
            normal_logits.detach().cpu(),
            shuffled_logits.detach().cpu(),
            off_logits.detach().cpu(),
            labels,
            evidence_cpu.support[:, 0],
        )
        a2_labels = a2_labels.detach().cpu().long().view(-1)
        if not torch.equal(labels, a2_labels):
            raise AssertionError("A2/evidence fold labels differ")
        held_out = fold_mask(
            output_ids.detach().cpu(),
            task["seed"],
            spec["num_folds"],
            task["fold"],
            held_out=True,
        )
        append_parts(
            parts, "edge_ids", output_ids.detach().cpu()[held_out])
        append_parts(parts, "labels", labels[held_out])
        append_parts(
            parts,
            "a2_scores",
            torch.sigmoid(a2_logits.detach().cpu().view(-1))[held_out],
        )
        append_parts(
            parts,
            "evidence_scores",
            torch.sigmoid(normal_logits.view(-1))[held_out],
        )
        append_parts(
            parts,
            "shuffled_scores",
            torch.sigmoid(shuffled_logits.view(-1))[held_out],
        )
        append_parts(
            parts,
            "off_scores",
            torch.sigmoid(off_logits.view(-1))[held_out],
        )
        append_parts(parts, "support_count", support[held_out])
        requested += int(target_ids.numel())
        paired += int(output_ids.numel())
    output = {key: torch.cat(values) for key, values in parts.items()}
    output["requested_samples"] = requested
    output["paired_samples"] = paired
    output["paired_retention_rate"] = paired / max(requested, 1)
    return output


def aggregate_unique(values):
    edge_ids = values["edge_ids"].long()
    unique_ids, inverse = torch.unique(
        edge_ids, sorted=True, return_inverse=True)
    counts = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    counts.scatter_add_(0, inverse, torch.ones_like(edge_ids).float())
    output = {"edge_ids": unique_ids}
    label_sums = torch.zeros(unique_ids.numel(), dtype=torch.long)
    label_sums.scatter_add_(0, inverse, values["labels"].long())
    if not torch.equal(
        label_sums, (label_sums > 0).long() * counts.long()
    ):
        raise AssertionError("OOF edge has inconsistent labels")
    output["labels"] = (label_sums > 0).long()
    for key in (
        "a2_scores",
        "evidence_scores",
        "shuffled_scores",
        "off_scores",
        "support_count",
    ):
        sums = torch.zeros(unique_ids.numel(), dtype=torch.float32)
        sums.scatter_add_(0, inverse, values[key].float())
        output[key] = sums / counts
    return output


def output_path(root, task, commit):
    return root / (
        f"{task['dataset']}_fold{task['fold']}_"
        f"seed{task['seed']}_{commit}"
    )


def run_task(spec, task, args):
    commit = verify_repository(spec)
    task_dir = output_path(args.output_dir, task, commit)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    config_relative = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    seed_process(int(task["seed"]) + int(task["fold"]) * 1009)
    configure_fraudgt(config_path, task["seed"], args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    train_loader = loaders[0]
    device = torch.device(args.device)
    store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=store.edge_index,
        timestamps=store.timestamps,
        raw_edge_attr=store.raw_edge_attr,
    )
    a2_model = create_model(dataset=dataset).to(device)
    evidence_model = TransactionEvidenceEncoder(
        num_currencies=int(store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(store.raw_edge_attr[:, 3].max()) + 1,
        family=spec["family"],
        hidden_dim=int(spec["hidden_dim"]),
        num_heads=int(spec["num_heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    a2_epochs = (
        args.a2_max_epochs
        if args.a2_max_epochs is not None
        else int(spec["a2_max_epochs"])
    )
    evidence_epochs = (
        args.evidence_max_epochs
        if args.evidence_max_epochs is not None
        else int(spec["evidence_max_epochs"])
    )
    a2_optimizer = torch.optim.AdamW(
        a2_model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    evidence_optimizer = torch.optim.AdamW(
        evidence_model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    a2_scheduler = cosine_schedule(
        a2_optimizer, a2_epochs, min(spec["warmup_epochs"], a2_epochs))
    evidence_scheduler = cosine_schedule(
        evidence_optimizer,
        evidence_epochs,
        min(spec["warmup_epochs"], evidence_epochs),
    )
    trajectory = []
    started = time.monotonic()
    train_a2(
        a2_model,
        train_loader,
        a2_optimizer,
        a2_scheduler,
        task,
        spec,
        device,
        a2_epochs,
        args.train_step_cap,
        trajectory,
    )
    del a2_optimizer, a2_scheduler
    torch.cuda.empty_cache()
    train_evidence(
        evidence_model,
        train_loader,
        index,
        evidence_optimizer,
        evidence_scheduler,
        task,
        spec,
        device,
        evidence_epochs,
        args.train_step_cap,
        trajectory,
    )
    del evidence_optimizer, evidence_scheduler
    torch.cuda.empty_cache()
    sampled = collect_fold_scores(
        a2_model,
        evidence_model,
        train_loader,
        index,
        task,
        spec,
        device,
        args.collect_step_cap,
    )
    unique = aggregate_unique(sampled)
    if not unique["labels"].numel():
        raise RuntimeError("fold produced no unique held-out OOF edges")

    # Thresholds are fit only on non-held-out outputs from a separate dynamic
    # pass. Reuse collection code by changing the fold predicate through a
    # temporary task copy and taking the complement is intentionally avoided:
    # held-out labels must never influence these thresholds.
    threshold_sample = collect_fit_threshold_scores(
        a2_model,
        evidence_model,
        train_loader,
        index,
        task,
        spec,
        device,
        args.collect_step_cap,
    )
    a2_threshold, _ = best_f1_threshold(
        threshold_sample["labels"], threshold_sample["a2_scores"])
    evidence_threshold, _ = best_f1_threshold(
        threshold_sample["labels"],
        threshold_sample["evidence_scores"],
    )
    oof_path = task_dir / "oof_scores.pt"
    torch.save({
        **unique,
        "fold": int(task["fold"]),
        "num_folds": int(spec["num_folds"]),
        "dataset": task["dataset"],
        "seed": int(task["seed"]),
        "a2_threshold": a2_threshold,
        "evidence_threshold": evidence_threshold,
        "sampling_protocol": "dynamic_random",
        "git_commit": commit,
    }, oof_path)
    a2_checkpoint = task_dir / "a2_teacher.ckpt"
    evidence_checkpoint = task_dir / "evidence_teacher.ckpt"
    torch.save({"model_state": a2_model.state_dict()}, a2_checkpoint)
    torch.save(
        {"model_state": evidence_model.state_dict()},
        evidence_checkpoint,
    )
    trajectory_path = task_dir / "training_trajectory.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in trajectory)
    )
    manifest = {
        "dataset": task["dataset"],
        "model": spec["model"],
        "variant": "flow_role_recent_crossfit_teacher",
        "seed": int(task["seed"]),
        "fold": int(task["fold"]),
        "num_folds": int(spec["num_folds"]),
        "git_commit": commit,
        "config": config_relative,
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "supervision_scope": "train_non_held_out_target_edges_only",
        "held_out_edges_remain_graph_context": True,
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "a2_epochs": a2_epochs,
        "evidence_epochs": evidence_epochs,
        "train_step_cap": args.train_step_cap or None,
        "collect_step_cap": args.collect_step_cap or None,
        "a2_checkpoint": str(a2_checkpoint),
        "evidence_checkpoint": str(evidence_checkpoint),
        "oof_scores": str(oof_path),
        "a2_fit_threshold": a2_threshold,
        "evidence_fit_threshold": evidence_threshold,
        "requested_samples": sampled["requested_samples"],
        "paired_samples": sampled["paired_samples"],
        "paired_retention_rate": sampled["paired_retention_rate"],
        "oof_unique_edges": int(unique["labels"].numel()),
        "oof_positives": int(unique["labels"].sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (task_dir / "phase2b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


@torch.no_grad()
def collect_fit_threshold_scores(
    a2_model,
    evidence_model,
    loader,
    index,
    task,
    spec,
    device,
    step_cap,
):
    a2_model.eval()
    evidence_model.eval()
    labels_all = []
    a2_all = []
    evidence_all = []
    for batch in capped(loader, step_cap):
        target_ids, labels, evidence, target_raw, _, _ = query_batch(
            index, batch, spec, spec["selection"], device)
        evidence_logits, _ = evidence_model(evidence, target_raw)
        batch.split = "train"
        batch.to(device)
        output_ids = a2_output_edge_ids(a2_model, batch)
        a2_logits, a2_labels = a2_model(batch)
        evidence_logits, labels = select_values_by_edge_id(
            output_ids.detach().cpu(),
            target_ids,
            evidence_logits.detach().cpu(),
            labels,
        )
        a2_labels = a2_labels.detach().cpu().long().view(-1)
        if not torch.equal(labels, a2_labels):
            raise AssertionError("fit threshold labels differ")
        fit = fold_mask(
            output_ids.detach().cpu(),
            task["seed"],
            spec["num_folds"],
            task["fold"],
            held_out=False,
        )
        labels_all.append(labels[fit])
        a2_all.append(
            torch.sigmoid(a2_logits.detach().cpu().view(-1))[fit])
        evidence_all.append(
            torch.sigmoid(evidence_logits.view(-1))[fit])
    return {
        "labels": torch.cat(labels_all),
        "a2_scores": torch.cat(a2_all),
        "evidence_scores": torch.cat(evidence_all),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "run/tier_phase2b_crossfit_spec.json",
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--a2-max-epochs", type=int)
    parser.add_argument("--evidence-max-epochs", type=int)
    parser.add_argument("--train-step-cap", type=int, default=0)
    parser.add_argument("--collect-step-cap", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    if not 0 <= args.task_index < len(spec["tasks"]):
        raise IndexError("task-index is outside the spec task list")
    manifest = run_task(spec, spec["tasks"][args.task_index], args)
    print(json.dumps({
        key: manifest[key]
        for key in (
            "dataset",
            "seed",
            "fold",
            "a2_epochs",
            "evidence_epochs",
            "oof_unique_edges",
            "oof_positives",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
