#!/usr/bin/env python3
"""Run the CET-FraudGT two-scale representation-fusion screen."""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml

import fraudGT  # noqa: F401 - register GraphGym components
from fraudGT.evidence.cet_model import CETFusionClassifier
from fraudGT.evidence.tier import EvidenceBatch, TemporalIncidentIndex
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.tier_phase1_evidence_qualification import (
    INITIAL_A2,
    TASK,
    aggregate_unique,
    audit_protocol,
    best_f1_threshold,
    binary_f1,
    configure_fraudgt,
    cosine_schedule,
    loader_audit,
    load_a2_model,
    metric_row,
    query_batch,
    seed_process,
    tensor_distribution,
    verify_repository,
)


CONDITIONS = ("normal", "shuffled", "off")


@dataclass(frozen=True)
class OOFTeacher:
    edge_ids: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    thresholds: torch.Tensor

    @classmethod
    def load(cls, root, dataset):
        paths = sorted(Path(root).glob(
            f"{dataset}_fold*/oof_scores.pt"))
        if len(paths) != 3:
            raise RuntimeError(
                f"expected three OOF folds for {dataset}, found {len(paths)}")
        rows = []
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            if payload.get("dataset") != dataset:
                raise ValueError(f"{path}: dataset differs")
            if payload.get("sampling_protocol") != "dynamic_random":
                raise ValueError(f"{path}: protocol differs")
            count = int(payload["edge_ids"].numel())
            rows.append({
                "edge_ids": payload["edge_ids"].long(),
                "labels": payload["labels"].long(),
                "scores": payload["a2_scores"].float(),
                "thresholds": torch.full(
                    (count,), float(payload["a2_threshold"])),
            })
        edge_ids = torch.cat([row["edge_ids"] for row in rows])
        if torch.unique(edge_ids).numel() != edge_ids.numel():
            raise ValueError("OOF teacher edge IDs overlap across folds")
        order = torch.argsort(edge_ids)
        return cls(
            edge_ids=edge_ids[order],
            labels=torch.cat([row["labels"] for row in rows])[order],
            scores=torch.cat([row["scores"] for row in rows])[order],
            thresholds=torch.cat(
                [row["thresholds"] for row in rows])[order],
        )

    def lookup(self, edge_ids, labels):
        edge_ids = edge_ids.detach().cpu().long().view(-1)
        labels = labels.detach().cpu().long().view(-1)
        positions = torch.searchsorted(self.edge_ids, edge_ids)
        safe = positions.clamp_max(max(self.edge_ids.numel() - 1, 0))
        available = (
            (positions < self.edge_ids.numel())
            & (self.edge_ids[safe] == edge_ids)
        )
        scores = torch.full((edge_ids.numel(),), 0.5)
        thresholds = torch.full((edge_ids.numel(),), 0.5)
        teacher_labels = torch.full_like(labels, -1)
        scores[available] = self.scores[safe[available]]
        thresholds[available] = self.thresholds[safe[available]]
        teacher_labels[available] = self.labels[safe[available]]
        if not torch.equal(
            teacher_labels[available], labels[available]
        ):
            raise AssertionError("OOF and current train labels differ")
        predictions = scores >= thresholds
        errors = available & (predictions != labels.bool())
        return {
            "available": available,
            "scores": scores,
            "thresholds": thresholds,
            "errors": errors,
        }


def audit_cet_protocol(spec, task, config):
    audit_protocol(spec, task, config)
    if task["variant"] not in CETFusionClassifier.VARIANTS:
        raise ValueError("unregistered CET variant")
    source = Path(__file__).read_text()
    forbidden = (
        "torch." + "Generator(",
        "get_" + "rng_state(",
        "set_" + "rng_state(",
        "fixed_target_panel" + "=True",
    )
    if any(token in source for token in forbidden):
        raise RuntimeError("fixed evaluation randomness detected")


def alignment_positions(requested_ids, source_ids):
    requested_ids = requested_ids.detach().cpu().long().view(-1)
    source_ids = source_ids.detach().cpu().long().view(-1)
    if (
        torch.unique(requested_ids).numel() != requested_ids.numel()
        or torch.unique(source_ids).numel() != source_ids.numel()
    ):
        raise AssertionError("batch edge IDs must be unique")
    source_order = torch.argsort(source_ids)
    sorted_source = source_ids[source_order]
    positions = torch.searchsorted(sorted_source, requested_ids)
    safe = positions.clamp_max(max(sorted_source.numel() - 1, 0))
    if (
        not sorted_source.numel()
        or (positions >= sorted_source.numel()).any()
        or not torch.equal(sorted_source[safe], requested_ids)
    ):
        raise AssertionError("FraudGT target is absent from evidence batch")
    return source_order[positions]


def select_evidence(evidence, positions):
    device_positions = positions.to(evidence.tokens.device)
    return EvidenceBatch(
        tokens=evidence.tokens[device_positions],
        mask=evidence.mask[device_positions],
        context_edge_ids=evidence.context_edge_ids[device_positions],
        support=evidence.support[device_positions],
    )


@torch.no_grad()
def frozen_base_batch(a2_model, batch, split):
    a2_model.eval()
    batch.split = split
    encoded = a2_model.encode_batch(batch)
    head = a2_model.post_gt
    mask = head._edge_mask(encoded)
    output_ids = encoded[TASK].e_id[mask]
    base_features, labels = head._apply_index(encoded)
    base_logits, output_labels = head(encoded)
    base_logits = base_logits.view(-1)
    labels = labels.long().view(-1)
    output_labels = output_labels.long().view(-1)
    if (
        output_ids.numel() != labels.numel()
        or not torch.equal(labels, output_labels)
    ):
        raise AssertionError("FraudGT base outputs are not edge aligned")
    return output_ids, labels, base_features.detach(), base_logits.detach()


def prepare_batch(
    a2_model,
    batch,
    index,
    spec,
    task,
    device,
    split,
):
    (
        target_ids,
        query_labels,
        evidence,
        target_raw,
        query_seconds,
        evidence_cpu,
    ) = query_batch(index, batch, spec, task["selection"], device)
    batch.to(device)
    output_ids, labels, base_features, base_logits = frozen_base_batch(
        a2_model, batch, split)
    positions = alignment_positions(
        output_ids.detach().cpu(), target_ids)
    query_labels = query_labels[positions]
    if not torch.equal(query_labels, labels.detach().cpu()):
        raise AssertionError("FraudGT and evidence labels differ")
    return {
        "edge_ids": output_ids.detach().cpu(),
        "labels": labels,
        "base_features": base_features,
        "base_logits": base_logits,
        "evidence": select_evidence(evidence, positions),
        "target_raw": target_raw[positions.to(device)],
        "support": evidence_cpu.support[positions],
        "query_seconds": query_seconds,
        "requested": int(target_ids.numel()),
    }


def class_weighted_bce(logits, labels, positive_weight, extra=None):
    labels = labels.float().view(-1)
    weights = torch.where(
        labels > 0,
        torch.full_like(labels, float(positive_weight)),
        torch.ones_like(labels),
    )
    if extra is not None:
        weights = weights * extra.float().view(-1)
    return F.binary_cross_entropy_with_logits(
        logits.view(-1), labels, weight=weights)


def counterfactual_ranking_loss(
    normal_logits,
    shuffled_logits,
    off_logits,
    labels,
    margin,
):
    labels = labels.float().view(-1)
    normal = F.binary_cross_entropy_with_logits(
        normal_logits, labels, reduction="none")
    shuffled = F.binary_cross_entropy_with_logits(
        shuffled_logits, labels, reduction="none")
    off = F.binary_cross_entropy_with_logits(
        off_logits, labels, reduction="none")
    counterfactual = 0.5 * (shuffled + off)
    return F.relu(float(margin) + normal - counterfactual).mean()


def train_epoch(
    model,
    a2_model,
    loader,
    index,
    teacher,
    optimizer,
    spec,
    task,
    device,
    step_cap,
):
    started = time.monotonic()
    model.train()
    a2_model.eval()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(cfg.optim.batch_accumulation)
    totals = {
        "loss": 0.0,
        "fraud_loss": 0.0,
        "counterfactual_loss": 0.0,
        "complementarity_loss": 0.0,
        "dropout_loss": 0.0,
        "distillation_loss": 0.0,
        "auxiliary_loss": 0.0,
    }
    samples = 0
    paired = 0
    oof_available = 0
    oof_errors = 0
    pending = 0
    query_seconds = 0.0
    for step, batch in enumerate(loader):
        prepared = prepare_batch(
            a2_model, batch, index, spec, task, device, "train")
        labels = prepared["labels"].float()
        evidence = prepared["evidence"]
        base_features = prepared["base_features"]
        target_raw = prepared["target_raw"]
        normal_logits, normal_diagnostics = model(
            base_features, evidence, target_raw)
        permutation = torch.randperm(
            evidence.tokens.size(0), device=device)
        shuffled_logits, _ = model(
            base_features, evidence.shuffled(permutation), target_raw)
        off_logits, _ = model(
            base_features, evidence.off(), target_raw)

        oof = teacher.lookup(
            prepared["edge_ids"], labels.detach().cpu().long())
        oof_error = oof["errors"].to(device)
        oof_available += int(oof["available"].sum())
        oof_errors += int(oof_error.sum())
        fraud_loss = class_weighted_bce(
            normal_logits,
            labels,
            spec["positive_weight"],
        )
        counterfactual_loss = counterfactual_ranking_loss(
            normal_logits,
            shuffled_logits,
            off_logits,
            labels,
            spec["counterfactual_margin"],
        )
        if oof_error.any():
            complementarity_loss = class_weighted_bce(
                normal_logits[oof_error],
                labels[oof_error],
                spec["positive_weight"],
            )
        else:
            complementarity_loss = normal_logits.sum() * 0.0
        dropout_loss = class_weighted_bce(
            off_logits,
            labels,
            spec["positive_weight"],
        )
        if task["variant"] == "fusion":
            base_probability = torch.sigmoid(prepared["base_logits"])
            distillation_loss = F.binary_cross_entropy_with_logits(
                off_logits, base_probability)
            auxiliary_loss = class_weighted_bce(
                normal_diagnostics["evidence_logits"],
                labels,
                spec["positive_weight"],
            )
        else:
            distillation_loss = normal_logits.sum() * 0.0
            auxiliary_loss = normal_logits.sum() * 0.0
            dropout_loss = normal_logits.sum() * 0.0
        loss = (
            fraud_loss
            + float(spec["lambda_counterfactual"])
            * counterfactual_loss
            + float(spec["lambda_complementarity"])
            * complementarity_loss
            + float(spec["lambda_dropout"]) * dropout_loss
            + float(spec["lambda_distillation"]) * distillation_loss
            + float(spec["lambda_auxiliary"]) * auxiliary_loss
        )
        (loss / accumulation).backward()
        pending += 1
        if pending == accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        count = int(labels.numel())
        values = {
            "loss": loss,
            "fraud_loss": fraud_loss,
            "counterfactual_loss": counterfactual_loss,
            "complementarity_loss": complementarity_loss,
            "dropout_loss": dropout_loss,
            "distillation_loss": distillation_loss,
            "auxiliary_loss": auxiliary_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach()) * count
        samples += count
        paired += int(prepared["edge_ids"].numel())
        query_seconds += prepared["query_seconds"]
        if step_cap and step + 1 >= step_cap:
            break
    if pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        **{
            key: value / max(samples, 1)
            for key, value in totals.items()
        },
        "samples": samples,
        "paired_samples": paired,
        "oof_available": oof_available,
        "oof_errors": oof_errors,
        "query_seconds": query_seconds,
        "elapsed_seconds": time.monotonic() - started,
    }


def append_parts(parts, key, values):
    parts.setdefault(key, []).append(values.detach().cpu())


@torch.no_grad()
def evaluate(
    model,
    a2_model,
    loader,
    index,
    spec,
    task,
    device,
    split,
    step_cap,
    conditions,
):
    started = time.monotonic()
    model.eval()
    a2_model.eval()
    parts = {}
    diagnostics = {
        "history_norm": [],
        "base_norm": [],
        "evidence_norm": [],
        "fusion_norm": [],
        "fusion_gain_norm": [],
        "support_count": [],
        "has_history": [],
    }
    requested = 0
    query_seconds = 0.0
    steps = 0
    for step, batch in enumerate(loader):
        prepared = prepare_batch(
            a2_model, batch, index, spec, task, device, split)
        evidence = prepared["evidence"]
        base_features = prepared["base_features"]
        target_raw = prepared["target_raw"]
        append_parts(parts, "edge_ids", prepared["edge_ids"])
        append_parts(parts, "labels", prepared["labels"])
        append_parts(
            parts, "base", torch.sigmoid(prepared["base_logits"]))
        if "normal" in conditions:
            logits, current = model(
                base_features, evidence, target_raw)
            append_parts(parts, "normal", torch.sigmoid(logits))
            for key in diagnostics:
                diagnostics[key].append(
                    current[key].detach().cpu().float())
        if "shuffled" in conditions:
            permutation = torch.randperm(
                evidence.tokens.size(0), device=device)
            logits, _ = model(
                base_features, evidence.shuffled(permutation), target_raw)
            append_parts(parts, "shuffled", torch.sigmoid(logits))
        if "off" in conditions:
            logits, _ = model(
                base_features, evidence.off(), target_raw)
            append_parts(parts, "off", torch.sigmoid(logits))
        requested += prepared["requested"]
        query_seconds += prepared["query_seconds"]
        steps += 1
        if step_cap and step + 1 >= step_cap:
            break
    output = {
        key: torch.cat(values) for key, values in parts.items()
    }
    output["requested_samples"] = requested
    output["paired_samples"] = int(output["labels"].numel())
    output["paired_retention_rate"] = (
        output["paired_samples"] / max(requested, 1))
    output["query_seconds"] = query_seconds
    output["steps"] = steps
    output["elapsed_seconds"] = time.monotonic() - started
    if diagnostics["has_history"]:
        merged = {
            key: torch.cat(values)
            for key, values in diagnostics.items()
        }
        output["diagnostics"] = {
            "coverage_rate": float(
                merged["has_history"].mean()),
            "history_norm": tensor_distribution(
                merged["history_norm"]),
            "base_norm": tensor_distribution(merged["base_norm"]),
            "evidence_norm": tensor_distribution(
                merged["evidence_norm"]),
            "fusion_norm": tensor_distribution(merged["fusion_norm"]),
            "fusion_gain_norm": tensor_distribution(
                merged["fusion_gain_norm"]),
            "support_count": tensor_distribution(
                merged["support_count"]),
        }
    return output


def intervention_statistics(labels, base_scores, model_scores,
                            base_threshold, model_threshold):
    labels = labels.bool()
    base = base_scores >= float(base_threshold)
    model = model_scores >= float(model_threshold)
    changed = base != model
    base_correct = base == labels
    model_correct = model == labels
    corrected = changed & (~base_correct) & model_correct
    broken = changed & base_correct & (~model_correct)
    corrected_count = int(corrected.sum())
    broken_count = int(broken.sum())
    return {
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": corrected_count,
        "broken_predictions": broken_count,
        "corrected_minus_broken": corrected_count - broken_count,
        "corrected_to_broken_ratio": (
            corrected_count / broken_count if broken_count else None
        ),
        "corrected_without_breaks": (
            corrected_count > 0 and broken_count == 0
        ),
        "base_f1": binary_f1(labels, base),
        "model_f1": binary_f1(labels, model),
    }


def evaluation_event(
    model,
    a2_model,
    loaders,
    index,
    spec,
    task,
    device,
    epoch,
    train,
    optimizer,
    eval_step_cap,
):
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
    threshold, val_f1 = best_f1_threshold(
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
    row = {
        "epoch": epoch,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "sampling_protocol": "dynamic_random",
        "train": train,
        "val": metric_row(
            val["labels"], val["normal"], threshold),
        "base_val": metric_row(
            val["labels"], val["base"], base_threshold),
        "test": {
            condition: metric_row(
                test["labels"], test[condition], threshold)
            for condition in CONDITIONS
        },
        "base_test": metric_row(
            test["labels"], test["base"], base_threshold),
        "test_unique": {},
        "diagnostics": test["diagnostics"],
        "intervention": intervention_statistics(
            test["labels"],
            test["base"],
            test["normal"],
            base_threshold,
            threshold,
        ),
        "val_threshold": threshold,
        "base_val_threshold": base_threshold,
        "val_paired_retention_rate": val["paired_retention_rate"],
        "test_paired_retention_rate": test["paired_retention_rate"],
        "val_query_seconds": val["query_seconds"],
        "test_query_seconds": test["query_seconds"],
        "val_elapsed_seconds": val["elapsed_seconds"],
        "test_elapsed_seconds": test["elapsed_seconds"],
        "validation_loader_iterations": val["steps"],
        "test_loader_iterations": test["steps"],
    }
    row["val"]["f1"] = val_f1
    row["base_val"]["f1"] = base_val_f1
    for condition in CONDITIONS:
        _, unique_labels, unique_scores = aggregate_unique(
            test["edge_ids"], test["labels"], test[condition])
        row["test_unique"][condition] = metric_row(
            unique_labels, unique_scores, threshold)
    return row


def write_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def train_task(spec, task, args):
    commit = verify_repository(spec)
    config_relative = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_cet_protocol(spec, task, config)
    seed_process(int(task["seed"]))
    configure_fraudgt(config_path, task["seed"], args.device)
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
    teacher = OOFTeacher.load(
        args.oof_root, task["dataset"])
    a2_model = load_a2_model(
        dataset, task["a2_checkpoint"], device).to(device)
    for parameter in a2_model.parameters():
        parameter.requires_grad_(False)
    base_feature_dim = int(cfg.gt.dim_hidden) * 3
    model = CETFusionClassifier(
        base_feature_dim=base_feature_dim,
        num_currencies=int(full_store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(
            full_store.raw_edge_attr[:, 3].max()) + 1,
        variant=task["variant"],
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
        max_epochs,
        min(int(spec["warmup_epochs"]), max_epochs),
    )
    task_dir = args.output_dir / (
        f"{task['dataset']}_{task['variant']}_"
        f"seed{task['seed']}_{commit}"
    )
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    trajectory = task_dir / "trajectory.jsonl"
    checkpoint = task_dir / "best_val.ckpt"
    events = []
    best_val = -1.0
    stale = 0
    stopped_early = False
    started = time.monotonic()
    for epoch in range(max_epochs):
        train = train_epoch(
            model,
            a2_model,
            loaders[0],
            index,
            teacher,
            optimizer,
            spec,
            task,
            device,
            args.train_step_cap,
        )
        scheduler.step()
        if (epoch + 1) % int(spec["eval_period"]):
            continue
        row = evaluation_event(
            model,
            a2_model,
            loaders,
            index,
            spec,
            task,
            device,
            epoch,
            train,
            optimizer,
            args.eval_step_cap,
        )
        write_jsonl(trajectory, row)
        events.append(row)
        previous = best_val
        if row["val"]["f1"] > best_val:
            best_val = row["val"]["f1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_threshold": row["val_threshold"],
                "val_f1": row["val"]["f1"],
                "git_commit": commit,
                "sampling_protocol": "dynamic_random",
            }, checkpoint)
        if row["val"]["f1"] > previous + float(
                spec["early_stop_delta"]):
            stale = 0
        else:
            stale += 1
        if (
            epoch + 1 >= int(spec["min_epochs"])
            and stale >= int(spec["early_stop_evaluations"])
        ):
            stopped_early = True
            break
    if not events:
        raise RuntimeError("no evaluation event was produced")
    selected = max(events, key=lambda row: row["val"]["f1"])
    raw_best = max(
        events, key=lambda row: row["test"]["normal"]["f1"])
    baseline = INITIAL_A2[task["dataset"]]
    manifest = {
        "dataset": task["dataset"],
        "model": "CET-FraudGT",
        "variant": task["variant"],
        "seed": int(task["seed"]),
        "git_commit": commit,
        "config": config_relative,
        "checkpoint": str(checkpoint),
        "a2_checkpoint": task["a2_checkpoint"],
        "oof_root": str(args.oof_root),
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "selection": task["selection"],
        "max_tokens": int(spec["max_tokens"]),
        "epochs_completed": events[-1]["epoch"] + 1,
        "stopped_early": stopped_early,
        "train_step_cap": args.train_step_cap or None,
        "eval_step_cap": args.eval_step_cap or None,
        "validation_loader_iterations": sum(
            row["validation_loader_iterations"] for row in events),
        "test_loader_iterations": sum(
            row["test_loader_iterations"] for row in events),
        "val_selected": selected,
        "raw_best": raw_best,
        "val_selected_test_f1": selected["test"]["normal"]["f1"],
        "raw_best_test_f1": raw_best["test"]["normal"]["f1"],
        "delta_val_selected_vs_initial_a2": (
            selected["test"]["normal"]["f1"]
            - baseline["val_selected_test_f1"]
        ),
        "delta_raw_best_vs_initial_a2": (
            raw_best["test"]["normal"]["f1"]
            - baseline["raw_best_test_f1"]
        ),
        "sampling_variation_note": (
            "possible_dynamic_sampling_variation"
            if abs(
                selected["test"]["normal"]["f1"]
                - baseline["val_selected_test_f1"]
            ) < 0.005
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = task_dir / "phaseb_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": task["dataset"],
        "variant": task["variant"],
        "commit": commit,
        "val_selected_test_f1": manifest["val_selected_test_f1"],
        "raw_best_test_f1": manifest["raw_best_test_f1"],
        "delta_val_selected": manifest[
            "delta_val_selected_vs_initial_a2"],
        "output": str(task_dir),
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--train-step-cap", type=int)
    parser.add_argument("--eval-step-cap", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    if args.task_index < 0 or args.task_index >= len(spec["tasks"]):
        raise IndexError("task index out of range")
    train_task(spec, spec["tasks"][args.task_index], args)


if __name__ == "__main__":
    main()
