#!/usr/bin/env python3
"""Train fraud-label-free CPSE folds and emit aligned OOF evidence."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.cpse import CausalPredictiveSurpriseEncoder
from fraudGT.evidence.tier import EvidenceBatch, TemporalIncidentIndex
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from run.tier_phase1_evidence_qualification import (
    TASK,
    audit_protocol,
    configure_fraudgt,
    loader_audit,
    seed_process,
    verify_repository,
)
from run.tier_phase2b_crossfit_teachers import fold_mask


def move_evidence(evidence, device):
    return EvidenceBatch(
        tokens=evidence.tokens.to(device, non_blocking=True),
        mask=evidence.mask.to(device, non_blocking=True),
        context_edge_ids=evidence.context_edge_ids.to(
            device, non_blocking=True),
        support=evidence.support.to(device, non_blocking=True),
    )


def query(index, edge_ids, spec, device):
    evidence_cpu = index.query(
        edge_ids.cpu().long(),
        max_tokens=int(spec["max_tokens"]),
        selection="recent",
        selection_pool_factor=int(spec["selection_pool_factor"]),
    )
    target_raw = index.raw_edge_attr[edge_ids.cpu().long()]
    return (
        move_evidence(evidence_cpu, device),
        target_raw.to(device),
        evidence_cpu,
    )


def locate_oof_manifest(root, task):
    pattern = (
        f"{task['dataset']}_fold{task['fold']}_"
        f"seed{task['seed']}_*/phase2b_manifest.json"
    )
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one source manifest for {pattern}, found {len(paths)}")
    return json.loads(paths[0].read_text())


def train_cpse(model, loader, index, optimizer, task, spec, device, step_cap):
    best_loss = float("inf")
    best_state = None
    patience = 0
    trajectory = []
    loader_iterations = 0
    for epoch in range(int(spec["max_epochs"])):
        model.train()
        loss_sum = 0.0
        update_count = 0
        supervised_targets = 0
        held_out_seen = 0
        for step, batch in enumerate(loader):
            if step_cap and step >= step_cap:
                break
            loader_iterations += 1
            target_ids = batch[TASK].target_edge_id.detach().cpu().long()
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
            evidence, target_raw, _ = query(index, target_ids, spec, device)
            keep_device = keep.to(device)
            kept_evidence = EvidenceBatch(
                tokens=evidence.tokens[keep_device],
                mask=evidence.mask[keep_device],
                context_edge_ids=evidence.context_edge_ids[keep_device],
                support=evidence.support[keep_device],
            )
            prediction = model(kept_evidence)
            loss = model.self_supervised_loss(
                prediction, kept_evidence, target_raw[keep_device])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            update_count += 1
            supervised_targets += int(keep.sum())
        mean_loss = loss_sum / max(update_count, 1)
        trajectory.append({
            "epoch": epoch,
            "self_supervised_loss": mean_loss,
            "updates": update_count,
            "non_held_out_targets": supervised_targets,
            "held_out_targets_seen_as_unsupervised_context": held_out_seen,
        })
        if mean_loss < best_loss - float(spec["early_stop_min_delta"]):
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(spec["early_stop_patience"]):
            break
    if best_state is None:
        raise RuntimeError("CPSE completed no training update")
    model.load_state_dict(best_state)
    return trajectory, loader_iterations, best_loss


@torch.no_grad()
def collect_oof(model, index, payload, spec, device):
    model.eval()
    edge_ids = payload["edge_ids"].long()
    parts = {"normal": [], "shuffled": [], "off": []}
    batch_size = int(spec["collect_batch_size"])
    for start in range(0, edge_ids.numel(), batch_size):
        ids = edge_ids[start:start + batch_size]
        evidence, target_raw, _ = query(index, ids, spec, device)
        normal_prediction = model(evidence)
        parts["normal"].append(
            model.surprise_features(
                normal_prediction, evidence, target_raw).cpu())
        permutation = torch.randperm(ids.numel(), device=device)
        shuffled = evidence.shuffled(permutation)
        shuffled_prediction = model(shuffled)
        parts["shuffled"].append(
            model.surprise_features(
                shuffled_prediction, shuffled, target_raw).cpu())
        off = evidence.off()
        off_prediction = model(off)
        parts["off"].append(
            model.surprise_features(
                off_prediction, off, target_raw).cpu())
    source = index.edge_index[0, edge_ids]
    destination = index.edge_index[1, edge_ids]
    timestamps = index.timestamps[edge_ids]
    return {
        "edge_ids": edge_ids,
        "labels": payload["labels"].long(),
        "a2_scores": payload["a2_scores"].float(),
        "a2_threshold": float(payload["a2_threshold"]),
        "normal_features": torch.cat(parts["normal"]),
        "shuffled_features": torch.cat(parts["shuffled"]),
        "off_features": torch.cat(parts["off"]),
        "source_ids": source,
        "destination_ids": destination,
        "timestamps": timestamps,
    }


def run_task(spec, task, args):
    commit = verify_repository(spec)
    config_relative = spec["config_template"].format(
        dataset=task["dataset"])
    config_path = REPO_ROOT / config_relative
    config = yaml.safe_load(config_path.read_text())
    audit_protocol(spec, task, config)
    seed_process(int(task["seed"]) + 2003 * int(task["fold"]))
    configure_fraudgt(config_path, task["seed"], args.device)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_rows = loader_audit(loaders)
    train_loader = loaders[0]
    store = dataset["test"][TASK]
    index = TemporalIncidentIndex(
        edge_index=store.edge_index,
        timestamps=store.timestamps,
        raw_edge_attr=store.raw_edge_attr,
    )
    source_manifest = locate_oof_manifest(
        args.oof_root, task)
    device = torch.device(args.device)
    model = CausalPredictiveSurpriseEncoder(
        num_currencies=int(store.raw_edge_attr[:, 2].max()) + 1,
        num_payment_formats=int(store.raw_edge_attr[:, 3].max()) + 1,
        hidden_dim=int(spec["hidden_dim"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    started = time.monotonic()
    trajectory, train_iterations, best_loss = train_cpse(
        model,
        train_loader,
        index,
        optimizer,
        task,
        spec,
        device,
        args.train_step_cap,
    )
    del optimizer
    source_payload = torch.load(
        source_manifest["oof_scores"],
        map_location="cpu",
        weights_only=False,
    )
    expected_fold = fold_mask(
        source_payload["edge_ids"],
        task["seed"],
        spec["num_folds"],
        task["fold"],
        held_out=True,
    )
    if not expected_fold.all():
        raise AssertionError("source OOF table contains a foreign fold")
    evidence = collect_oof(
        model, index, source_payload, spec, device)
    output = args.output_dir / (
        f"{task['dataset']}_fold{task['fold']}_"
        f"seed{task['seed']}_{commit}"
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence_path = output / "cpse_oof_features.pt"
    checkpoint_path = output / "cpse_checkpoint.pt"
    trajectory_path = output / "training_trajectory.jsonl"
    torch.save({
        **evidence,
        "dataset": task["dataset"],
        "fold": int(task["fold"]),
        "seed": int(task["seed"]),
        "sampling_protocol": "dynamic_random",
        "git_commit": commit,
    }, evidence_path)
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    trajectory_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in trajectory))
    manifest = {
        "experiment": "CPSE_Phase0B_train_only_OOF",
        "dataset": task["dataset"],
        "fold": int(task["fold"]),
        "num_folds": int(spec["num_folds"]),
        "seed": int(task["seed"]),
        "git_commit": commit,
        "config": config_relative,
        "sampling_protocol": "dynamic_random",
        "loader_audit": loader_rows,
        "fraud_labels_used_for_cpse_training": False,
        "supervision": "self_supervised_next_transaction_only",
        "held_out_edges_excluded_from_cpse_loss": True,
        "held_out_edges_remain_unlabeled_context": True,
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "train_loader_iterations": train_iterations,
        "epochs_completed": len(trajectory),
        "best_self_supervised_loss": best_loss,
        "train_step_cap": args.train_step_cap or None,
        "oof_edges": int(evidence["edge_ids"].numel()),
        "oof_positives": int(evidence["labels"].sum()),
        "feature_dim": int(evidence["normal_features"].size(1)),
        "cpse_checkpoint": str(checkpoint_path),
        "cpse_oof_features": str(evidence_path),
        "source_oof_manifest": str(source_manifest["oof_scores"]),
        "runtime_seconds": time.monotonic() - started,
    }
    (output / "cpse_phase0b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path,
        default=REPO_ROOT / "run/cpse_phase0b_spec.json")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-step-cap", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    manifest = run_task(spec, spec["tasks"][args.task_index], args)
    print(json.dumps({
        key: manifest[key]
        for key in (
            "dataset", "fold", "epochs_completed", "oof_edges",
            "best_self_supervised_loss", "runtime_seconds")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
