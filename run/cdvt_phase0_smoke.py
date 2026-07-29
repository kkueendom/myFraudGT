#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

import fraudGT  # noqa: F401
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.model_builder import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overfit-steps", type=int, default=24)
    parser.add_argument("--min-relative-loss-drop", type=float, default=0.05)
    parser.add_argument("--max-batch-search", type=int, default=64)
    return parser.parse_args()


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def configure(path, device):
    raw = yaml.safe_load(path.read_text())
    if raw.get("val", {}).get("fixed_target_panel", False):
        raise RuntimeError("fixed target panel is forbidden")
    if raw["train"]["sampler"] != "link_neighbor":
        raise RuntimeError("CDVT smoke requires LinkNeighborLoader")
    set_cfg(cfg)
    load_cfg(cfg, Namespace(cfg_file=str(path), opts=[]))
    cfg.device = device
    cfg.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False
    cfg.val.fixed_target_panel = False
    return raw


def gradient_norm(module):
    values = [
        parameter.grad.detach().float().norm()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu())


def clone_batch(raw_batch, device):
    batch = raw_batch.clone()
    batch.split = "train"
    return batch.to(device)


def target_labels(raw_batch):
    store = raw_batch[("node", "to", "node")]
    mask = torch.isin(store.e_id, store.target_edge_id)
    return store.y[mask]


def main():
    args = parse_args()
    raw_config = configure(args.config.resolve(), args.device)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    for split, wrapped in zip(("train", "val", "test"), loaders):
        loader = wrapped.loader
        if getattr(loader, "generator", None) is not None:
            raise RuntimeError(f"{split} loader has a dedicated generator")
        sampler = getattr(loader, "sampler", None)
        if getattr(sampler, "generator", None) is not None:
            raise RuntimeError(f"{split} sampler has a dedicated generator")

    device = torch.device(args.device)
    model = create_model(dataset=dataset).to(device)
    model.train()
    raw_batch = None
    batch_search_index = None
    for index, candidate in enumerate(loaders[0].loader):
        labels = target_labels(candidate)
        positives = int(labels.long().sum())
        if 0 < positives < labels.numel():
            raw_batch = candidate
            batch_search_index = index
            break
        if index + 1 >= args.max_batch_search:
            break
    if raw_batch is None:
        raise AssertionError(
            "could not find a dynamic training batch containing both classes "
            f"within {args.max_batch_search} draws")

    batch = clone_batch(raw_batch, device)
    logits, labels, diagnostics = model.forward_details(batch, "normal")
    loss, _ = compute_loss(logits, labels)
    loss.backward()
    account_gradient = gradient_norm(model.account_encoder)
    event_gradient = gradient_norm(model.dual_view.event_encoder)
    fusion_gradient = gradient_norm(model.dual_view.cross_attention)
    if min(account_gradient, event_gradient, fusion_gradient) <= 0:
        raise AssertionError("both encoders and fusion must receive gradients")

    model.eval()
    with torch.no_grad():
        initial_logits, initial_labels, _ = model.forward_details(
            clone_batch(raw_batch, device), "normal")
        initial_overfit_loss, _ = compute_loss(initial_logits, initial_labels)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.base_lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    overfit_losses = []
    model.train()
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        step_logits, step_labels, _ = model.forward_details(
            clone_batch(raw_batch, device), "normal")
        step_loss, _ = compute_loss(step_logits, step_labels)
        step_loss.backward()
        optimizer.step()
        overfit_losses.append(float(step_loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        normal, _, normal_diagnostics = model.forward_details(
            clone_batch(raw_batch, device), "normal")
        final_overfit_loss, _ = compute_loss(normal, initial_labels)
        shuffled, _, _ = model.forward_details(
            clone_batch(raw_batch, device), "shuffled")
        off, _, off_diagnostics = model.forward_details(
            clone_batch(raw_batch, device), "off")
    if torch.equal(normal, shuffled) or torch.equal(normal, off):
        raise AssertionError("event interventions must alter predictions")

    initial_loss_value = float(initial_overfit_loss.detach().cpu())
    final_loss_value = float(final_overfit_loss.detach().cpu())
    relative_loss_drop = (
        initial_loss_value - final_loss_value) / max(initial_loss_value, 1e-12)
    if relative_loss_drop < args.min_relative_loss_drop:
        raise AssertionError(
            "tiny-batch overfit failed: "
            f"relative loss drop {relative_loss_drop:.6f} is below "
            f"{args.min_relative_loss_drop:.6f}")

    event_count = normal_diagnostics["event_count"].detach().cpu().float()
    row = {
        "phase": "phase0_gpu_smoke",
        "sampling_protocol": "dynamic_random",
        "dataset": str(cfg.dataset.name),
        "variant": str(cfg.cdvt.variant),
        "seed": int(cfg.seed),
        "git_commit": git_commit(),
        "config": str(args.config.resolve()),
        "checkpoint": None,
        "device": args.device,
        "batch_targets": int(labels.numel()),
        "positive_targets": int(labels.long().sum().cpu()),
        "negative_targets": int(
            labels.numel() - labels.long().sum().cpu()),
        "batch_search_index": batch_search_index,
        "max_batch_search": args.max_batch_search,
        "loss": float(loss.detach().cpu()),
        "account_gradient_norm": account_gradient,
        "event_gradient_norm": event_gradient,
        "fusion_gradient_norm": fusion_gradient,
        "mean_event_count": float(event_count.mean()),
        "min_event_count": int(event_count.min()),
        "max_event_count": int(event_count.max()),
        "off_event_count": int(
            off_diagnostics["event_count"].detach().cpu().sum()),
        "normal_shuffled_mean_abs_logit_delta": float(
            (normal - shuffled).abs().mean().cpu()),
        "normal_off_mean_abs_logit_delta": float(
            (normal - off).abs().mean().cpu()),
        "overfit_steps": args.overfit_steps,
        "overfit_initial_loss": initial_loss_value,
        "overfit_final_loss": final_loss_value,
        "overfit_relative_loss_drop": relative_loss_drop,
        "overfit_training_losses": overfit_losses,
        "fixed_target_panel": False,
        "loader_shuffle": True,
        "config_snapshot": raw_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: row[key]
        for key in (
            "git_commit", "batch_targets", "loss",
            "account_gradient_norm", "event_gradient_norm",
            "fusion_gradient_norm", "mean_event_count",
            "normal_shuffled_mean_abs_logit_delta",
            "normal_off_mean_abs_logit_delta",
            "overfit_steps", "overfit_initial_loss",
            "overfit_final_loss", "overfit_relative_loss_drop",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
