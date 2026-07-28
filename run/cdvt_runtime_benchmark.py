#!/usr/bin/env python3
"""Benchmark normal-only inference from an existing formal checkpoint."""

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

import fraudGT  # noqa: F401
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_dataset, create_loader
from fraudGT.graphgym.model_builder import create_model
from run.cdvt_phase1_screen import (
    audit_loaders,
    configure,
    git_output,
    normal_forward,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=256)
    return parser.parse_args()


def effective_config(payload):
    config = deepcopy(payload["config_snapshot"])
    config["seed"] = int(payload["seed"])
    config["model"]["type"] = (
        "GTModel" if payload["architecture_variant"] == "account_only"
        else "CDVTModel")
    config["cdvt"]["variant"] = payload["architecture_variant"]
    config["cdvt"]["lambda_cons"] = float(payload["lambda_cons"])
    config["cdvt"]["history_k"] = int(payload.get(
        "history_k", config["cdvt"].get("history_k", 4)))
    config["cdvt"]["use_relation_types"] = payload.get(
        "use_relation_types",
        config["cdvt"].get("use_relation_types", True),
    )
    return config


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def consume(model, loader, device, max_batches):
    steps = 0
    targets = 0
    for raw_batch in loader:
        raw_batch.split = "test"
        raw_batch.to(device)
        _, labels, _ = normal_forward(model, raw_batch)
        targets += int(labels.numel())
        steps += 1
        if steps >= max_batches:
            break
    return steps, targets


def main():
    args = parse_args()
    if args.warmup_batches < 0 or args.max_batches < 1:
        raise ValueError("invalid benchmark batch budget")
    protected = (
        args.output_dir / "benchmark.json",
        args.output_dir / "benchmark_config.yaml",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite benchmark artifacts: {existing}")
    payload = json.loads(args.manifest.read_text())
    if payload.get("sampling_protocol") != "dynamic_random":
        raise ValueError("source manifest is not dynamic_random")
    if int(payload.get("seed", -1)) != 42:
        raise ValueError("runtime comparison is frozen to seed 42")
    variant = str(payload["variant"])
    architecture = str(payload["architecture_variant"])
    if variant not in {"account_only", "dual_view"}:
        raise ValueError("benchmark requires account_only or dual_view")
    if architecture != variant:
        raise ValueError("source variant and architecture differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "benchmark_config.yaml"
    config = effective_config(payload)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    runner_args = Namespace(
        config=config_path,
        device=args.device,
        variant=architecture,
        experiment_label=variant,
        lambda_cons=0.0,
        max_epochs=int(config["optim"]["max_epoch"]),
    )
    configure(runner_args)
    seed_everything(int(cfg.seed))
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_audit = audit_loaders(loaders)
    device = torch.device(args.device)
    model = create_model(dataset=dataset).to(device)
    checkpoint_path = Path(payload["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    if args.warmup_batches:
        consume(model, loaders[2], device, args.warmup_batches)
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.monotonic()
    steps, targets = consume(model, loaders[2], device, args.max_batches)
    synchronize(device)
    elapsed = time.monotonic() - started
    result = {
        "phase": "CDVT_runtime",
        "sampling_protocol": "dynamic_random",
        "dataset": str(payload["dataset"]),
        "variant": variant,
        "architecture_variant": architecture,
        "seed": 42,
        "benchmark_git_commit": git_output("rev-parse", "HEAD"),
        "source_git_commit": str(payload["git_commit"]),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path.resolve()),
        "config": str(config_path.resolve()),
        "warmup_batches": int(args.warmup_batches),
        "steps": steps,
        "targets": targets,
        "elapsed_seconds": elapsed,
        "seconds_per_batch": elapsed / max(steps, 1),
        "seconds_per_target": elapsed / max(targets, 1),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
        "loader_audit": loader_audit,
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
