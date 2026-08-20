#!/usr/bin/env python3
"""Benchmark end-to-end inference for matched Multi-FraudGT variants."""

import argparse
import hashlib
import json
import statistics
import sys
import time
from argparse import Namespace
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
    audit_multi_dataset,
    configure,
    git_output,
    normal_forward,
    seed_everything,
)


VARIANTS = {
    "multi_account_only": "account_only",
    "multi_cdvt": "dual_view",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--batches-per-repeat", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


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
        steps += 1
        targets += int(labels.numel())
        if steps >= max_batches:
            break
    return steps, targets


def main():
    args = parse_args()
    if args.warmup_batches < 0:
        raise ValueError("warmup batches must be non-negative")
    if args.batches_per_repeat < 1 or args.repeats < 1:
        raise ValueError("measurement budget must be positive")
    protected = (
        args.output_dir / "benchmark.json",
        args.output_dir / "benchmark_config.yaml",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite benchmark artifacts: {existing}")

    config = yaml.safe_load(args.config.read_text())
    architecture = VARIANTS[args.variant]
    if int(config.get("seed", -1)) != 42:
        raise ValueError("quick runtime comparison is frozen to seed 42")
    if config.get("dataset", {}).get("reverse_mp") is not True:
        raise ValueError("runtime benchmark requires Multi-FraudGT RMP")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_copy = args.output_dir / "benchmark_config.yaml"
    config_copy.write_text(yaml.safe_dump(config, sort_keys=False))

    runner_args = Namespace(
        config=config_copy,
        device=args.device,
        variant=architecture,
        experiment_label=args.variant,
        lambda_cons=0.0,
        max_epochs=int(config["optim"]["max_epoch"]),
        early_stop_min_epoch=80,
        early_stop_patience_evals=10,
        disable_early_stop=True,
    )
    configure(runner_args)
    seed_everything(int(cfg.seed))
    dataset = create_dataset()
    multi_dataset_audit = audit_multi_dataset(dataset)
    loaders = create_loader(dataset=dataset, shuffle=True)
    loader_audit = audit_loaders(loaders)
    device = torch.device(args.device)
    model = create_model(dataset=dataset).to(device)
    model.eval()

    if args.warmup_batches:
        warmup_steps, _ = consume(
            model, loaders[2], device, args.warmup_batches)
        if warmup_steps != args.warmup_batches:
            raise RuntimeError("test loader ended during warm-up")
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    repeats = []
    for index in range(args.repeats):
        synchronize(device)
        started = time.monotonic()
        steps, targets = consume(
            model, loaders[2], device, args.batches_per_repeat)
        synchronize(device)
        elapsed = time.monotonic() - started
        if steps != args.batches_per_repeat:
            raise RuntimeError("test loader ended during measurement")
        repeats.append({
            "repeat": index + 1,
            "steps": steps,
            "targets": targets,
            "elapsed_seconds": elapsed,
            "seconds_per_batch": elapsed / steps,
            "targets_per_second": targets / elapsed,
        })

    latency = [row["seconds_per_batch"] for row in repeats]
    throughput = [row["targets_per_second"] for row in repeats]
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else "CPU")
    result = {
        "phase": "CDVT_multi_runtime_quick",
        "benchmark_mode": "normal_only_end_to_end_inference",
        "sampling_protocol": "dynamic_random",
        "dataset": str(cfg.dataset.name),
        "variant": args.variant,
        "architecture_variant": architecture,
        "account_backbone": "Multi-FraudGT",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "config": str(config_copy.resolve()),
        "config_sha256": hashlib.sha256(
            config_copy.read_bytes()).hexdigest(),
        "checkpoint_loaded": False,
        "weight_note": "fresh initialization; runtime is weight-independent",
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_batches": args.warmup_batches,
        "batches_per_repeat": args.batches_per_repeat,
        "repeats": args.repeats,
        "total_measured_batches": sum(row["steps"] for row in repeats),
        "total_measured_targets": sum(row["targets"] for row in repeats),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
        "mean_seconds_per_batch": statistics.mean(latency),
        "std_seconds_per_batch": statistics.pstdev(latency),
        "mean_targets_per_second": statistics.mean(throughput),
        "std_targets_per_second": statistics.pstdev(throughput),
        "repeat_rows": repeats,
        "loader_audit": loader_audit,
        "multi_dataset_audit": multi_dataset_audit,
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
