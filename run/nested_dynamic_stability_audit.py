#!/usr/bin/env python3
"""Expand and run one nested model-seed/dynamic-stream stability task."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from run.a2_dynamic_stability_audit import run_audit  # noqa: E402


def expand_tasks(spec):
    tasks = []
    for dataset in spec["datasets"]:
        run_dir = Path(dataset["run_dir"])
        for model_seed in dataset["model_seeds"]:
            for stream_index, audit_seed in enumerate(spec["audit_seeds"]):
                tasks.append({
                    "dataset": dataset["dataset"],
                    "config": str(run_dir / "config.yaml"),
                    "checkpoint": str(
                        run_dir
                        / str(model_seed)
                        / "ckpt"
                        / "{}.ckpt".format(spec["checkpoint_epoch"])
                    ),
                    "model": spec["model"],
                    "variant": spec["variant"],
                    "model_seed": int(model_seed),
                    "audit_seed": int(audit_seed),
                    "stream_index": int(stream_index),
                    "experiment_label": (
                        "v4_seed{}_stream{}".format(
                            model_seed, stream_index)
                    ),
                    "repeats": int(spec["repeats"]),
                    "expected_batch_size": int(
                        dataset["expected_batch_size"]),
                    "expected_val_iter_per_epoch": int(
                        spec["expected_val_iter_per_epoch"]),
                })
    return tasks


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
    tasks = expand_tasks(spec)
    if len(tasks) != int(spec["task_count"]):
        raise RuntimeError("expanded task count differs from specification")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError("task index is outside the registered task list")
    task = tasks[args.task_index]
    for path in (task["config"], task["checkpoint"]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    run_audit(spec, task, args)


if __name__ == "__main__":
    main()
