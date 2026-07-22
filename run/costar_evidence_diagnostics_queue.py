#!/usr/bin/env python3
"""Low-frequency queue for post-training COSTAR evidence diagnostics."""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = Path(os.environ.get(
    "COSTAR_DIAG_SPEC",
    str(REPO / "run" / "costar_evidence_diagnostics_spec.json")))
SPEC = json.loads(SPEC_PATH.read_text())
PYTHON = os.environ.get(
    "FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
RESULT_ROOT = Path(os.environ.get(
    "COSTAR_RESULT_ROOT",
    "/e/yky/FraudGT_dynamic_results/costar_4abe58c6_dynamic500"))
RUNTIME_ROOT = Path(os.environ.get(
    "COSTAR_DIAG_RUNTIME", "/e/yky/FraudGT_dynamic_runtime"))
GPU_ALLOWLIST = tuple(int(item) for item in os.environ.get(
    "COSTAR_DIAG_GPUS", "4,6").split(",") if item.strip())
POLL_SECONDS = int(os.environ.get("COSTAR_DIAG_POLL_SECONDS", "1800"))
OUTPUT_ROOT = RESULT_ROOT / "evidence_diagnostics"
ACTIVE_ROOT = RUNTIME_ROOT / ".costar_evidence_diagnostics_active"
FAILED_ROOT = RUNTIME_ROOT / ".costar_evidence_diagnostics_failed"
EVENTS = RUNTIME_ROOT / ".costar_evidence_diagnostics.events"


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def command_output(args):
    result = subprocess.run(
        args, cwd=REPO, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout.strip()


def verify_version():
    branch = command_output(["git", "branch", "--show-current"])
    if branch != SPEC["expected_branch"]:
        raise RuntimeError(f"unexpected branch: {branch}")
    if command_output(["git", "status", "--porcelain"]):
        raise RuntimeError("diagnostic worktree must be clean")
    if SPEC.get("sampling_protocol") != "dynamic_random":
        raise RuntimeError("diagnostics must declare dynamic_random")
    if 0 in GPU_ALLOWLIST:
        raise RuntimeError("GPU0 is reserved")
    return command_output(["git", "rev-parse", "--short=8", "HEAD"])


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "epoch" in row:
                rows.append(row)
    return rows


def paths(task):
    seed_dir = RESULT_ROOT / task["run_name"] / str(task["seed"])
    output_dir = OUTPUT_ROOT / f"{task['dataset']}-seed{task['seed']}"
    return {
        "seed_dir": seed_dir,
        "checkpoint": seed_dir / "ckpt" /
        f"{SPEC['required_last_epoch']}.ckpt",
        "output_dir": output_dir,
        "output": output_dir / "diagnostics.json",
        "marker": ACTIVE_ROOT /
        f"{task['dataset']}-seed{task['seed']}.json",
    }


def task_ready(task):
    item = paths(task)
    train = read_rows(item["seed_dir"] / "train" / "stats.json")
    return bool(
        train and
        int(train[-1]["epoch"]) >= SPEC["required_last_epoch"] and
        item["checkpoint"].exists()
    )


def task_done(task, diagnostic_commit):
    output = paths(task)["output"]
    if not output.exists():
        return False
    result = json.loads(output.read_text())
    experiment = result.get("experiment", {})
    formal = result.get("formal_result", {})
    return (
        experiment.get("diagnostic_commit") == diagnostic_commit and
        experiment.get("model_commit") == SPEC["model_commit"] and
        experiment.get("sampling_protocol") == "dynamic_random" and
        int(formal.get("last_epoch", -1)) >= SPEC["required_last_epoch"]
    )


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], text=True,
        capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def active_markers(diagnostic_commit):
    ACTIVE_ROOT.mkdir(parents=True, exist_ok=True)
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    active = []
    for marker_path in ACTIVE_ROOT.glob("*.json"):
        marker = json.loads(marker_path.read_text())
        task = next(item for item in SPEC["tasks"]
                    if item["dataset"] == marker["dataset"] and
                    int(item["seed"]) == int(marker["seed"]))
        if task_done(task, diagnostic_commit):
            marker_path.unlink()
            preserve_artifacts(task)
        elif process_alive(marker["pid"]):
            active.append(marker)
        else:
            marker["failure"] = "process_exited_without_valid_output"
            (FAILED_ROOT / marker_path.name).write_text(
                json.dumps(marker, sort_keys=True))
            marker_path.unlink()
            log(f"failed dataset={task['dataset']} seed={task['seed']}")
    return active


def gpu_idle(gpu):
    query = subprocess.run([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    if query.returncode or not query.stdout.strip():
        return False
    free, utilization = (
        int(item.strip()) for item in query.stdout.split(",")[:2])
    processes = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    return free >= 9000 and utilization <= 15 and not processes.stdout.strip()


def launch(task, gpu, diagnostic_commit):
    item = paths(task)
    item["output_dir"].mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME_ROOT / (
        f".costar_diag_{task['dataset']}_seed{task['seed']}_gpu{gpu}.log")
    command = [
        PYTHON, str(REPO / "run" / "costar_evidence_diagnostics.py"),
        "--cfg", str(REPO / task["config"]),
        "--checkpoint", str(item["checkpoint"]),
        "--formal-run-dir", str(item["seed_dir"]),
        "--output", str(item["output"]),
        "--gpu", "0", "--seed", str(task["seed"]),
        "--model-commit", SPEC["model_commit"],
        "--required-last-epoch", str(SPEC["required_last_epoch"]),
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "disabled",
    })
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    item["marker"].write_text(json.dumps({
        "dataset": task["dataset"], "seed": task["seed"], "gpu": gpu,
        "pid": process.pid, "diagnostic_commit": diagnostic_commit,
        "model_commit": SPEC["model_commit"], "config": task["config"],
        "checkpoint": str(item["checkpoint"]), "log": str(log_path),
        "sampling_protocol": "dynamic_random",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, sort_keys=True))
    log(f"launch dataset={task['dataset']} seed={task['seed']} gpu={gpu} "
        f"pid={process.pid}")
    time.sleep(20)


def preserve_artifacts(task):
    item = paths(task)
    result = json.loads(item["output"].read_text())
    artifact_dir = item["output_dir"] / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_copy = artifact_dir / "final_epoch499.ckpt"
    if not final_copy.exists():
        shutil.copy2(item["checkpoint"], final_copy)
    config_copy = artifact_dir / Path(task["config"]).name
    if not config_copy.exists():
        shutil.copy2(REPO / task["config"], config_copy)

    selected_epoch = result["formal_result"]["selected_epoch"]
    selected_source = item["seed_dir"] / "ckpt" / f"{selected_epoch}.ckpt"
    if selected_source.exists():
        selected_copy = artifact_dir / "best_val.ckpt"
        if not selected_copy.exists():
            shutil.copy2(selected_source, selected_copy)
        best_status = {
            "status": "exact_checkpoint_preserved",
            "selected_epoch": selected_epoch,
            "checkpoint": str(selected_copy),
        }
    else:
        best_status = {
            "status": "exact_checkpoint_not_persisted_by_training_config",
            "selected_epoch": selected_epoch,
            "checkpoint": None,
            "reason": "train.ckpt_best=False and selected epoch was not periodic",
        }
    (artifact_dir / "best_val_checkpoint_status.json").write_text(
        json.dumps(best_status, indent=2, sort_keys=True) + "\n")
    write_manifest()


def write_manifest():
    records = []
    for task in SPEC["tasks"]:
        output = paths(task)["output"]
        if not output.exists():
            continue
        result = json.loads(output.read_text())
        formal = result["formal_result"]
        experiment = result["experiment"]
        records.append({
            "dataset": task["dataset"],
            "model": SPEC["model"],
            "variant": SPEC["variant"],
            "seed": task["seed"],
            "git_commit": SPEC["model_commit"],
            "diagnostic_commit": experiment["diagnostic_commit"],
            "config": experiment["config"],
            "checkpoint": experiment["checkpoint"],
            "selected_epoch": formal["selected_epoch"],
            "val_selected_test_f1": formal["val_selected_test_f1"],
            "raw_best_epoch": formal["raw_best_epoch"],
            "raw_best_test_f1": formal["raw_best_test_f1"],
            "delta_val_selected_f1": formal["delta_val_selected_f1"],
            "delta_raw_best_f1": formal["delta_raw_best_f1"],
            "sampling_protocol": "dynamic_random",
            "diagnostic_output": str(output),
        })
    manifest = OUTPUT_ROOT / "experiment_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records))


def main():
    diagnostic_commit = verify_version()
    log(f"start diagnostic_commit={diagnostic_commit}")
    last_state = None
    while True:
        active = active_markers(diagnostic_commit)
        pending = [task for task in SPEC["tasks"]
                   if not task_done(task, diagnostic_commit)]
        failed = {(item["dataset"], int(item["seed"]))
                  for path in FAILED_ROOT.glob("*.json")
                  for item in [json.loads(path.read_text())]}
        pending = [task for task in pending
                   if (task["dataset"], int(task["seed"])) not in failed]
        if not active and not pending:
            write_manifest()
            log("finished")
            return 0 if not failed else 1

        active_keys = {(item["dataset"], int(item["seed"]))
                       for item in active}
        ready = [task for task in pending if task_ready(task) and
                 (task["dataset"], int(task["seed"])) not in active_keys]
        used_gpus = {int(item["gpu"]) for item in active}
        for gpu in GPU_ALLOWLIST:
            if not ready:
                break
            if gpu not in used_gpus and gpu_idle(gpu):
                launch(ready.pop(0), gpu, diagnostic_commit)
                used_gpus.add(gpu)

        active = active_markers(diagnostic_commit)
        state = (
            tuple(sorted(task["dataset"] for task in pending)),
            tuple(sorted((item["dataset"], item["gpu"]) for item in active)),
        )
        if state != last_state:
            log(f"state pending={state[0]} active={state[1]}")
        last_state = state
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
