#!/usr/bin/env python3
"""GPU queue for the six-dataset 500-epoch DMPRD P0 formal screen."""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(os.environ.get(
    "DMPRD_P0_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
PYTHON = os.environ.get(
    "FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
MAX_EPOCH = int(os.environ.get("DMPRD_P0_MAX_EPOCH", "500"))
DONE_EPOCH = MAX_EPOCH - 1
POLL_SECONDS = int(os.environ.get("DMPRD_P0_POLL_SECONDS", "1200"))
LAUNCH_SETTLE_SECONDS = int(os.environ.get(
    "DMPRD_P0_LAUNCH_SETTLE_SECONDS", "20"))
MIN_FREE_MIB = int(os.environ.get("DMPRD_P0_MIN_FREE_MIB", "14000"))
MAX_UTIL = int(os.environ.get("DMPRD_P0_MAX_UTIL", "15"))
OUT_DIR = Path(os.environ.get(
    "DMPRD_P0_OUT_DIR", str(REPO / "results" / "dmprd_p0_formal500")))
EVENTS = REPO / ".dmprd_p0_formal500_queue.events"
ACTIVE_DIR = REPO / ".dmprd_p0_formal500_active"
FAILED_DIR = REPO / ".dmprd_p0_formal500_failed"


VARIANTS = {
    "p0_full": {
        "reliability": True,
        "sample_gate": True,
    },
}

# Start the expensive datasets first so the sixth task can overlap their tail.
TASKS = [
    ("Large-LI", "p0_full", 44),
    ("Large-HI", "p0_full", 43),
    ("Medium-LI", "p0_full", 44),
    ("Medium-HI", "p0_full", 42),
    ("Small-LI", "p0_full", 42),
    ("Small-HI", "p0_full", 42),
]


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            output.append(row)
    return output


def run_stem(dataset, variant, seed):
    return f"AML-{dataset}-DMPRDP0Formal500-{variant}-Seed{seed}"


def run_dirs(dataset, variant, seed):
    return sorted(OUT_DIR.glob(run_stem(dataset, variant, seed) + "-gpu*"))


def task_done(dataset, variant, seed):
    for run_dir in run_dirs(dataset, variant, seed):
        seed_dir = run_dir / str(seed)
        train = rows(seed_dir / "train" / "stats.json")
        val = rows(seed_dir / "val" / "stats.json")
        test = rows(seed_dir / "test" / "stats.json")
        if train and val and test and int(train[-1]["epoch"]) >= DONE_EPOCH:
            return True
    return False


def marker_path(dataset, variant, seed, gpu):
    return ACTIVE_DIR / f"{dataset}_{variant}_seed{seed}_gpu{gpu}.json"


def failed_path(dataset, variant, seed):
    return FAILED_DIR / f"{dataset}_{variant}_seed{seed}.json"


def write_marker(dataset, variant, seed, gpu, pid, log_path):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(dataset, variant, seed, gpu).write_text(json.dumps({
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "pid": pid,
        "log": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, sort_keys=True))


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        text=True, capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def active_markers():
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    markers = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            marker = json.loads(path.read_text())
            key = (
                marker["dataset"], marker["variant"], int(marker["seed"]))
            pid = int(marker["pid"])
        except Exception:
            log(f"invalid marker retained for review: {path}")
            continue
        if task_done(*key):
            path.unlink(missing_ok=True)
            continue
        if not process_alive(pid):
            failure = dict(marker)
            failure["detected_at"] = datetime.now().isoformat(
                timespec="seconds")
            failure["reason"] = "process exited before completion"
            failed_path(*key).write_text(json.dumps(failure, sort_keys=True))
            path.unlink(missing_ok=True)
            log(
                f"task failed dataset={key[0]} variant={key[1]} "
                f"seed={key[2]} pid={pid}; no automatic retry")
            continue
        markers.append(marker)
    return markers


def failed_keys():
    output = set()
    for path in FAILED_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            output.add((item["dataset"], item["variant"], int(item["seed"])))
        except Exception:
            continue
    return output


def gpu_indices():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False)
    output = []
    for line in result.stdout.splitlines():
        try:
            output.append(int(line.strip()))
        except Exception:
            continue
    return sorted(output)


def gpu_compute_pids(gpu):
    result = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    output = []
    for line in result.stdout.splitlines():
        try:
            output.append(int(line.strip()))
        except Exception:
            continue
    return output


def gpu_free_and_util(gpu):
    result = subprocess.run([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return 0, 100
    free, util = [
        int(part.strip()) for part in result.stdout.splitlines()[0].split(",")]
    return free, util


def idle_gpus(markers):
    active_gpus = {int(marker["gpu"]) for marker in markers}
    output = []
    for gpu in gpu_indices():
        if gpu in active_gpus or gpu_compute_pids(gpu):
            continue
        free, util = gpu_free_and_util(gpu)
        if free >= MIN_FREE_MIB and util <= MAX_UTIL:
            output.append(gpu)
    return output


def pending_tasks(markers):
    active = {
        (item["dataset"], item["variant"], int(item["seed"]))
        for item in markers
    }
    failed = failed_keys()
    return [
        task for task in TASKS
        if not task_done(*task) and task not in active and task not in failed
    ]


def launch(dataset, variant_name, seed, gpu):
    variant = VARIANTS[variant_name]
    run_name = run_stem(dataset, variant_name, seed)
    stdout_path = REPO / f".dmprd_p0_formal500_{run_name}_gpu{gpu}.log"
    cmd = [
        PYTHON, "-m", "fraudGT.main",
        "--cfg", f"configs/evidence_gate_v4/AML-{dataset}.yaml",
        "--repeat", "1", "--gpu", "0",
        "out_dir", str(OUT_DIR),
        "name_tag", f"DMPRDP0Formal500-{variant_name}-Seed{seed}",
        "seed", str(seed),
        "optim.max_epoch", str(MAX_EPOCH),
        "train.early_stop", "False",
        "train.tqdm", "False",
        "val.tqdm", "False",
        "train.auto_resume", "False",
        "model.edge_decoding", "dmprd",
        "model.dmprd_num_slots", "4",
        "model.dmprd_use_distribution_stats", "False",
        "model.dmprd_use_reliability_gate", str(variant["reliability"]),
        "model.dmprd_use_sample_gate", str(variant["sample_gate"]),
        "model.dmprd_support_tau", "16.0",
        "model.dmprd_variance_tau", "0.25",
        "model.dmprd_margin_tau", "0.10",
        "model.dmprd_reliability_floor", "0.25",
        "model.dmprd_delta_max", "1.0",
        "model.dmprd_beta_max", "1.0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    with stdout_path.open("ab") as handle:
        process = subprocess.Popen(
            cmd, cwd=str(REPO), env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    write_marker(dataset, variant_name, seed, gpu, process.pid, stdout_path)
    log(
        f"launch dataset={dataset} variant={variant_name} seed={seed} "
        f"gpu={gpu} pid={process.pid}")
    time.sleep(LAUNCH_SETTLE_SECONDS)


def main():
    log(
        f"DMPRD P0 formal queue start max_epoch={MAX_EPOCH} "
        f"tasks={len(TASKS)} poll={POLL_SECONDS}s")
    last_wait_state = None
    while True:
        markers = active_markers()
        pending = pending_tasks(markers)
        if not pending and not markers:
            failures = len(failed_keys())
            log(f"DMPRD P0 formal queue finished failures={failures}")
            return 1 if failures else 0

        launched = False
        for gpu in idle_gpus(markers):
            markers = active_markers()
            pending = pending_tasks(markers)
            if not pending:
                break
            launch(*pending[0], gpu)
            launched = True
            markers = active_markers()

        wait_state = (
            len(pending_tasks(markers)),
            tuple(sorted(
                (item["dataset"], item["variant"], int(item["gpu"]))
                for item in markers)),
        )
        if not launched and wait_state != last_wait_state:
            log(
                f"waiting pending={wait_state[0]} active={len(markers)}; "
                f"next check in {POLL_SECONDS}s")
        last_wait_state = wait_state
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
