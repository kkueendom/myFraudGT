#!/usr/bin/env python3
"""Wait for idle GPUs, then run the evidence-gate v4 screening sequence.

The queue never kills existing work and never retries a failed task silently.
It logs only state changes plus a sparse heartbeat.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(
    os.environ.get(
        "FRAUDGT_V4_REPO",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()
RESULT_ROOT = REPO / "results" / "evidence_gate_v4_screen"
ACTIVE_DIR = REPO / ".evidence_gate_v4_active"
FAILED_DIR = REPO / ".evidence_gate_v4_failed"
EVENTS = REPO / ".evidence_gate_v4_queue.events"
POLL_SECONDS = int(os.environ.get("EVIDENCE_GATE_V4_POLL_SECONDS", "900"))
MIN_FREE_MIB = int(os.environ.get("EVIDENCE_GATE_V4_MIN_FREE_MIB", "9000"))
MAX_UTIL = int(os.environ.get("EVIDENCE_GATE_V4_MAX_UTIL", "10"))


TASKS = [
    {
        "name": "smoke-router-small-hi",
        "stage": 0,
        "cfg": "configs/evidence_gate_v4/AML-Small-HI.yaml",
        "dataset": "Small-HI",
        "max_epoch": 4,
        "train_iters": 16,
        "val_iters": 16,
        "eval_period": 1,
    },
    {
        "name": "nogate-small-hi",
        "stage": 1,
        "cfg": "configs/evidence_gate_v4_nogate/AML-Small-HI.yaml",
        "dataset": "Small-HI",
        "max_epoch": 80,
        "train_iters": 256,
        "val_iters": 256,
        "eval_period": 2,
    },
    {
        "name": "nogate-small-li",
        "stage": 1,
        "cfg": "configs/evidence_gate_v4_nogate/AML-Small-LI.yaml",
        "dataset": "Small-LI",
        "max_epoch": 80,
        "train_iters": 256,
        "val_iters": 256,
        "eval_period": 2,
    },
    {
        "name": "router-small-hi",
        "stage": 1,
        "cfg": "configs/evidence_gate_v4/AML-Small-HI.yaml",
        "dataset": "Small-HI",
        "max_epoch": 80,
        "train_iters": 256,
        "val_iters": 256,
        "eval_period": 2,
    },
    {
        "name": "router-small-li",
        "stage": 1,
        "cfg": "configs/evidence_gate_v4/AML-Small-LI.yaml",
        "dataset": "Small-LI",
        "max_epoch": 80,
        "train_iters": 256,
        "val_iters": 256,
        "eval_period": 2,
    },
]


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def json_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            rows.append(row)
    return rows


def task_out(task):
    return RESULT_ROOT / task["name"]


def task_done(task):
    expected_last_epoch = task["max_epoch"] - 1
    for stats_path in task_out(task).glob("**/train/stats.json"):
        rows = json_rows(stats_path)
        seed_dir = stats_path.parent.parent
        if rows and (
            int(rows[-1]["epoch"]) >= expected_last_epoch
            or (seed_dir / "early_stop.json").exists()
        ):
            return True
    return False


def marker_path(task):
    return ACTIVE_DIR / f"{task['name']}.json"


def failed_path(task):
    return FAILED_DIR / f"{task['name']}.json"


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def load_marker(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def reconcile_markers():
    for task in TASKS:
        path = marker_path(task)
        if not path.exists():
            continue
        marker = load_marker(path)
        if marker and process_alive(marker.get("pid", -1)):
            continue
        path.unlink(missing_ok=True)
        if task_done(task):
            log(f"completed task={task['name']}")
        else:
            FAILED_DIR.mkdir(parents=True, exist_ok=True)
            payload = marker or {"task": task["name"], "reason": "invalid marker"}
            payload["detected_at"] = datetime.now().isoformat()
            failed_path(task).write_text(json.dumps(payload, indent=2) + "\n")
            log(f"failed task={task['name']}; manual inspection required")


def running_gpu_ids():
    gpu_ids = set()
    for path in ACTIVE_DIR.glob("*.json"):
        marker = load_marker(path)
        if marker and process_alive(marker.get("pid", -1)):
            gpu_ids.add(int(marker["gpu"]))
    return gpu_ids


def free_gpus():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    reserved = running_gpu_ids()
    candidates = []
    for line in result.stdout.splitlines():
        try:
            index, free_mib, util = [int(x.strip()) for x in line.split(",")]
        except Exception:
            continue
        if index in reserved:
            continue
        if free_mib >= MIN_FREE_MIB and util <= MAX_UTIL:
            candidates.append((index, free_mib, util))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [item[0] for item in candidates]


def runnable_tasks():
    smoke = TASKS[0]
    if not task_done(smoke):
        if marker_path(smoke).exists() or failed_path(smoke).exists():
            return []
        return [smoke]

    tasks = []
    for task in TASKS[1:]:
        if task_done(task):
            continue
        if marker_path(task).exists() or failed_path(task).exists():
            continue
        tasks.append(task)
    return tasks


def launch(task, gpu):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    task_out(task).mkdir(parents=True, exist_ok=True)
    log_path = REPO / f".evidence_gate_v4_{task['name']}_gpu{gpu}.stdout"
    command = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "conda activate fraudgt_dual_gate; "
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        f"exec python -m fraudGT.main --cfg {task['cfg']} --repeat 1 --gpu 0 "
        f"out_dir {task_out(task)} seed 42 "
        f"optim.max_epoch {task['max_epoch']} "
        f"train.iter_per_epoch {task['train_iters']} "
        f"val.iter_per_epoch {task['val_iters']} "
        f"train.eval_period {task['eval_period']} "
        "train.early_stop False train.tqdm False val.tqdm False "
        "train.auto_resume True train.epoch_resume -1"
    )
    with log_path.open("ab") as output:
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(REPO),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    marker = {
        "task": task["name"],
        "dataset": task["dataset"],
        "gpu": gpu,
        "pid": process.pid,
        "log": str(log_path),
        "started_at": datetime.now().isoformat(),
    }
    marker_path(task).write_text(json.dumps(marker, indent=2) + "\n")
    log(f"launched task={task['name']} gpu={gpu} pid={process.pid}")


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    log(
        f"v4 queue started poll={POLL_SECONDS}s min_free={MIN_FREE_MIB}MiB "
        f"max_util={MAX_UTIL}%"
    )
    heartbeat = 0
    while True:
        reconcile_markers()
        if all(task_done(task) for task in TASKS):
            log("v4 screening queue completed")
            return 0

        tasks = runnable_tasks()
        gpus = free_gpus()
        for task, gpu in zip(tasks, gpus):
            launch(task, gpu)

        heartbeat += 1
        if heartbeat % 6 == 0 and not gpus:
            remaining = sum(not task_done(task) for task in TASKS)
            log(f"waiting for idle gpu; remaining={remaining}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
