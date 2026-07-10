#!/usr/bin/env python3
"""GPU queue for the evidence-gate M1 formal runs.

The queue is intentionally conservative:
- it launches at most one training process per GPU;
- it uses any idle GPU with enough free memory;
- it avoids duplicate dataset/seed tasks already running in this output dir;
- it logs only queue-level events.
"""

import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(
    os.environ.get(
        "FRAUDGT_M1_REPO",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()
EVENTS = REPO / ".evidence_gate_m1_queue.events"
LEGACY_QUEUE_PID = REPO / ".evidence_gate_queue.pid"
ACTIVE_DIR = REPO / ".evidence_gate_m1_active"
POLL_SECONDS = int(
    os.environ.get(
        "EVIDENCE_GATE_M1_POLL_SECONDS",
        os.environ.get("EVIDENCE_GATE_V2_POLL_SECONDS", "1800"),
    )
)
LAUNCH_SETTLE_SECONDS = int(
    os.environ.get(
        "EVIDENCE_GATE_M1_LAUNCH_SETTLE_SECONDS",
        os.environ.get("EVIDENCE_GATE_V2_LAUNCH_SETTLE_SECONDS", "300"),
    )
)
MIN_FREE_MIB = int(os.environ.get("EVIDENCE_GATE_MIN_FREE_MIB", "14000"))
GPU_IDS_RAW = os.environ.get("EVIDENCE_GATE_M1_GPU_IDS", "").strip()
ALLOWED_GPU_IDS = (
    {int(token.strip()) for token in GPU_IDS_RAW.split(",") if token.strip()}
    if GPU_IDS_RAW
    else None
)
PYTHON = os.environ.get(
    "FRAUDGT_PYTHON",
    "/d/miniconda3/envs/fraudGT/bin/python3.9",
)
PRIMARY_OUT_DIR = Path(
    os.environ.get(
        "EVIDENCE_GATE_M1_OUT_DIR",
        str(REPO / "results" / "evidence_gate_m1_proto"),
    )
)
DONE_EPOCH = 499

DATASETS = [
    "Small-HI",
    "Small-LI",
    "Medium-HI",
    "Medium-LI",
    "Large-HI",
    "Large-LI",
]
SEEDS = [42, 43, 44]
CFG_RE = re.compile(r"configs/evidence_gate_proto/AML-(?P<dataset>[^/\s]+)\.yaml")


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


def cfg_for(dataset):
    return f"configs/evidence_gate_proto/AML-{dataset}.yaml"


def run_dirs(out_dir, dataset):
    stem = f"AML-{dataset}"
    return sorted(Path(out_dir).glob(stem + "-gpu*"))


def seed_done_in(out_dir, dataset, seed):
    for run_dir in run_dirs(out_dir, dataset):
        seed_dir = run_dir / str(seed)
        train_rows = rows(seed_dir / "train" / "stats.json")
        val_rows = rows(seed_dir / "val" / "stats.json")
        test_rows = rows(seed_dir / "test" / "stats.json")
        if (
            train_rows
            and val_rows
            and test_rows
            and (
                int(train_rows[-1]["epoch"]) >= DONE_EPOCH
                or (seed_dir / "early_stop.json").exists()
            )
        ):
            return True
    return False


def task_done(dataset, seed):
    return seed_done_in(PRIMARY_OUT_DIR, dataset, seed)


def pending_tasks():
    return [
        (dataset, seed)
        for seed in SEEDS
        for dataset in DATASETS
        if not task_done(dataset, seed)
    ]


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return not result.stdout.strip().startswith("Z")


def reap_children():
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        log(f"reaped child pid={pid} status={status}")


def marker_path(dataset, seed, gpu):
    safe_dataset = dataset.replace("/", "_")
    return ACTIVE_DIR / f"{safe_dataset}_seed{seed}_gpu{gpu}.json"


def write_marker(dataset, seed, gpu, pid, log_path):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "pid": pid,
        "log": str(log_path),
        "out_dir": str(PRIMARY_OUT_DIR),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    marker_path(dataset, seed, gpu).write_text(json.dumps(payload, sort_keys=True))


def active_markers():
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    markers = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            marker = json.loads(path.read_text())
            dataset = marker["dataset"]
            seed = int(marker["seed"])
            gpu = int(marker["gpu"])
            pid = int(marker["pid"])
        except Exception:
            path.unlink(missing_ok=True)
            continue

        if task_done(dataset, seed):
            path.unlink(missing_ok=True)
            continue

        if not process_alive(pid):
            log(
                f"stale marker removed dataset={dataset} seed={seed} "
                f"gpu={gpu} pid={pid}; task will be retried"
            )
            path.unlink(missing_ok=True)
            continue

        markers.append(marker)
    return markers


def legacy_queue_alive():
    if not LEGACY_QUEUE_PID.exists():
        return False
    try:
        pid = int(LEGACY_QUEUE_PID.read_text().strip())
    except Exception:
        return False
    return process_alive(pid)


def driver_inventory_ok():
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        text=True,
        capture_output=True,
        check=False,
    )
    text = result.stdout + result.stderr
    return result.returncode == 0 and "Unable to determine" not in text


def gpu_indices():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    indices = []
    for line in result.stdout.splitlines():
        try:
            index = int(line.strip())
        except Exception:
            continue
        if ALLOWED_GPU_IDS is not None and index not in ALLOWED_GPU_IDS:
            continue
        indices.append(index)
    return indices


def gpu_compute_pids(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except Exception:
            continue
    return pids


def torch_cuda_ok(gpu):
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} {shlex.quote(PYTHON)} - <<'PY'\n"
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "x = torch.tensor([1.0], device='cuda:0')\n"
        "assert float(x.cpu()[0]) == 1.0\n"
        "PY"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return result.returncode == 0


def parse_task_from_args(args):
    if "-m fraudGT.main" not in args:
        return None
    if str(PRIMARY_OUT_DIR) not in args:
        return None

    cfg_match = CFG_RE.search(args)
    if not cfg_match:
        return None
    dataset = cfg_match.group("dataset")

    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()

    seed = None
    for idx, token in enumerate(tokens[:-1]):
        if token == "seed":
            try:
                seed = int(tokens[idx + 1])
            except Exception:
                return None
            break
    if seed is None:
        return None
    return dataset, seed


def active_tasks_from_ps():
    result = subprocess.run(
        ["ps", "-u", os.environ.get("USER", "yky"), "-o", "pid=,args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    tasks = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        task = parse_task_from_args(parts[1])
        if task is not None:
            tasks.setdefault(task, set()).add(pid)
    return tasks


def active_tasks():
    tasks = active_tasks_from_ps()
    for marker in active_markers():
        task = (marker["dataset"], int(marker["seed"]))
        tasks.setdefault(task, set()).add(int(marker["pid"]))
    return tasks


def reserved_gpus():
    # Allow lightweight external services when the free-memory threshold passes.
    # Only jobs launched by this queue are explicitly reserved.
    reserved = set()
    for marker in active_markers():
        reserved.add(int(marker["gpu"]))
    return reserved


def free_gpus():
    if not driver_inventory_ok():
        log(f"nvidia driver inventory unhealthy; wait {POLL_SECONDS}s")
        return []
    candidates = []
    reserved = reserved_gpus()
    for idx in gpu_indices():
        if idx in reserved:
            continue
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu_result.returncode != 0:
            log(f"skip gpu={idx}; nvidia-smi query failed")
            continue
        try:
            free_mib, util = [int(value.strip()) for value in gpu_result.stdout.split(",")[:2]]
        except Exception:
            log(f"skip gpu={idx}; cannot parse memory query: {gpu_result.stdout.strip()}")
            continue
        if free_mib < MIN_FREE_MIB:
            log(f"skip gpu={idx}; free={free_mib}MiB < min_free={MIN_FREE_MIB}MiB")
            continue
        if not torch_cuda_ok(idx):
            log(f"skip gpu={idx}; torch cuda probe failed")
            continue
        candidates.append((idx, free_mib, util))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [idx for idx, _, _ in candidates]


def launch_detached(dataset, seed, gpu):
    cfg_path = cfg_for(dataset)
    log_path = REPO / f".evidence_gate_m1_{dataset}_seed{seed}_gpu{gpu}.stdout"
    cmd = (
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        f"exec {shlex.quote(PYTHON)} -m fraudGT.main "
        f"--cfg {cfg_path} --repeat 1 --gpu 0 "
        f"out_dir {PRIMARY_OUT_DIR} seed {seed} "
        "train.tqdm False val.tqdm False "
        "train.auto_resume True train.epoch_resume -1"
    )
    log(f"launch dataset={dataset} seed={seed} gpu={gpu} cfg={cfg_path}")
    with log_path.open("ab") as output:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_marker(dataset, seed, gpu, proc.pid, log_path)
    return proc.pid


def runnable_tasks():
    active = active_tasks()
    return [
        (dataset, seed)
        for dataset, seed in pending_tasks()
        if (dataset, seed) not in active
    ]


def main():
    gpu_scope = (
        ",".join(str(gpu) for gpu in sorted(ALLOWED_GPU_IDS))
        if ALLOWED_GPU_IDS is not None
        else "all"
    )
    log(f"evidence_gate M1 queue started gpu_ids={gpu_scope}")
    while True:
        reap_children()

        pending = pending_tasks()
        if not pending:
            log("evidence_gate M1 queue finished")
            return 0

        if legacy_queue_alive():
            log(
                f"legacy seed42 queue active; pending={len(pending)}; "
                f"wait {POLL_SECONDS}s"
            )
            time.sleep(POLL_SECONDS)
            continue

        runnable = runnable_tasks()
        if not runnable:
            active = sorted(f"{dataset}:seed{seed}" for dataset, seed in active_tasks())
            log(f"no runnable task; pending={len(pending)} active={active}; wait {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
            continue

        free = free_gpus()
        if not free:
            dataset, seed = runnable[0]
            log(f"no idle usable gpu; pending={len(pending)} next={dataset}:seed{seed}; wait {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
            continue

        launched = []
        active = active_tasks()
        for gpu in free:
            runnable = [
                (dataset, seed)
                for dataset, seed in pending_tasks()
                if (dataset, seed) not in active
            ]
            if not runnable:
                break
            dataset, seed = runnable[0]
            pid = launch_detached(dataset, seed, gpu)
            active[(dataset, seed)] = {pid}
            launched.append(f"{dataset}:seed{seed}@gpu{gpu}/pid{pid}")

        if launched:
            log(f"launched {len(launched)} job(s): {', '.join(launched)}")
            time.sleep(min(LAUNCH_SETTLE_SECONDS, POLL_SECONDS))
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
