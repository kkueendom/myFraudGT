#!/usr/bin/env python3
"""Single-GPU queue for the evidence-gate v2 formal runs.

The queue is intentionally conservative:
- it waits for the older smoke/seed42 queue to finish, avoiding two yyk FraudGT
  jobs at once;
- it launches at most one training process;
- it uses a GPU only when enough free memory is available;
- it logs only queue-level events.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path("/e/yyk/FraudGT_evidence_gate_decoder")
EVENTS = REPO / ".evidence_gate_v2_queue.events"
LEGACY_QUEUE_PID = REPO / ".evidence_gate_queue.pid"
POLL_SECONDS = int(os.environ.get("EVIDENCE_GATE_V2_POLL_SECONDS", "1800"))
MIN_FREE_MIB = int(os.environ.get("EVIDENCE_GATE_MIN_FREE_MIB", "14000"))
PRIMARY_OUT_DIR = REPO / "results" / "evidence_gate_v2"
SEED42_OUT_DIR = REPO / "results" / "evidence_gate_seed42"
DONE_EPOCH = 239

DATASETS = [
    "Small-HI",
    "Small-LI",
    "Medium-HI",
    "Medium-LI",
    "Large-HI",
    "Large-LI",
]
SEEDS = [42, 43, 44, 45, 46]


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
    return f"configs/evidence_gate/AML-{dataset}.yaml"


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
            and int(train_rows[-1]["epoch"]) >= DONE_EPOCH
            and val_rows
            and test_rows
        ):
            return True
    return False


def task_done(dataset, seed):
    if seed_done_in(PRIMARY_OUT_DIR, dataset, seed):
        return True
    return seed == 42 and seed_done_in(SEED42_OUT_DIR, dataset, seed)


def pending_tasks():
    return [
        (dataset, seed)
        for seed in SEEDS
        for dataset in DATASETS
        if not task_done(dataset, seed)
    ]


def process_alive(pid):
    return subprocess.run(
        ["ps", "-p", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


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


def torch_cuda_ok(gpu):
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "conda activate fraudgt_dual_gate; "
        f"CUDA_VISIBLE_DEVICES={gpu} python - <<'PY'\n"
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


def own_fraudgt_process_active():
    result = subprocess.run(
        ["ps", "-u", "yyk", "-o", "args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return True
    for line in result.stdout.splitlines():
        if "python -m fraudGT.main" in line:
            return True
    return False


def free_gpus():
    if not driver_inventory_ok():
        log(f"nvidia driver inventory unhealthy; wait {POLL_SECONDS}s")
        return []
    candidates = []
    for idx in (0, 1):
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


def launch(dataset, seed, gpu):
    cfg_path = cfg_for(dataset)
    log_path = REPO / f".evidence_gate_v2_{dataset}_seed{seed}_gpu{gpu}.stdout"
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "conda activate fraudgt_dual_gate; "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python -m fraudGT.main --cfg {cfg_path} --repeat 1 --gpu 0 "
        f"out_dir {PRIMARY_OUT_DIR} seed {seed} "
        "train.tqdm False val.tqdm False"
    )
    log(f"launch dataset={dataset} seed={seed} gpu={gpu} cfg={cfg_path}")
    with log_path.open("ab") as output:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
    log(f"finish dataset={dataset} seed={seed} gpu={gpu} status={code} log={log_path.name}")
    return code


def main():
    log("evidence_gate v2 5-seed queue started")
    while True:
        pending = pending_tasks()
        if not pending:
            log("evidence_gate v2 5-seed queue finished")
            return 0

        dataset, seed = pending[0]
        if legacy_queue_alive():
            log(
                f"legacy seed42 queue active; pending={len(pending)} "
                f"next={dataset}:seed{seed}; wait {POLL_SECONDS}s"
            )
            time.sleep(POLL_SECONDS)
            continue

        if own_fraudgt_process_active():
            log(
                f"existing yyk FraudGT training active; pending={len(pending)} "
                f"next={dataset}:seed{seed}; wait {POLL_SECONDS}s"
            )
            time.sleep(POLL_SECONDS)
            continue

        free = free_gpus()
        if not free:
            log(f"no usable gpu; pending={len(pending)} next={dataset}:seed{seed}; wait {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
            continue

        code = launch(dataset, seed, free[0])
        if code != 0:
            log("nonzero exit; stop queue for inspection")
            return code


if __name__ == "__main__":
    raise SystemExit(main())
