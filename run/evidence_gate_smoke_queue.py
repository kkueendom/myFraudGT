#!/usr/bin/env python3
"""Single-GPU queue for the evidence_gate decoder.

This intentionally keeps polling sparse and logs compact. It waits for a GPU
with no compute process, runs two short smoke checks, then runs the six-dataset
seed42 screen if the smoke stage succeeds.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path("/e/yyk/FraudGT_evidence_gate_decoder")
EVENTS = REPO / ".evidence_gate_queue.events"
POLL_SECONDS = 1800

TASKS = [
    {
        "stage": "smoke",
        "dataset": "Small-HI",
        "cfg": "configs/evidence_gate/AML-Small-HI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_smoke",
        "overrides": "optim.max_epoch 4 train.iter_per_epoch 16 val.iter_per_epoch 16",
        "done_epoch": 3,
    },
    {
        "stage": "smoke",
        "dataset": "Large-LI",
        "cfg": "configs/evidence_gate/AML-Large-LI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_smoke",
        "overrides": "optim.max_epoch 4 train.iter_per_epoch 16 val.iter_per_epoch 16",
        "done_epoch": 3,
    },
    {
        "stage": "seed42",
        "dataset": "Small-HI",
        "cfg": "configs/evidence_gate/AML-Small-HI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
    {
        "stage": "seed42",
        "dataset": "Small-LI",
        "cfg": "configs/evidence_gate/AML-Small-LI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
    {
        "stage": "seed42",
        "dataset": "Medium-HI",
        "cfg": "configs/evidence_gate/AML-Medium-HI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
    {
        "stage": "seed42",
        "dataset": "Medium-LI",
        "cfg": "configs/evidence_gate/AML-Medium-LI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
    {
        "stage": "seed42",
        "dataset": "Large-HI",
        "cfg": "configs/evidence_gate/AML-Large-HI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
    {
        "stage": "seed42",
        "dataset": "Large-LI",
        "cfg": "configs/evidence_gate/AML-Large-LI.yaml",
        "out_dir": "/e/yyk/FraudGT_evidence_gate_decoder/results/evidence_gate_seed42",
        "overrides": "",
        "done_epoch": 239,
    },
]


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def free_gpus():
    if not driver_inventory_ok():
        log(f"nvidia driver inventory unhealthy; wait {POLL_SECONDS}s")
        return []
    free = []
    for idx in (0, 1):
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu_result.returncode != 0:
            log(f"skip gpu={idx}; nvidia-smi query failed")
            continue

        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc_result.returncode != 0:
            log(f"skip gpu={idx}; compute-app query failed")
            continue
        if proc_result.stdout.strip():
            continue
        if not torch_cuda_ok(idx):
            log(f"skip gpu={idx}; torch cuda probe failed")
            continue
        free.append(idx)
    return free


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


def run_dirs(task):
    stem = Path(task["cfg"]).stem
    root = Path(task["out_dir"])
    return sorted(root.glob(stem + "-gpu*"))


def task_done(task):
    for run_dir in run_dirs(task):
        seed_dir = run_dir / "42"
        train_rows = rows(seed_dir / "train" / "stats.json")
        val_rows = rows(seed_dir / "val" / "stats.json")
        test_rows = rows(seed_dir / "test" / "stats.json")
        if (
            train_rows
            and int(train_rows[-1]["epoch"]) >= task["done_epoch"]
            and val_rows
            and test_rows
        ):
            return True
    return False


def launch(task, gpu):
    dataset = task["dataset"]
    cfg_path = task["cfg"]
    log_path = REPO / f".evidence_gate_{task['stage']}_{dataset}_gpu{gpu}.stdout"
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "conda activate fraudgt_dual_gate; "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python -m fraudGT.main --cfg {cfg_path} --repeat 1 --gpu 0 "
        f"out_dir {task['out_dir']} "
        f"{task['overrides']} "
        "train.tqdm False val.tqdm False"
    )
    log(f"launch stage={task['stage']} dataset={dataset} gpu={gpu} cfg={cfg_path}")
    with log_path.open("ab") as output:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
    log(
        f"finish stage={task['stage']} dataset={dataset} "
        f"gpu={gpu} status={code} log={log_path.name}"
    )
    return code


def main():
    log("evidence_gate smoke+seed42 queue started")
    while True:
        pending = [task for task in TASKS if not task_done(task)]
        if not pending:
            log("evidence_gate smoke+seed42 queue finished")
            return 0

        free = free_gpus()
        if not free:
            next_task = pending[0]
            log(
                f"no free gpu; pending={len(pending)} "
                f"next={next_task['stage']}:{next_task['dataset']}; "
                f"wait {POLL_SECONDS}s"
            )
            time.sleep(POLL_SECONDS)
            continue

        code = launch(pending[0], free[0])
        if code != 0:
            log("nonzero exit; stop queue for inspection")
            return code


if __name__ == "__main__":
    raise SystemExit(main())
