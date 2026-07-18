#!/usr/bin/env python3
"""Two-dataset, 500-epoch formal queue for CPAR-K4.

The queue never uses GPU0. By default it may use GPU1-6 and treats processes
whose command contains ``ocr`` as non-blocking, while retaining a free-memory
guard. Set CPAR_GPU_ALLOWLIST to pin this pair beside other concurrent screens.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(os.environ.get(
    "CPAR_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
PYTHON = os.environ.get(
    "FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
MAX_EPOCH = int(os.environ.get("CPAR_MAX_EPOCH", "500"))
DONE_EPOCH = MAX_EPOCH - 1
POLL_SECONDS = int(os.environ.get("CPAR_POLL_SECONDS", "1200"))
LAUNCH_SETTLE_SECONDS = int(os.environ.get(
    "CPAR_LAUNCH_SETTLE_SECONDS", "20"))
MIN_FREE_MIB = int(os.environ.get("CPAR_MIN_FREE_MIB", "9000"))
MAX_UTIL = int(os.environ.get("CPAR_MAX_UTIL", "15"))
GPU_ALLOWLIST = tuple(int(item) for item in os.environ.get(
    "CPAR_GPU_ALLOWLIST", "1,2,3,4,5,6").split(",") if item.strip())
IGNORE_PROCESS_PATTERNS = tuple(item.strip().lower() for item in os.environ.get(
    "CPAR_IGNORE_PROCESS_PATTERNS", "ocr").split(",") if item.strip())
EXPECTED_BRANCH = os.environ.get(
    "CPAR_EXPECTED_BRANCH", "feature/cpar-k4-counterfactual-action-router")
OUT_DIR = Path(os.environ.get(
    "CPAR_OUT_DIR", str(REPO / "results" / "cpar_k4_formal500")))
EVENTS = REPO / ".cpar_k4_formal500_queue.events"
ACTIVE_DIR = REPO / ".cpar_k4_formal500_active"
FAILED_DIR = REPO / ".cpar_k4_formal500_failed"


TASKS = [
    ("Large-LI", 44),
    ("Small-LI", 42),
]


def command_output(args):
    result = subprocess.run(
        args, cwd=str(REPO), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stderr.strip()}")
    return result.stdout.strip()


def verify_version():
    branch = command_output(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"expected branch {EXPECTED_BRANCH!r}, found {branch!r}")
    dirty = command_output(["git", "status", "--porcelain"])
    if dirty:
        raise RuntimeError(
            "refusing to launch from a dirty worktree:\n" + dirty)
    return command_output(["git", "rev-parse", "--short=8", "HEAD"])


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


def run_stem(dataset, seed, git_sha):
    return (
        f"AML-{dataset}-CPARK4Formal500-full-Seed{seed}-{git_sha}")


def run_dirs(dataset, seed, git_sha):
    return sorted(OUT_DIR.glob(run_stem(dataset, seed, git_sha) + "-gpu*"))


def task_done(dataset, seed, git_sha):
    for run_dir in run_dirs(dataset, seed, git_sha):
        seed_dir = run_dir / str(seed)
        train = rows(seed_dir / "train" / "stats.json")
        val = rows(seed_dir / "val" / "stats.json")
        test = rows(seed_dir / "test" / "stats.json")
        if train and val and test and int(train[-1]["epoch"]) >= DONE_EPOCH:
            return True
    return False


def marker_path(dataset, seed, gpu):
    return ACTIVE_DIR / f"{dataset}_seed{seed}_gpu{gpu}.json"


def failed_path(dataset, seed):
    return FAILED_DIR / f"{dataset}_seed{seed}.json"


def write_marker(dataset, seed, gpu, pid, log_path, git_sha):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(dataset, seed, gpu).write_text(json.dumps({
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "pid": pid,
        "log": str(log_path),
        "git_sha": git_sha,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, sort_keys=True))


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        text=True, capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def active_markers(git_sha):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    markers = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            marker = json.loads(path.read_text())
            key = (marker["dataset"], int(marker["seed"]))
            pid = int(marker["pid"])
        except Exception:
            log(f"invalid marker retained for review: {path}")
            continue
        if marker.get("git_sha") != git_sha:
            raise RuntimeError(
                f"active marker belongs to another commit: {path}")
        if task_done(*key, git_sha):
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
                f"task failed dataset={key[0]} seed={key[1]} pid={pid}; "
                "no automatic retry")
            continue
        markers.append(marker)
    return markers


def failed_keys(git_sha):
    output = set()
    for path in FAILED_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            if item.get("git_sha") == git_sha:
                output.add((item["dataset"], int(item["seed"])))
        except Exception:
            continue
    return output


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


def ignored_process(pid):
    if not IGNORE_PROCESS_PATTERNS:
        return False
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        text=True, capture_output=True, check=False)
    command = result.stdout.strip().lower()
    return any(pattern in command for pattern in IGNORE_PROCESS_PATTERNS)


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
    for gpu in GPU_ALLOWLIST:
        if gpu == 0 or gpu in active_gpus:
            continue
        compute_pids = gpu_compute_pids(gpu)
        blockers = [pid for pid in compute_pids if not ignored_process(pid)]
        if blockers:
            continue
        free, util = gpu_free_and_util(gpu)
        ignored_only = bool(compute_pids) and not blockers
        if free >= MIN_FREE_MIB and (ignored_only or util <= MAX_UTIL):
            output.append(gpu)
    return output


def pending_tasks(markers, git_sha):
    active = {
        (item["dataset"], int(item["seed"])) for item in markers}
    failed = failed_keys(git_sha)
    return [
        task for task in TASKS
        if not task_done(*task, git_sha) and task not in active and
        task not in failed
    ]


def launch(dataset, seed, gpu, git_sha):
    run_name = run_stem(dataset, seed, git_sha)
    stdout_path = REPO / f".cpar_k4_formal500_{run_name}_gpu{gpu}.log"
    cmd = [
        PYTHON, "-m", "fraudGT.main",
        "--cfg", f"configs/evidence_gate_v4/AML-{dataset}.yaml",
        "--repeat", "1", "--gpu", "0",
        "out_dir", str(OUT_DIR),
        "name_tag", f"CPARK4Formal500-full-Seed{seed}-{git_sha}",
        "seed", str(seed),
        "optim.max_epoch", str(MAX_EPOCH),
        "train.early_stop", "False",
        "train.tqdm", "False",
        "val.tqdm", "False",
        "train.auto_resume", "False",
        "model.edge_decoding", "cpar_k4",
        "model.dmprd_num_slots", "4",
        "model.dmprd_use_distribution_stats", "False",
        "model.dmprd_use_reliability_gate", "False",
        "model.dmprd_delta_max", "1.0",
        "model.dmprd_beta_max", "1.0",
        "model.cpar_aux_weight", "0.05",
        "model.cpar_aux_start_epoch", "10",
        "model.cpar_aux_end_epoch", "150",
        "model.cpar_gap_floor", "0.0001",
        "model.cpar_gap_threshold", "0.05",
        "model.cpar_router_hidden", "32",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    with stdout_path.open("ab") as handle:
        process = subprocess.Popen(
            cmd, cwd=str(REPO), env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    write_marker(dataset, seed, gpu, process.pid, stdout_path, git_sha)
    log(
        f"launch dataset={dataset} seed={seed} gpu={gpu} "
        f"pid={process.pid} commit={git_sha}")
    time.sleep(LAUNCH_SETTLE_SECONDS)


def main():
    git_sha = verify_version()
    log(
        f"CPAR-K4 pair queue start commit={git_sha} max_epoch={MAX_EPOCH} "
        f"tasks={len(TASKS)} gpus={GPU_ALLOWLIST} poll={POLL_SECONDS}s")
    last_wait_state = None
    while True:
        markers = active_markers(git_sha)
        pending = pending_tasks(markers, git_sha)
        if not pending and not markers:
            failures = len(failed_keys(git_sha))
            log(f"CPAR-K4 pair queue finished failures={failures}")
            return 1 if failures else 0

        launched = False
        for gpu in idle_gpus(markers):
            markers = active_markers(git_sha)
            pending = pending_tasks(markers, git_sha)
            if not pending:
                break
            launch(*pending[0], gpu, git_sha)
            launched = True
            markers = active_markers(git_sha)

        wait_state = (
            len(pending_tasks(markers, git_sha)),
            tuple(sorted(
                (item["dataset"], int(item["gpu"]))
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
