#!/usr/bin/env python3
"""Version-locked 500-epoch Small-LI/Large-LI queue for ACDR-Help."""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(os.environ.get(
    "ACDR_HELP_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
PYTHON = os.environ.get(
    "FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
MAX_EPOCH = int(os.environ.get("ACDR_HELP_MAX_EPOCH", "500"))
DONE_EPOCH = MAX_EPOCH - 1
POLL_SECONDS = int(os.environ.get("ACDR_HELP_POLL_SECONDS", "1200"))
LAUNCH_SETTLE_SECONDS = int(os.environ.get(
    "ACDR_HELP_LAUNCH_SETTLE_SECONDS", "20"))
MIN_FREE_MIB = int(os.environ.get("ACDR_HELP_MIN_FREE_MIB", "8500"))
MAX_UTIL = int(os.environ.get("ACDR_HELP_MAX_UTIL", "20"))
ALLOWED_GPUS = tuple(int(item) for item in os.environ.get(
    "ACDR_HELP_GPUS", "1,2,3,4,5,6").split(",") if item.strip())

TASKS = [
    ("Large-LI", 44),
    ("Small-LI", 42),
]


def command_output(command):
    result = subprocess.run(
        command, cwd=str(REPO), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}")
    return result.stdout.strip()


def source_revision():
    branch = command_output(["git", "branch", "--show-current"])
    revision = command_output(["git", "rev-parse", "--short=12", "HEAD"])
    dirty = command_output(["git", "status", "--porcelain"])
    if dirty:
        raise RuntimeError(
            "refusing to launch from a dirty worktree; commit method/config "
            "changes first")
    if branch != "feature/acdr-marginal-help-critic":
        raise RuntimeError(
            f"unexpected branch {branch!r}; expected "
            "'feature/acdr-marginal-help-critic'")
    return revision


REVISION = source_revision()
OUT_DIR = Path(os.environ.get(
    "ACDR_HELP_OUT_DIR",
    str(REPO / "results" / f"acdr_help_pair500_{REVISION}")))
EVENTS = REPO / f".acdr_help_pair500_{REVISION}.events"
ACTIVE_DIR = REPO / f".acdr_help_pair500_{REVISION}_active"
FAILED_DIR = REPO / f".acdr_help_pair500_{REVISION}_failed"


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


def run_stem(dataset, seed):
    return f"AML-{dataset}-ACDRHelpPair500-Seed{seed}-{REVISION}"


def run_dirs(dataset, seed):
    return sorted(OUT_DIR.glob(run_stem(dataset, seed) + "-gpu*"))


def task_done(dataset, seed):
    for run_dir in run_dirs(dataset, seed):
        seed_dir = run_dir / str(seed)
        train = rows(seed_dir / "train" / "stats.json")
        val = rows(seed_dir / "val" / "stats.json")
        test = rows(seed_dir / "test" / "stats.json")
        if (
                train and val and test and
                int(train[-1]["epoch"]) >= DONE_EPOCH):
            return True
    return False


def marker_path(dataset, seed, gpu):
    return ACTIVE_DIR / f"{dataset}_seed{seed}_gpu{gpu}.json"


def failed_path(dataset, seed):
    return FAILED_DIR / f"{dataset}_seed{seed}.json"


def write_marker(dataset, seed, gpu, process, log_path, command):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(dataset, seed, gpu).write_text(json.dumps({
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "pid": process.pid,
        "log": str(log_path),
        "revision": REVISION,
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, sort_keys=True))


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        text=True, capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def active_markers():
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    output = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            marker = json.loads(path.read_text())
            key = marker["dataset"], int(marker["seed"])
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
            failure["reason"] = "process exited before epoch 499"
            failed_path(*key).write_text(json.dumps(
                failure, indent=2, sort_keys=True))
            path.unlink(missing_ok=True)
            log(
                f"task failed dataset={key[0]} seed={key[1]} pid={pid}; "
                "no automatic retry")
            continue
        output.append(marker)
    return output


def failed_keys():
    output = set()
    for path in FAILED_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            output.add((item["dataset"], int(item["seed"])))
        except Exception:
            continue
    return output


def compute_pids(gpu):
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


def process_command(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        text=True, capture_output=True, check=False)
    return result.stdout.strip().lower()


def compute_occupancy(gpu):
    # OCR workers are explicitly non-blocking for this screen; the independent
    # free-memory check still prevents overcommitting a card they occupy.
    all_pids = compute_pids(gpu)
    blocking = [
        pid for pid in all_pids if "ocr" not in process_command(pid)]
    return blocking, bool(all_pids) and not blocking


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
    active = {int(item["gpu"]) for item in markers}
    output = []
    for gpu in ALLOWED_GPUS:
        if gpu == 0 or gpu in active:
            continue
        blocking, only_ocr = compute_occupancy(gpu)
        if blocking:
            continue
        free, util = gpu_free_and_util(gpu)
        if free >= MIN_FREE_MIB and (util <= MAX_UTIL or only_ocr):
            output.append(gpu)
    return output


def pending_tasks(markers):
    active = {(item["dataset"], int(item["seed"])) for item in markers}
    failed = failed_keys()
    return [
        task for task in TASKS
        if not task_done(*task) and task not in active and task not in failed
    ]


def launch(dataset, seed, gpu):
    run_name = run_stem(dataset, seed)
    stdout_path = REPO / f".acdr_help_pair500_{run_name}_gpu{gpu}.log"
    command = [
        PYTHON, "-m", "fraudGT.main",
        "--cfg", f"configs/acdr_help/AML-{dataset}.yaml",
        "--repeat", "1", "--gpu", "0",
        "out_dir", str(OUT_DIR),
        "name_tag", f"ACDRHelpPair500-Seed{seed}-{REVISION}",
        "seed", str(seed),
        "optim.max_epoch", str(MAX_EPOCH),
        "train.early_stop", "False",
        "train.tqdm", "False",
        "val.tqdm", "False",
        "train.auto_resume", "False",
        "model.edge_decoding", "acdr_help",
        "model.dmprd_num_slots", "4",
        "model.dmprd_use_distribution_stats", "False",
        "model.dmprd_use_reliability_gate", "False",
        "model.dmprd_delta_max", "1.0",
        "model.dmprd_beta_max", "1.0",
        "model.acdr_help_router_hidden", "32",
        "model.acdr_help_route_radius", "1.0",
        "model.acdr_help_aux_weight", "0.05",
        "model.acdr_help_aux_start_epoch", "10",
        "model.acdr_help_aux_end_epoch", "150",
        "model.acdr_help_low_dose", "0.75",
        "model.acdr_help_high_dose", "1.25",
        "model.acdr_help_deadzone", "0.0001",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["WANDB_MODE"] = "disabled"
    with stdout_path.open("ab") as handle:
        process = subprocess.Popen(
            command, cwd=str(REPO), env=environment, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    write_marker(dataset, seed, gpu, process, stdout_path, command)
    log(
        f"launch dataset={dataset} seed={seed} gpu={gpu} "
        f"pid={process.pid} revision={REVISION}")
    time.sleep(LAUNCH_SETTLE_SECONDS)


def main():
    log(
        f"ACDR-Help pair queue start revision={REVISION} "
        f"max_epoch={MAX_EPOCH} allowed_gpus={ALLOWED_GPUS} "
        f"poll={POLL_SECONDS}s")
    last_wait_state = None
    while True:
        markers = active_markers()
        pending = pending_tasks(markers)
        if not pending and not markers:
            failures = len(failed_keys())
            log(f"ACDR-Help pair queue finished failures={failures}")
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
                (item["dataset"], int(item["gpu"])) for item in markers)),
        )
        if not launched and wait_state != last_wait_state:
            log(
                f"waiting pending={wait_state[0]} active={len(markers)}; "
                f"next check in {POLL_SECONDS}s")
        last_wait_state = wait_state
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
