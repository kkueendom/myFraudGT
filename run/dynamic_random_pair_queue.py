#!/usr/bin/env python3
"""Run a commit-locked dynamic-random Small-LI/Large-LI screen."""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(os.environ.get(
    "DYNAMIC_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
SPEC_PATH = Path(os.environ.get(
    "DYNAMIC_SPEC", str(REPO / "run" / "dynamic_random_pair_spec.json")))
SPEC = json.loads(SPEC_PATH.read_text())
METHOD_TAG = str(SPEC["method_tag"])
EXPECTED_BRANCH = str(SPEC["expected_branch"])
TASKS = [(str(dataset), int(seed)) for dataset, seed in SPEC["tasks"]]
OVERRIDES = [str(item) for item in SPEC["overrides"]]
CONFIG_TEMPLATE = str(SPEC.get(
    "config_template", "configs/evidence_gate_v4/AML-{dataset}.yaml"))

PYTHON = os.environ.get(
    "FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
MAX_EPOCH = int(os.environ.get("DYNAMIC_MAX_EPOCH", "500"))
DONE_EPOCH = MAX_EPOCH - 1
POLL_SECONDS = int(os.environ.get("DYNAMIC_POLL_SECONDS", "1800"))
GPU_ALLOWLIST = tuple(int(item) for item in os.environ.get(
    "DYNAMIC_GPU_ALLOWLIST", "1,2,3,4,5,6").split(",") if item.strip())
OUT_BASE = Path(os.environ.get(
    "DYNAMIC_OUT_BASE", str(REPO / "results"))).resolve()
RUNTIME_BASE = Path(os.environ.get(
    "DYNAMIC_RUNTIME_BASE", str(REPO.parent / "FraudGT_dynamic_runtime"))).resolve()
OUT_DIR = None
EVENTS = None
ACTIVE_DIR = None
FAILED_DIR = None


def command_output(args):
    result = subprocess.run(
        args, cwd=str(REPO), text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout.strip()


def protocol_audit():
    if SPEC.get("sampling_protocol") != "dynamic_random":
        raise RuntimeError("spec must declare sampling_protocol=dynamic_random")
    if len(OVERRIDES) % 2:
        raise RuntimeError("overrides must be key/value pairs")
    pairs = dict(zip(OVERRIDES[::2], OVERRIDES[1::2]))
    if pairs.get("val.fixed_target_panel", "").lower() != "false":
        raise RuntimeError("val.fixed_target_panel must be explicitly False")
    forbidden = {"val.fixed_panel_seed", "val.iter_per_epoch", "train.batch_size"}
    if forbidden.intersection(pairs):
        raise RuntimeError("spec overrides a protected sampling setting")
    sampler = (REPO / "fraudGT" / "sampler" / "custom_sampler.py").read_text()
    loader = (REPO / "fraudGT" / "graphgym" / "loader.py").read_text()
    if any(token in sampler for token in (
            "_fixed_target_panel", "reset_generator", "generator=reset_generator")):
        raise RuntimeError("fixed-panel sampler logic detected")
    if "shuffle=shuffle" not in sampler:
        raise RuntimeError("LinkNeighborLoader no longer forwards shuffle")
    if "def create_loader(dataset = None, shuffle = True" not in loader:
        raise RuntimeError("create_loader no longer defaults to shuffle=True")


def verify_version():
    if command_output(["git", "branch", "--show-current"]) != EXPECTED_BRANCH:
        raise RuntimeError("unexpected Git branch")
    if command_output(["git", "status", "--porcelain"]):
        raise RuntimeError("refusing to run from a dirty worktree")
    protocol_audit()
    return command_output(["git", "rev-parse", "--short=8", "HEAD"])


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def configure_paths(commit):
    global OUT_DIR, EVENTS, ACTIVE_DIR, FAILED_DIR
    namespace = f"{slug(METHOD_TAG)}_{commit}_dynamic500"
    OUT_DIR = OUT_BASE / namespace
    RUNTIME_BASE.mkdir(parents=True, exist_ok=True)
    EVENTS = RUNTIME_BASE / f".{namespace}.events"
    ACTIVE_DIR = RUNTIME_BASE / f".{namespace}_active"
    FAILED_DIR = RUNTIME_BASE / f".{namespace}_failed"


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def rows(path):
    output = []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "epoch" in row:
                output.append(row)
    return output


def run_stem(dataset, seed, commit):
    return f"AML-{dataset}-{METHOD_TAG}Dynamic500-Seed{seed}-{commit}"


def run_dirs(dataset, seed, commit):
    return sorted(OUT_DIR.glob(run_stem(dataset, seed, commit) + "-gpu*"))


def task_done(dataset, seed, commit):
    for run_dir in run_dirs(dataset, seed, commit):
        seed_dir = run_dir / str(seed)
        train = rows(seed_dir / "train" / "stats.json")
        val = rows(seed_dir / "val" / "stats.json")
        test = rows(seed_dir / "test" / "stats.json")
        if train and val and test and int(train[-1]["epoch"]) >= DONE_EPOCH:
            return True
    return False


def marker_path(dataset, seed, gpu):
    return ACTIVE_DIR / f"{dataset}_seed{seed}_gpu{gpu}.json"


def process_alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], text=True,
        capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def active_markers(commit):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    active = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        marker = json.loads(path.read_text())
        key = (marker["dataset"], int(marker["seed"]))
        if marker.get("git_commit") != commit:
            raise RuntimeError(f"stale marker from another commit: {path}")
        if task_done(*key, commit):
            path.unlink()
        elif process_alive(int(marker["pid"])):
            active.append(marker)
        else:
            marker["failure"] = "process_exited_before_completion"
            (FAILED_DIR / path.name).write_text(json.dumps(marker, sort_keys=True))
            path.unlink()
            log(f"failed dataset={key[0]} seed={key[1]}")
    return active


def gpu_idle(gpu):
    if gpu == 0:
        return False
    query = subprocess.run([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    if query.returncode or not query.stdout.strip():
        return False
    free, util = (int(item.strip()) for item in query.stdout.split(",")[:2])
    procs = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    return free >= 9000 and util <= 15 and not procs.stdout.strip()


def failed_keys():
    output = set()
    for path in FAILED_DIR.glob("*.json"):
        item = json.loads(path.read_text())
        output.add((item["dataset"], int(item["seed"])))
    return output


def pending_tasks(active, commit):
    active_keys = {(item["dataset"], int(item["seed"])) for item in active}
    failed = failed_keys()
    return [task for task in TASKS if task not in active_keys and
            task not in failed and not task_done(*task, commit)]


def launch(dataset, seed, gpu, commit):
    run_name = run_stem(dataset, seed, commit)
    stdout_path = RUNTIME_BASE / f".{run_name}_gpu{gpu}.log"
    config_path = CONFIG_TEMPLATE.format(dataset=dataset)
    cmd = [
        PYTHON, "-m", "fraudGT.main", "--cfg",
        config_path,
        "--repeat", "1", "--gpu", "0", "out_dir", str(OUT_DIR),
        "name_tag", f"{METHOD_TAG}Dynamic500-Seed{seed}-{commit}",
        "seed", str(seed), "optim.max_epoch", str(MAX_EPOCH),
        "train.early_stop", "False", "train.tqdm", "False",
        "val.tqdm", "False", "train.auto_resume", "False",
    ] + OVERRIDES
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "disabled",
    })
    with stdout_path.open("ab") as handle:
        process = subprocess.Popen(
            cmd, cwd=str(REPO), env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(dataset, seed, gpu).write_text(json.dumps({
        "dataset": dataset, "variant": SPEC.get("variant", METHOD_TAG),
        "seed": seed, "gpu": gpu, "pid": process.pid,
        "git_commit": commit, "config": config_path,
        "log": str(stdout_path), "sampling_protocol": "dynamic_random",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, sort_keys=True))
    log(f"launch dataset={dataset} seed={seed} gpu={gpu} pid={process.pid}")
    time.sleep(20)


def write_manifest(commit):
    manifest = OUT_DIR / "experiment_manifest.jsonl"
    subprocess.run([
        PYTHON, str(REPO / "run" / "dynamic_random_result_audit.py"),
        "--spec", str(SPEC_PATH), "--root", str(OUT_DIR),
        "--commit", commit, "--epoch-limit", str(DONE_EPOCH),
        "--write-manifest", str(manifest),
    ], cwd=str(REPO), check=True)


def main():
    commit = verify_version()
    configure_paths(commit)
    log(f"start method={METHOD_TAG} commit={commit} protocol=dynamic_random")
    last_state = None
    while True:
        active = active_markers(commit)
        pending = pending_tasks(active, commit)
        if not active and not pending:
            write_manifest(commit)
            log(f"finished failures={len(failed_keys())}")
            return 1 if failed_keys() else 0
        for gpu in GPU_ALLOWLIST:
            active = active_markers(commit)
            pending = pending_tasks(active, commit)
            if not pending:
                break
            if gpu not in {int(item["gpu"]) for item in active} and gpu_idle(gpu):
                launch(*pending[0], gpu, commit)
        active = active_markers(commit)
        state = (len(pending_tasks(active, commit)), tuple(sorted(
            (item["dataset"], item["gpu"]) for item in active)))
        if state != last_state:
            log(f"state pending={state[0]} active={state[1]}")
        last_state = state
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
