#!/usr/bin/env python3
"""Queue original FraudGT baseline runs for raw best test F1.

This queue uses the repository's original SparseNodeGT+ports+Ego configs and
only rewrites environment-specific paths, seed, and wandb logging. It does not
enable any of the added decoder evidence modules.
"""

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path


REPO = Path("/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining")
CONDA_ENV = "fraudgt_dual_gate"
RESULT_ROOT = "/e/yyk/FraudGT_multi6/original_raw_baseline"
DATA_ROOT = "/e/yyk/data/archive"
CONFIG_DIR = REPO / ".original_raw_baseline_configs"

SEEDS = [42, 43, 44]
DATASETS = ["Small-HI", "Small-LI", "Medium-HI", "Medium-LI", "Large-HI", "Large-LI"]
MAX_EPOCH = 500


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{ts()}] {message}", flush=True)


def shell_stdout(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True, check=False).stdout


def replace_line(text, prefix, replacement):
    pattern = re.compile(rf"^{re.escape(prefix)}.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"missing config line prefix: {prefix}")
    return pattern.sub(replacement, text, count=1)


def dataset_template(dataset):
    return REPO / "configs" / f"AML-{dataset}" / f"AML-{dataset}-SparseNodeGT+ports+Ego.yaml"


def stem(dataset, seed):
    return f"AML-{dataset}-OriginalSparseNodeGTPortsEgoRawSeed{seed}"


def generated_cfg(dataset, seed):
    CONFIG_DIR.mkdir(exist_ok=True)
    cfg_path = CONFIG_DIR / f"{stem(dataset, seed)}.yaml"
    text = dataset_template(dataset).read_text()
    text = replace_line(text, "out_dir:", f"out_dir: {RESULT_ROOT}")
    text = replace_line(text, "seed:", f"seed: {seed}")
    text = replace_line(text, "  dir:", f"  dir: {DATA_ROOT}")
    text = text.replace("wandb:\n  use: True", "wandb:\n  use: False")
    cfg_path.write_text(text)
    return cfg_path


def run_dirs(dataset, seed):
    return sorted(Path(RESULT_ROOT).glob(f"{stem(dataset, seed)}-gpu*"))


def exit_paths(dataset, seed):
    return sorted(REPO.glob(f".original_raw_baseline_{stem(dataset, seed)}_gpu*.exit"))


def terminal_attempt(dataset, seed):
    return bool(exit_paths(dataset, seed))


def read_last_epoch(stats_path):
    if not stats_path.exists():
        return None
    last = None
    for line in stats_path.read_text(errors="ignore").splitlines():
        match = re.search(r'"epoch"\s*:\s*(\d+)', line)
        if match:
            last = int(match.group(1))
    return last


def job_complete(dataset, seed):
    for run_dir in run_dirs(dataset, seed):
        last = read_last_epoch(run_dir / str(seed) / "train" / "stats.json")
        if last is not None and last >= MAX_EPOCH - 1:
            return True
    return False


def active_lines():
    return [
        line
        for line in shell_stdout("ps -eo pid=,args=").splitlines()
        if "python -m fraudGT.main" in line
    ]


def active_for_cfg(cfg_path):
    return any(str(cfg_path) in line for line in active_lines())


def gpu_state():
    out = shell_stdout(
        "nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    state = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            state[int(parts[0])] = {"free_mib": int(parts[1]), "util": int(parts[2])}
    return state


def launch(dataset, seed, gpu, cfg_path):
    item_stem = stem(dataset, seed)
    log_path = REPO / f".original_raw_baseline_{item_stem}_gpu{gpu}.stdout"
    pid_path = REPO / f".original_raw_baseline_{item_stem}_gpu{gpu}.pid"
    exit_path = REPO / f".original_raw_baseline_{item_stem}_gpu{gpu}.exit"
    q = shlex.quote
    command = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"conda activate {CONDA_ENV} && "
        f"cd {q(str(REPO))} && "
        f"python -m fraudGT.main --cfg {q(str(cfg_path))} --repeat 1 --gpu {gpu}; "
        "status=$?; "
        f"printf '%s\\n' \"$status\" > {q(str(exit_path))}; "
        "exit \"$status\""
    )
    with log_path.open("a") as out:
        out.write(f"\n===== launch {ts()} dataset={dataset} seed={seed} gpu={gpu} =====\n")
        out.flush()
        proc = subprocess.Popen(["bash", "-lc", command], stdout=out, stderr=subprocess.STDOUT)
    pid_path.write_text(str(proc.pid))
    log(f"launched dataset={dataset} seed={seed} gpu={gpu} pid={proc.pid}")


def classify_jobs():
    pending = []
    blocked = []
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg_path = generated_cfg(dataset, seed)
            if job_complete(dataset, seed) or active_for_cfg(cfg_path):
                continue
            if terminal_attempt(dataset, seed):
                blocked.append((dataset, seed, cfg_path))
                continue
            pending.append((dataset, seed, cfg_path))
    return pending, blocked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=300)
    parser.add_argument("--min-free-mib", type=int, default=18000)
    parser.add_argument("--max-util", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log("starting original raw baseline queue")
    while True:
        pending, blocked = classify_jobs()
        active = active_lines()
        if not pending and not active:
            if blocked:
                log(f"queue stopped with blocked terminal attempts={len(blocked)}")
                return 2
            log("original raw baseline queue finished")
            return 0

        state = gpu_state()
        launched = 0
        for gpu, gpu_info in sorted(state.items()):
            if launched >= len(pending):
                break
            if gpu_info["free_mib"] < args.min_free_mib or gpu_info["util"] > args.max_util:
                continue
            if any(f"--gpu {gpu}" in line or f"--gpu={gpu}" in line for line in active):
                continue
            dataset, seed, cfg_path = pending[launched]
            launch(dataset, seed, gpu, cfg_path)
            launched += 1

        log(
            f"waiting; pending={len(pending)} blocked={len(blocked)} "
            f"active={len(active)} gpu_state={state} launched={launched}"
        )
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
