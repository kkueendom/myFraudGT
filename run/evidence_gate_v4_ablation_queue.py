#!/usr/bin/env python3
"""GPU queue for formal evidence-gate v4 ablations.

Launches only when a GPU is idle enough. It avoids duplicate dataset/seed/variant
runs in the ablation output dir and coexists with the remaining formal v4 runs.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(os.environ.get("FRAUDGT_V4_ABLATION_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
EVENTS = REPO / ".evidence_gate_v4_formal_ablation_queue.events"
ACTIVE_DIR = REPO / ".evidence_gate_v4_formal_ablation_active"
POLL_SECONDS = int(os.environ.get("EVIDENCE_GATE_V4_ABLATION_POLL_SECONDS", "600"))
LAUNCH_SETTLE_SECONDS = int(os.environ.get("EVIDENCE_GATE_V4_ABLATION_LAUNCH_SETTLE_SECONDS", "60"))
MIN_FREE_MIB = int(os.environ.get("EVIDENCE_GATE_V4_ABLATION_MIN_FREE_MIB", "14000"))
MAX_UTIL = int(os.environ.get("EVIDENCE_GATE_V4_ABLATION_MAX_UTIL", "15"))
GPU_IDS_RAW = os.environ.get("EVIDENCE_GATE_V4_ABLATION_GPU_IDS", "").strip()
ALLOWED_GPU_IDS = {int(x.strip()) for x in GPU_IDS_RAW.split(",") if x.strip()} if GPU_IDS_RAW else None
PYTHON = os.environ.get("FRAUDGT_PYTHON", "/d/miniconda3/envs/fraudGT/bin/python3.9")
OUT_DIR = Path(os.environ.get("EVIDENCE_GATE_V4_ABLATION_OUT_DIR", str(REPO / "results" / "evidence_gate_v4_formal_ablation")))
DONE_EPOCH = int(os.environ.get("EVIDENCE_GATE_V4_ABLATION_DONE_EPOCH", "499"))
DRY_RUN = os.environ.get("EVIDENCE_GATE_V4_ABLATION_DRY_RUN", "0") == "1"

DATASETS = ["Small-HI", "Small-LI", "Medium-HI", "Medium-LI", "Large-HI", "Large-LI"]
BEST_SEED_BY_DATASET = {
    "Small-HI": 42,
    "Small-LI": 42,
    "Medium-HI": 42,
    "Medium-LI": 44,
    "Large-HI": 43,
    "Large-LI": 44,
}

VARIANTS = [
    {
        "name": "proto_only",
        "overrides": ["model.edge_decoding", "evidence_gate_proto"],
        "purpose": "prototype residual only; removes structural residual and v4 router",
    },
    {
        "name": "no_gate",
        "overrides": ["model.edge_decoding", "evidence_gate_v4_nogate"],
        "purpose": "same structure and auxiliary loss, with an always-open gate",
    },
    {
        "name": "no_prototype",
        "overrides": ["model.edge_decoding", "evidence_gate_v4_noproto"],
        "purpose": "remove prototype residual, prototype-bank updates and router inputs",
    },
    {
        "name": "no_aux",
        "overrides": ["model.eg_struct_aux_weight", "0", "model.eg_struct_aux_epochs", "0"],
        "purpose": "remove early auxiliary supervision for structural expert",
    },
    {
        "name": "no_budget",
        "overrides": ["model.eg_gate_budget_weight", "0"],
        "purpose": "remove gate-budget regularization",
    },
    {
        "name": "weak_residual",
        "overrides": ["model.eg_struct_residual_scale", "0.5"],
        "purpose": "halve structural residual magnitude",
    },
    {
        "name": "strong_residual",
        "overrides": ["model.eg_struct_residual_scale", "1.5"],
        "purpose": "increase structural residual magnitude",
    },
]

STAGE_A_VARIANTS = ["proto_only", "no_gate", "no_prototype", "no_aux", "no_budget"]


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with EVENTS.open("a") as handle:
        handle.write(line + "\n")


def rows(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            out.append(row)
    return out


def cfg_for(dataset):
    return f"configs/evidence_gate_v4/AML-{dataset}.yaml"


def variant_by_name(name):
    for variant in VARIANTS:
        if variant["name"] == name:
            return variant
    raise KeyError(name)


def run_stem(dataset, variant, seed):
    return f"AML-{dataset}-V4FormalAblation-{variant}-Seed{seed}"


def run_dirs(dataset, variant, seed):
    return sorted(OUT_DIR.glob(run_stem(dataset, variant, seed) + "-gpu*"))


def task_done(dataset, variant, seed):
    for run_dir in run_dirs(dataset, variant, seed):
        train_rows = rows(run_dir / str(seed) / "train" / "stats.json")
        val_rows = rows(run_dir / str(seed) / "val" / "stats.json")
        test_rows = rows(run_dir / str(seed) / "test" / "stats.json")
        if train_rows and val_rows and test_rows and int(train_rows[-1]["epoch"]) >= DONE_EPOCH:
            return True
    return False


def tasks():
    names = os.environ.get("EVIDENCE_GATE_V4_ABLATION_VARIANTS", ",".join(STAGE_A_VARIANTS))
    variant_names = [x.strip() for x in names.split(",") if x.strip()]
    dataset_names = [x.strip() for x in os.environ.get("EVIDENCE_GATE_V4_ABLATION_DATASETS", ",".join(DATASETS)).split(",") if x.strip()]
    seed_policy = os.environ.get("EVIDENCE_GATE_V4_ABLATION_SEED_POLICY", "best")
    seeds_raw = os.environ.get("EVIDENCE_GATE_V4_ABLATION_SEEDS", "")
    fixed_seeds = [int(x.strip()) for x in seeds_raw.split(",") if x.strip()]
    pending = []
    for variant_name in variant_names:
        variant_by_name(variant_name)
        for dataset in dataset_names:
            if seed_policy == "best":
                seeds = [BEST_SEED_BY_DATASET[dataset]]
            else:
                seeds = fixed_seeds or [42]
            for seed in seeds:
                if not task_done(dataset, variant_name, seed):
                    pending.append((dataset, variant_name, seed))
    return pending


def process_alive(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True, check=False)
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


def marker_path(dataset, variant, seed, gpu):
    safe = f"{dataset}_{variant}_seed{seed}_gpu{gpu}".replace("/", "_")
    return ACTIVE_DIR / f"{safe}.json"


def write_marker(dataset, variant, seed, gpu, pid, log_path):
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(dataset, variant, seed, gpu).write_text(json.dumps({
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "pid": pid,
        "log": str(log_path),
        "out_dir": str(OUT_DIR),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, sort_keys=True))


def active_markers():
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    markers = []
    for path in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            marker = json.loads(path.read_text())
            dataset = marker["dataset"]
            variant = marker["variant"]
            seed = int(marker["seed"])
            pid = int(marker["pid"])
        except Exception:
            path.unlink(missing_ok=True)
            continue
        if task_done(dataset, variant, seed):
            path.unlink(missing_ok=True)
            continue
        if not process_alive(pid):
            log(f"stale marker removed dataset={dataset} variant={variant} seed={seed} pid={pid}; task will be retried")
            path.unlink(missing_ok=True)
            continue
        markers.append(marker)
    return markers


def gpu_indices():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
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
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except Exception:
            continue
    return pids


def gpu_free_and_util(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return 0, 100
    parts = [p.strip() for p in result.stdout.splitlines()[0].split(",")]
    return int(parts[0]), int(parts[1])


def idle_gpus(markers):
    active_gpus = {int(marker["gpu"]) for marker in markers}
    out = []
    for gpu in gpu_indices():
        if gpu in active_gpus:
            continue
        pids = gpu_compute_pids(gpu)
        if pids:
            continue
        free, util = gpu_free_and_util(gpu)
        if free < MIN_FREE_MIB:
            log(f"skip gpu={gpu}; free={free}MiB < min_free={MIN_FREE_MIB}MiB")
            continue
        if util > MAX_UTIL:
            log(f"skip gpu={gpu}; util={util}% > max_util={MAX_UTIL}%")
            continue
        out.append(gpu)
    return out


def active_task_keys(markers):
    return {(m["dataset"], m["variant"], int(m["seed"])) for m in markers}


def launch(dataset, variant_name, seed, gpu):
    variant = variant_by_name(variant_name)
    run_name = f"{run_stem(dataset, variant_name, seed)}-gpu{gpu}"
    log_path = REPO / f".evidence_gate_v4_formal_ablation_{run_name}.log"
    cmd = [
        PYTHON, "-m", "fraudGT.main",
        "--cfg", cfg_for(dataset),
        "--repeat", "1",
        "--gpu", "0",
        "out_dir", str(OUT_DIR),
        "name_tag", f"V4FormalAblation-{variant_name}-Seed{seed}",
        "seed", str(seed),
        "optim.max_epoch", "500",
        "train.early_stop", "False",
        "train.tqdm", "False",
        "val.tqdm", "False",
        "train.auto_resume", "True",
        "train.epoch_resume", "-1",
    ] + variant["overrides"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WANDB_MODE"] = "disabled"
    if DRY_RUN:
        log("dry-run " + " ".join(cmd))
        return None
    with log_path.open("ab") as handle:
        process = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_marker(dataset, variant_name, seed, gpu, process.pid, log_path)
    log(f"launch dataset={dataset} variant={variant_name} seed={seed} gpu={gpu} pid={process.pid}")
    time.sleep(LAUNCH_SETTLE_SECONDS)
    return process.pid


def main():
    log("v4 ablation queue start")
    while True:
        reap_children()
        markers = active_markers()
        pending = [t for t in tasks() if t not in active_task_keys(markers)]
        if not pending and not markers:
            log("all requested v4 ablations complete")
            return 0
        launched = []
        for gpu in idle_gpus(markers):
            pending = [t for t in tasks() if t not in active_task_keys(active_markers())]
            if not pending:
                break
            dataset, variant, seed = pending[0]
            pid = launch(dataset, variant, seed, gpu)
            if pid is not None:
                launched.append(f"{dataset}:{variant}:seed{seed}@gpu{gpu}/pid{pid}")
        if launched:
            log("launched " + ", ".join(launched))
        else:
            active = [f"{m['dataset']}:{m['variant']}:seed{m['seed']}" for m in markers]
            next_task = pending[0] if pending else None
            log(f"no launch; pending={len(pending)} next={next_task} active={active}; wait {POLL_SECONDS}s")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
