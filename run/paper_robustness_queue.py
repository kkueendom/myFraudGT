#!/usr/bin/env python3
"""Queue robustness runs for the paper-facing scale-adaptive result.

This script is intentionally conservative:

- It does not kill any process.
- It only launches when a GPU has enough free memory and no fraudGT.main job.
- It runs the weakest robustness checks first.
"""

import argparse
import subprocess
import time
from pathlib import Path


REPO = Path("/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining")
CONDA_ENV = "fraudgt_dual_gate"

STAGES = [
    [
        {
            "name": "smallhi_gate_seed44",
            "cfg": "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml",
            "stats": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed44_screen180_main/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44-gpu0/44/test/stats.json",
            "gpu": 0,
            "done_epoch": 179,
            "min_free_mib": 12000,
        },
        {
            "name": "smallhi_gate_seed45",
            "cfg": "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed45.yaml",
            "stats": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed45_screen180_main/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed45-gpu1/45/test/stats.json",
            "gpu": 1,
            "done_epoch": 179,
            "min_free_mib": 12000,
        },
    ],
    [
        {
            "name": "largehi_fallback_seed44",
            "cfg": "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed44.yaml",
            "stats": "/e/yyk/FraudGT_multi6/classmixprotobound_largehi_seed44_full240_main/AML-Large-HI-ClassMixProtoBoundResid240Seed44-gpu0/44/test/stats.json",
            "gpu": 0,
            "done_epoch": 239,
            "min_free_mib": 12000,
        },
        {
            "name": "largehi_fallback_seed45",
            "cfg": "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed45.yaml",
            "stats": "/e/yyk/FraudGT_multi6/classmixprotobound_largehi_seed45_full240_main/AML-Large-HI-ClassMixProtoBoundResid240Seed45-gpu1/45/test/stats.json",
            "gpu": 1,
            "done_epoch": 239,
            "min_free_mib": 12000,
        },
    ],
    [
        {
            "name": "largeli_fallback_seed44",
            "cfg": "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed44.yaml",
            "stats": "/e/yyk/FraudGT_multi6/classmixprotobound_largeli_seed44_full240_main/AML-Large-LI-ClassMixProtoBoundResid240Seed44-gpu0/44/test/stats.json",
            "gpu": 0,
            "done_epoch": 239,
            "min_free_mib": 12000,
        },
        {
            "name": "largeli_fallback_seed45",
            "cfg": "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed45.yaml",
            "stats": "/e/yyk/FraudGT_multi6/classmixprotobound_largeli_seed45_full240_main/AML-Large-LI-ClassMixProtoBoundResid240Seed45-gpu1/45/test/stats.json",
            "gpu": 1,
            "done_epoch": 239,
            "min_free_mib": 12000,
        },
    ],
]


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def shell_stdout(cmd):
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def active_lines():
    return [
        line
        for line in shell_stdout("ps -eo pid=,args=").splitlines()
        if "python -m fraudGT.main" in line
    ]


def active_for_cfg(cfg):
    return any(cfg in line for line in active_lines())


def gpu_free_mib():
    out = shell_stdout(
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    )
    free = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            free[int(parts[0])] = int(parts[1])
    return free


def gpu_has_fraudgt(gpu):
    needles = (f"--gpu {gpu}", f"--gpu={gpu}")
    return any(any(needle in line for needle in needles) for line in active_lines())


def stats_complete(job):
    stats_path = job["stats"]
    path = Path(stats_path)
    seed_dir = path.parents[1]
    train_stats = seed_dir / "train" / "stats.json"
    if not train_stats.exists():
        return False
    last_epoch = None
    for line in train_stats.read_text(errors="ignore").splitlines():
        if '"epoch"' not in line:
            continue
        try:
            last_epoch = int(line.split('"epoch"')[1].split(":")[1].split(",")[0])
        except Exception:
            continue
    return last_epoch is not None and last_epoch >= job["done_epoch"]


def launch(job, gpu):
    log_path = REPO / f".paper_robustness_{job['name']}_gpu{gpu}.stdout"
    pid_path = REPO / f".paper_robustness_{job['name']}_gpu{gpu}.pid"
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"conda activate {CONDA_ENV}; "
        f"cd {REPO}; "
        f"python -m fraudGT.main --cfg {job['cfg']} --repeat 1 --gpu {gpu}"
    )
    with log_path.open("w") as out:
        proc = subprocess.Popen(["bash", "-lc", cmd], stdout=out, stderr=subprocess.STDOUT)
    pid_path.write_text(str(proc.pid))
    log(f"launched {job['name']} gpu={gpu} pid={proc.pid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=300)
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=None,
        help="Override per-job free-memory thresholds.",
    )
    args = parser.parse_args()

    log("starting paper robustness queue")
    for stage_idx, stage in enumerate(STAGES, start=1):
        log(f"stage {stage_idx} pending: {', '.join(job['name'] for job in stage)}")
        while True:
            pending = [
                job
                for job in stage
                if not stats_complete(job) and not active_for_cfg(job["cfg"])
            ]
            active = [job for job in stage if active_for_cfg(job["cfg"])]
            if not pending and not active:
                log(f"stage {stage_idx} complete or already finished")
                break
            free = gpu_free_mib()
            for job in list(pending):
                gpu = job["gpu"]
                min_free = args.min_free_mib or job["min_free_mib"]
                if free.get(gpu, 0) < min_free or gpu_has_fraudgt(gpu):
                    continue
                launch(job, gpu)
                free[gpu] = 0
                pending.remove(job)
            if pending or active:
                log(
                    f"stage {stage_idx} waiting; pending={len(pending)} active={len(active)} "
                    f"free={free}"
                )
                time.sleep(args.poll)
    log("paper robustness queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
