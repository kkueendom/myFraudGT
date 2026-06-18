#!/usr/bin/env python3
"""Prepare and optionally run clean paper ablations.

This queue is intentionally gated behind the formal mainline package. Running it
without --start only generates configs and prints the planned jobs. Running with
--start still waits until the 18 formal mainline runs are complete, unless
--ignore-formal-gate is explicitly supplied.
"""

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path


REPO = Path("/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining")
CONDA_ENV = "fraudgt_dual_gate"
ABLATION_ROOT = "/e/yyk/FraudGT_multi6/paper_ablation_clean_v1"
DATA_ROOT = "/e/yyk/data/archive"
FORMAL_ROOT = Path("/e/yyk/FraudGT_multi6/paper_formal_scale_adaptive_mainline")
CONFIG_DIR = REPO / ".paper_ablation_configs"

DATASETS = [
    "Small-HI",
    "Small-LI",
    "Medium-HI",
    "Medium-LI",
    "Large-HI",
    "Large-LI",
]

FORMAL_MAX_EPOCH = {
    "Small-HI": 180,
    "Small-LI": 240,
    "Medium-HI": 180,
    "Medium-LI": 180,
    "Large-HI": 240,
    "Large-LI": 240,
}

FORMAL_SEEDS = [42, 43, 44]

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
        "name": "base_dot",
        "template": "configs/AML-{dataset}/AML-{dataset}-SparseNodeGT+ports+Ego.yaml",
        "max_epoch": 240,
        "purpose": "Original FraudGT-style direct edge decoder baseline.",
    },
    {
        "name": "supportmix_consis_proto",
        "template": (
            "configs/AML-{dataset}/AML-{dataset}-SparseNodeGT+ports+Ego+"
            "WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBank"
            "WindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240"
            "UnifiedFullCalibMemSafe.yaml"
        ),
        "max_epoch": 240,
        "purpose": "Support-conditioned evidence plus class prototype context.",
    },
    {
        "name": "classmix_proto_bound",
        "template": (
            "configs/AML-{dataset}/AML-{dataset}-SparseNodeGT+ports+Ego+"
            "WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBank"
            "WindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProto"
            "BoundResid240UnifiedFullCalibMemSafe.yaml"
        ),
        "max_epoch": 240,
        "purpose": "Prototype/classmix bounded residual branch.",
    },
    {
        "name": "class_split",
        "template": "configs/AML-{dataset}/AML-{dataset}-UnifiedClassSplit240.yaml",
        "max_epoch": 240,
        "purpose": "Positive/negative class-split evidence branch.",
    },
    {
        "name": "full_dual_uncert_gate",
        "template": (
            "configs/AML-{dataset}/AML-{dataset}-"
            "UnifiedClassSplitSubgraphDualUncertGate180.yaml"
        ),
        "max_epoch": 180,
        "purpose": "Full high-order subgraph route with uncertainty gate.",
    },
]


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{ts()}] {message}", flush=True)


def shell_stdout(command):
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def replace_line(text, prefix, replacement):
    pattern = re.compile(rf"^{re.escape(prefix)}.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"missing config line prefix: {prefix}")
    return pattern.sub(replacement, text, count=1)


def disable_wandb(text):
    lines = text.splitlines()
    in_wandb = False
    found = False
    for index, line in enumerate(lines):
        if line == "wandb:":
            in_wandb = True
            continue
        if not in_wandb:
            continue
        if line and not line.startswith(" "):
            break
        if line.startswith("  use:"):
            lines[index] = "  use: False"
            found = True
            break
    if not found:
        raise ValueError("missing wandb.use config line")
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def train_last_epoch(stats_path):
    if not stats_path.exists():
        return None
    last = None
    for line in stats_path.read_text(errors="ignore").splitlines():
        match = re.search(r'"epoch"\s*:\s*(\d+)', line)
        if match:
            last = int(match.group(1))
    return last


def formal_complete():
    complete = 0
    total = len(DATASETS) * len(FORMAL_SEEDS)
    for dataset in DATASETS:
        max_epoch = FORMAL_MAX_EPOCH[dataset]
        for seed in FORMAL_SEEDS:
            stem = f"AML-{dataset}-PaperFormalScaleAdaptiveSeed{seed}"
            run_dirs = sorted(FORMAL_ROOT.glob(f"{stem}-gpu*"))
            done = False
            for run_dir in run_dirs:
                stats_path = run_dir / str(seed) / "train" / "stats.json"
                last = train_last_epoch(stats_path)
                if last is not None and last >= max_epoch - 1:
                    done = True
                    break
            complete += int(done)
    return complete, total


def variant_by_name(name):
    for variant in VARIANTS:
        if variant["name"] == name:
            return variant
    raise KeyError(name)


def cfg_path_for(dataset, variant, seed):
    CONFIG_DIR.mkdir(exist_ok=True)
    stem = f"AML-{dataset}-PaperAblation-{variant['name']}-Seed{seed}"
    cfg_path = CONFIG_DIR / f"{stem}.yaml"
    template = REPO / variant["template"].format(dataset=dataset)
    text = template.read_text()
    text = replace_line(text, "out_dir:", f"out_dir: {ABLATION_ROOT}")
    text = replace_line(text, "  dir:", f"  dir: {DATA_ROOT}")
    text = replace_line(text, "seed:", f"seed: {seed}")
    text = replace_line(text, "  max_epoch:", f"  max_epoch: {variant['max_epoch']}")
    text = disable_wandb(text)
    cfg_path.write_text(text)
    return cfg_path


def run_dirs(dataset, variant, seed):
    stem = f"AML-{dataset}-PaperAblation-{variant['name']}-Seed{seed}"
    return sorted(Path(ABLATION_ROOT).glob(f"{stem}-gpu*"))


def exit_paths(dataset, variant, seed):
    stem = f"AML-{dataset}-PaperAblation-{variant['name']}-Seed{seed}"
    return sorted(REPO.glob(f".paper_ablation_{stem}_gpu*.exit"))


def terminal_attempt(dataset, variant, seed):
    return bool(exit_paths(dataset, variant, seed))


def job_complete(dataset, variant, seed):
    for run_dir in run_dirs(dataset, variant, seed):
        stats_path = run_dir / str(seed) / "train" / "stats.json"
        last = train_last_epoch(stats_path)
        if last is not None and last >= variant["max_epoch"] - 1:
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
            state[int(parts[0])] = {
                "free_mib": int(parts[1]),
                "util": int(parts[2]),
            }
    return state


def launch(dataset, variant, seed, gpu, cfg_path):
    stem = f"AML-{dataset}-PaperAblation-{variant['name']}-Seed{seed}"
    log_path = REPO / f".paper_ablation_{stem}_gpu{gpu}.stdout"
    pid_path = REPO / f".paper_ablation_{stem}_gpu{gpu}.pid"
    exit_path = REPO / f".paper_ablation_{stem}_gpu{gpu}.exit"
    q = shlex.quote
    command = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"conda activate {CONDA_ENV} && "
        f"cd {q(str(REPO))} && "
        "WANDB_MODE=disabled "
        f"python -m fraudGT.main --cfg {q(str(cfg_path))} --repeat 1 --gpu {gpu}; "
        "status=$?; "
        f"printf '%s\\n' \"$status\" > {q(str(exit_path))}; "
        "exit \"$status\""
    )
    with log_path.open("a") as out:
        out.write(f"\n===== launch {ts()} gpu={gpu} cfg={cfg_path} =====\n")
        out.flush()
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    pid_path.write_text(str(proc.pid))
    log(
        f"launched dataset={dataset} variant={variant['name']} "
        f"seed={seed} gpu={gpu} pid={proc.pid}"
    )


def build_jobs(seeds, variant_names, seed_policy):
    variants = [variant_by_name(name) for name in variant_names]
    jobs = []
    for dataset in DATASETS:
        dataset_seeds = (
            [BEST_SEED_BY_DATASET[dataset]] if seed_policy == "best" else seeds
        )
        for variant in variants:
            for seed in dataset_seeds:
                cfg_path = cfg_path_for(dataset, variant, seed)
                jobs.append((dataset, variant, seed, cfg_path))
    return jobs


def classify_jobs(jobs):
    pending = []
    blocked = []
    for dataset, variant, seed, cfg_path in jobs:
        if job_complete(dataset, variant, seed) or active_for_cfg(cfg_path):
            continue
        if terminal_attempt(dataset, variant, seed):
            blocked.append((dataset, variant, seed, cfg_path))
            continue
        pending.append((dataset, variant, seed, cfg_path))
    return pending, blocked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--poll", type=int, default=300)
    parser.add_argument("--min-free-mib", type=int, default=18000)
    parser.add_argument("--max-util", type=int, default=5)
    parser.add_argument("--max-active", type=int, default=1)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--seed-policy",
        choices=["fixed", "best"],
        default="fixed",
        help="Use --seeds for all datasets or the recorded best mainline seed per dataset.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(variant["name"] for variant in VARIANTS),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ignore-formal-gate", action="store_true")
    parser.add_argument(
        "--wait-formal-gate",
        action="store_true",
        help="When --start is set, wait until the formal 18-run package is complete.",
    )
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item]
    variant_names = [item for item in args.variants.split(",") if item]
    jobs = build_jobs(seeds, variant_names, args.seed_policy)
    seed_desc = (
        ",".join(f"{dataset}:{seed}" for dataset, seed in BEST_SEED_BY_DATASET.items())
        if args.seed_policy == "best"
        else ",".join(map(str, seeds))
    )
    log(
        f"prepared ablation configs jobs={len(jobs)} "
        f"variants={','.join(variant_names)} seed_policy={args.seed_policy} "
        f"seeds={seed_desc}"
    )

    if not args.start:
        for dataset, variant, seed, cfg_path in jobs:
            print(
                "\t".join(
                    [
                        dataset,
                        variant["name"],
                        str(seed),
                        str(variant["max_epoch"]),
                        str(cfg_path),
                    ]
                )
            )
        log("not started; pass --start after formal mainline is complete")
        return 0

    if not args.ignore_formal_gate:
        while True:
            complete, total = formal_complete()
            if complete >= total:
                log(f"formal gate satisfied complete={complete}/{total}")
                break
            if not args.wait_formal_gate:
                log(
                    f"formal gate not satisfied complete={complete}/{total}; "
                    "ablation queue will not start"
                )
                return 2
            log(
                f"formal gate not satisfied complete={complete}/{total}; "
                f"waiting {args.poll}s before recheck"
            )
            if args.once:
                return 2
            time.sleep(args.poll)

    log("starting clean ablation queue")
    while True:
        pending, blocked = classify_jobs(jobs)
        active = active_lines()
        active_count = len(active)
        if not pending and not active:
            if blocked:
                log(
                    "clean ablation queue stopped with terminal incomplete "
                    f"attempts blocked={len(blocked)}; inspect .paper_ablation_*.exit"
                )
                return 2
            log("clean ablation queue finished")
            return 0

        state = gpu_state()
        launched = 0
        if active_count < args.max_active:
            for gpu, gpu_info in sorted(state.items()):
                if launched >= len(pending):
                    break
                if active_count + launched >= args.max_active:
                    break
                if gpu_info["free_mib"] < args.min_free_mib:
                    continue
                if gpu_info["util"] > args.max_util:
                    continue
                if any(f"--gpu {gpu}" in line or f"--gpu={gpu}" in line for line in active):
                    continue
                dataset, variant, seed, cfg_path = pending[launched]
                launch(dataset, variant, seed, gpu, cfg_path)
                launched += 1

        if args.once:
            log(
                f"one-shot complete; pending={len(pending)} "
                f"blocked={len(blocked)} launched={launched}"
            )
            return 0
        log(
            f"waiting; pending={len(pending)} blocked={len(blocked)} "
            f"active={active_count} max_active={args.max_active} "
            f"gpu_state={state} launched={launched}"
        )
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
