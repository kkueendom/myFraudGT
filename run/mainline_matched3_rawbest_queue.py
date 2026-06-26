#!/usr/bin/env python3
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining")
EVENTS = REPO / ".mainline_matched3_rawbest_queue.events"
NOHUP_DIR = REPO
POLL = 600

TASKS = [
    ("Small-HI", 42, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml"),
    ("Small-HI", 43, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml"),
    ("Small-HI", 44, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml"),
    ("Small-LI", 42, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed42.yaml"),
    ("Small-LI", 43, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed43.yaml"),
    ("Small-LI", 44, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed44.yaml"),
    ("Medium-HI", 42, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml"),
    ("Medium-HI", 43, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml"),
    ("Medium-HI", 44, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml"),
    ("Medium-LI", 42, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180.yaml"),
    ("Medium-LI", 43, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml"),
    ("Medium-LI", 44, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml"),
    ("Large-HI", 42, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed42.yaml"),
    ("Large-HI", 43, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed43.yaml"),
    ("Large-HI", 44, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed44.yaml"),
    ("Large-LI", 42, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed42.yaml"),
    ("Large-LI", 43, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed43.yaml"),
    ("Large-LI", 44, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed44.yaml"),
]


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with EVENTS.open("a") as f:
        f.write(line + "\n")


def cfg_text(cfg):
    return (REPO / cfg).read_text(errors="ignore")


def cfg_value(text, key):
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else None


def max_epoch(text):
    match = re.search(r"^optim:\n(?:^  .+\n)*?^  max_epoch:\s*(\d+)", text, re.M)
    return int(match.group(1)) if match else None


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row and "f1" in row:
            result.append(row)
    return result


def run_dirs(cfg):
    cfg_path = REPO / cfg
    text = cfg_text(cfg)
    out_dir = Path(cfg_value(text, "out_dir"))
    return sorted(out_dir.glob(cfg_path.stem + "-gpu*"))


def state(dataset, seed, cfg):
    text = cfg_text(cfg)
    expected_max_epoch = max_epoch(text)
    best_dir = None
    train_last = None
    best = None
    for run_dir in run_dirs(cfg):
        seed_dir = run_dir / str(seed)
        train_rows = rows(seed_dir / "train" / "stats.json")
        test_rows = rows(seed_dir / "test" / "stats.json")
        this_last = int(train_rows[-1]["epoch"]) if train_rows else None
        if this_last is not None and (train_last is None or this_last > train_last):
            train_last = this_last
            best_dir = run_dir
        if test_rows:
            raw_best = max(test_rows, key=lambda item: float(item["f1"]))
            if best is None or float(raw_best["f1"]) > best[0]:
                best = (float(raw_best["f1"]), int(raw_best["epoch"]))
    complete = (
        train_last is not None
        and expected_max_epoch is not None
        and train_last >= expected_max_epoch - 1
    )
    return {
        "complete": complete,
        "train_last": train_last,
        "max_epoch": expected_max_epoch,
        "run_dir": best_dir,
        "best": best,
    }


def free_gpus():
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    bus_to_idx = {}
    for line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            bus_to_idx[parts[1]] = int(parts[0])

    busy = set()
    proc_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in proc_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] in bus_to_idx:
            busy.add(bus_to_idx[parts[0]])
    return [idx for idx in sorted(bus_to_idx.values()) if idx not in busy]


def preferred_gpu(st):
    run_dir = st.get("run_dir")
    if not run_dir:
        return None
    match = re.search(r"-gpu(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def has_ckpt(st, seed):
    run_dir = st.get("run_dir")
    if not run_dir:
        return False
    return any((run_dir / str(seed) / "ckpt").glob("*.ckpt"))


def launch(dataset, seed, cfg, gpu, resume):
    log_path = NOHUP_DIR / f".mainline_matched3_{dataset}_seed{seed}_gpu{gpu}.stdout"
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "conda activate fraudgt_dual_gate; "
        f"python -m fraudGT.main --cfg {cfg} --repeat 1 --gpu {gpu}"
    )
    if resume:
        cmd += " train.auto_resume True train.epoch_resume -1"
    log(
        f"launch dataset={dataset} seed={seed} gpu={gpu} "
        f"resume={int(resume)} cfg={cfg} log={log_path.name}"
    )
    with log_path.open("ab") as output:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()
    log(f"finish dataset={dataset} seed={seed} gpu={gpu} status={return_code}")
    return return_code


def main():
    log("mainline matched3 rawbest single-GPU queue started")
    while True:
        pending = []
        for dataset, seed, cfg in TASKS:
            st = state(dataset, seed, cfg)
            if not st["complete"]:
                pending.append((dataset, seed, cfg, st))

        if not pending:
            log("mainline matched3 rawbest queue finished")
            return 0

        dataset, seed, cfg, st = pending[0]
        free = free_gpus()
        if not free:
            log(f"no free gpu; pending={len(pending)} next={dataset} seed={seed}; wait {POLL}s")
            time.sleep(POLL)
            continue

        pref = preferred_gpu(st)
        gpu = pref if pref in free else free[0]
        resume = has_ckpt(st, seed) and pref == gpu
        return_code = launch(dataset, seed, cfg, gpu, resume)
        if return_code != 0:
            log("nonzero exit; continue after wait so logs can be inspected")
            time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
