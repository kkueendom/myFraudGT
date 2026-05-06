#!/usr/bin/env python3
import argparse
import ast
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from unified_family_queue import FAMILIES, THRESHOLDS, read_config_meta


CONDA_ENV = "fraudgt_dual_gate"
GPU_ORDER = {
    1: ["Small-LI", "Small-HI", "Large-LI"],
    0: ["Medium-HI", "Large-HI", "Medium-LI"],
}


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def family_meta(family):
    items = []
    for cfg_path in FAMILIES[family]:
        out_dir, dataset, max_epoch = read_config_meta(cfg_path)
        gpu = 1 if dataset in GPU_ORDER[1] else 0
        run_dir = str(Path(out_dir) / f"{Path(cfg_path).stem}-gpu{gpu}")
        items.append(
            {
                "dataset": dataset,
                "cfg": cfg_path,
                "gpu": gpu,
                "run_dir": run_dir,
                "target": THRESHOLDS[dataset],
                "max_epoch": max_epoch,
            }
        )
    return items


def fetch_state(family):
    items = family_meta(family)
    ps_out = subprocess.run(
        ["ps", "-eo", "pid,args="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    active = {}
    active_gpus = set()
    for line in ps_out:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        command = parts[1]
        if not (
            command.startswith("python -m fraudGT.main ") or
            command.startswith("python3 -m fraudGT.main ")
        ):
            continue
        for item in items:
            if item["cfg"] in command:
                active[item["dataset"]] = command
                active_gpus.add(int(item["gpu"]))
                break

    states = {}
    for item in items:
        log_path = Path(item["run_dir"]) / "42" / "logging.log"
        state = {
            "exists": log_path.exists(),
            "epoch_max": -1,
            "raw_peak": None,
            "raw_peak_epoch": None,
            "done": False,
        }
        if log_path.exists():
            for raw in log_path.read_text(errors="ignore").splitlines():
                if raw.startswith("test: "):
                    try:
                        obj = ast.literal_eval(raw.split("test: ", 1)[1])
                    except Exception:
                        continue
                    epoch = int(obj.get("epoch", -1))
                    f1 = float(obj.get("f1", 0.0))
                    state["epoch_max"] = max(state["epoch_max"], epoch)
                    if state["raw_peak"] is None or f1 > state["raw_peak"]:
                        state["raw_peak"] = f1
                        state["raw_peak_epoch"] = epoch
                elif raw.startswith("val: "):
                    try:
                        obj = ast.literal_eval(raw.split("val: ", 1)[1])
                    except Exception:
                        continue
                    state["epoch_max"] = max(
                        state["epoch_max"], int(obj.get("epoch", -1))
                    )
                elif "Task done, results saved" in raw or "[*] All done:" in raw:
                    state["done"] = True
        state["complete"] = (
            state["done"] or
            state["epoch_max"] >= item["max_epoch"] - 1
        )
        state["passed"] = (
            state["raw_peak"] is not None and
            state["raw_peak"] >= item["target"]
        )
        states[item["dataset"]] = state
    return {
        "active": active,
        "active_gpus": sorted(active_gpus),
        "states": states,
    }


def prepare_family(family):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for item in family_meta(family):
        run_dir = Path(item["run_dir"])
        if not run_dir.is_dir():
            continue
        backup = Path(f"{run_dir}_fresh_{stamp}")
        run_dir.rename(backup)
        log(f"moved {run_dir} -> {backup}")


def write_summary(repo_root, family, state):
    summary_path = repo_root / f".server_family_{family}_status.md"
    lines = [
        "# Server Unified Family Supervisor",
        "",
        f"- 更新时间: `{ts()}`",
        f"- 当前家族: `{family}`",
        "",
        "| Dataset | Active | Epoch Max | Raw Peak | Peak Epoch | Target | Pass | Run Dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in family_meta(family):
        st = state["states"][item["dataset"]]
        peak = st["raw_peak"]
        lines.append(
            "| {dataset} | {active} | {epoch} | {peak} | {peak_epoch} | {target:.4f} | {passed} | `{run_dir}` |".format(
                dataset=item["dataset"],
                active="yes" if item["dataset"] in state["active"] else "no",
                epoch=st["epoch_max"],
                peak="-" if peak is None else f"{peak:.5f}",
                peak_epoch="-" if st["raw_peak_epoch"] is None else st["raw_peak_epoch"],
                target=item["target"],
                passed="yes" if st["passed"] else "no",
                run_dir=item["run_dir"],
            )
        )
    summary_path.write_text("\n".join(lines) + "\n")


def next_pending(state, family, gpu):
    active = set(state["active"])
    for item in family_meta(family):
        if item["gpu"] != gpu:
            continue
        if item["dataset"] in active:
            return None
        if not state["states"][item["dataset"]]["complete"]:
            return item
    return None


def launch(repo_root, item):
    cfg = item["cfg"]
    gpu = item["gpu"]
    run_dir = item["run_dir"]
    tag = item["dataset"].lower().replace("-", "")
    stdout_log = repo_root / f".server_family_{tag}_gpu{gpu}_stdout"
    pid_file = repo_root / f".server_family_{tag}_gpu{gpu}_pid"
    backup = Path(f"{run_dir}_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if Path(run_dir).is_dir():
        Path(run_dir).rename(backup)
        log(f"moved {run_dir} -> {backup}")
    q = shlex.quote
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true && "
        f"conda activate {CONDA_ENV} && "
        f"cd {q(str(repo_root))} && "
        f"exec python -m fraudGT.main --cfg {q(cfg)} --repeat 1 --gpu {gpu}"
    )
    with stdout_log.open("w") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    pid_file.write_text(f"{proc.pid}\n")
    log(f"launch {item['dataset']} gpu={gpu} pid={proc.pid}")


def run_family(repo_root, family, poll_seconds):
    log(f"family {family} start")
    state = fetch_state(family)
    if state["active"]:
        log(f"resume family {family}; active={sorted(state['active'])}, skip prepare")
    else:
        prepare_family(family)

    while True:
        state = fetch_state(family)
        write_summary(repo_root, family, state)
        status_parts = []
        for item in family_meta(family):
            st = state["states"][item["dataset"]]
            peak = "-" if st["raw_peak"] is None else f"{st['raw_peak']:.4f}"
            status_parts.append(
                f"{item['dataset']}:{peak}@{st['raw_peak_epoch']}"
            )
        log(
            "status "
            + " ".join(status_parts)
            + f" active={sorted(state['active'])} active_gpus={state['active_gpus']}"
        )

        if all(
            state["states"][item["dataset"]]["complete"]
            for item in family_meta(family)
        ):
            all_pass = all(
                state["states"][item["dataset"]]["passed"]
                for item in family_meta(family)
            )
            log(f"family {family} complete all_pass={all_pass}")
            return all_pass

        active_gpus = set(state["active_gpus"])
        for gpu in (0, 1):
            if gpu in active_gpus:
                continue
            item = next_pending(state, family, gpu)
            if item is not None:
                launch(repo_root, item)
                time.sleep(5)

        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ok = run_family(repo_root, args.family, args.poll_seconds)
    if ok:
        log(f"family {args.family} satisfied all targets")
    else:
        log(f"family {args.family} finished without satisfying all targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
