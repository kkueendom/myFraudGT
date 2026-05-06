#!/usr/bin/env python3
import ast
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import paramiko


HOST = "10.168.1.101"
USER = "yyk"
PASSWORD = "tj654478"
REMOTE_REPO = "/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining"
CONDA_ENV = "fraudgt_dual_gate"
POLL_SECONDS = 300

LOG_PATH = Path("/Users/kun/classsplit_subgraph_family.log")
STATE_JSON = Path("/Users/kun/classsplit_subgraph_family_state.json")
SUMMARY_MD = Path("/Users/kun/Desktop/classsplit_subgraph_family_status.md")

META = [
    {
        "dataset": "Small-LI",
        "cfg": "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 1,
        "tag": "smallli",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Small-LI-UnifiedClassSplitSubgraphRoute240-gpu1",
        "target": 0.5101,
        "max_epoch": 240,
    },
    {
        "dataset": "Small-HI",
        "cfg": "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 1,
        "tag": "smallhi",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Small-HI-UnifiedClassSplitSubgraphRoute240-gpu1",
        "target": 0.8013,
        "max_epoch": 240,
    },
    {
        "dataset": "Large-LI",
        "cfg": "configs/AML-Large-LI/AML-Large-LI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 1,
        "tag": "largeli",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Large-LI-UnifiedClassSplitSubgraphRoute240-gpu1",
        "target": 0.4143,
        "max_epoch": 240,
    },
    {
        "dataset": "Medium-HI",
        "cfg": "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 0,
        "tag": "mediumhi",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Medium-HI-UnifiedClassSplitSubgraphRoute240-gpu0",
        "target": 0.7993,
        "max_epoch": 240,
    },
    {
        "dataset": "Large-HI",
        "cfg": "configs/AML-Large-HI/AML-Large-HI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 0,
        "tag": "largehi",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Large-HI-UnifiedClassSplitSubgraphRoute240-gpu0",
        "target": 0.7734,
        "max_epoch": 240,
    },
    {
        "dataset": "Medium-LI",
        "cfg": "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphRoute240.yaml",
        "gpu": 0,
        "tag": "mediumli",
        "run_dir": "/e/yyk/FraudGT_multi6/unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main/AML-Medium-LI-UnifiedClassSplitSubgraphRoute240-gpu0",
        "target": 0.4806,
        "max_epoch": 240,
    },
]

GPU_ORDER = {1: ["Small-LI", "Small-HI", "Large-LI"], 0: ["Medium-HI", "Large-HI", "Medium-LI"]}
BY_DATASET = {item["dataset"]: item for item in META}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return cli


def run_remote(cli, cmd, timeout=120):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def kill_stale_queue(cli):
    cmd = r"""
python3 - <<'PY2'
import subprocess
for line in subprocess.run(['ps','-eo','pid,args='], capture_output=True, text=True, check=False).stdout.splitlines():
    if 'classsplit_remaining_queue.py' in line and 'grep' not in line:
        pid = int(line.strip().split(None, 1)[0])
        subprocess.run(['kill', str(pid)], check=False)
        print('killed', pid)
PY2
"""
    code, out, err = run_remote(cli, cmd, timeout=60)
    if out.strip():
        log(out.strip())


def fetch_state(cli):
    payload = json.dumps(META)
    cmd = f"""
python3 - <<'PY2'
import ast, json, os, subprocess
meta = json.loads({payload!r})
ps = subprocess.run(['ps', '-eo', 'pid,args='], capture_output=True, text=True, check=False).stdout.splitlines()
active = {{}}
active_gpus = set()
for line in ps:
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    command = parts[1]
    if 'fraudGT.main' not in command:
        continue
    for item in meta:
        if item['cfg'] in command:
            active[item['dataset']] = command
            active_gpus.add(int(item['gpu']))
            break
res = {{'active': active, 'active_gpus': sorted(active_gpus), 'states': {{}}}}
for item in meta:
    log_path = os.path.join(item['run_dir'], '42', 'logging.log')
    st = {{'exists': os.path.exists(log_path), 'epoch_max': -1, 'raw_peak': None, 'raw_peak_epoch': None, 'done': False}}
    if os.path.exists(log_path):
        with open(log_path, 'r', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('test: '):
                    try:
                        obj = ast.literal_eval(line.split('test: ', 1)[1])
                        ep = int(obj.get('epoch', -1))
                        f1 = float(obj.get('f1', 0.0))
                        st['epoch_max'] = max(st['epoch_max'], ep)
                        if st['raw_peak'] is None or f1 > st['raw_peak']:
                            st['raw_peak'] = f1
                            st['raw_peak_epoch'] = ep
                    except Exception:
                        pass
                elif line.startswith('val: '):
                    try:
                        obj = ast.literal_eval(line.split('val: ', 1)[1])
                        ep = int(obj.get('epoch', -1))
                        st['epoch_max'] = max(st['epoch_max'], ep)
                    except Exception:
                        pass
                elif 'Task done, results saved' in line or '[*] All done:' in line:
                    st['done'] = True
    st['complete'] = st['done'] or st['epoch_max'] >= item['max_epoch'] - 1
    st['passed'] = st['raw_peak'] is not None and st['raw_peak'] >= item['target']
    res['states'][item['dataset']] = st
print(json.dumps(res))
PY2
"""
    code, out, err = run_remote(cli, cmd, timeout=240)
    if code != 0:
        raise RuntimeError(err or out or "fetch_state failed")
    return json.loads(out.strip())


def write_summary(state):
    lines = [
        "# Unified ClassSplit SubgraphRoute Status",
        "",
        f"- 更新时间: `{ts()}`",
        "",
        "| Dataset | GPU | Active | Epoch Max | Raw Peak | Peak Epoch | Target | Pass | Run Dir |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in META:
        st = state["states"].get(item["dataset"], {})
        peak = st.get("raw_peak")
        lines.append(
            "| {dataset} | {gpu} | {active} | {epoch} | {peak} | {peak_epoch} | {target:.4f} | {passed} | `{run_dir}` |".format(
                dataset=item["dataset"],
                gpu=item["gpu"],
                active="yes" if item["dataset"] in state.get("active_datasets", []) else "no",
                epoch=st.get("epoch_max", "-"),
                peak="-" if peak is None else f"{peak:.5f}",
                peak_epoch="-" if st.get("raw_peak_epoch") is None else st.get("raw_peak_epoch"),
                target=item["target"],
                passed="yes" if st.get("passed") else "no",
                run_dir=item["run_dir"],
            )
        )
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def next_pending(state, gpu):
    active = set(state.get("active_datasets", []))
    for dataset in GPU_ORDER[gpu]:
        if dataset in active:
            return None
        if not state["states"].get(dataset, {}).get("complete"):
            return BY_DATASET[dataset]
    return None


def launch(cli, item):
    dataset = item["dataset"]
    cfg = item["cfg"]
    gpu = item["gpu"]
    tag = item["tag"]
    run_dir = item["run_dir"]
    backup = f"{run_dir}_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stdout_log = f"{REMOTE_REPO}/.last_classsplit_subgraph_{tag}_gpu{gpu}_stdout"
    pid_file = f"{REMOTE_REPO}/.last_classsplit_subgraph_{tag}_gpu{gpu}_pid"
    q = shlex.quote
    cmd = f"""
bash -lc 'set -e
resume_args=""
ckpt_glob={q(run_dir + "/42/ckpt/*.ckpt")}
if ls $ckpt_glob >/dev/null 2>&1; then
  resume_args=" train.auto_resume True train.epoch_resume -1"
elif [ -d {q(run_dir)} ] && [ ! -f {q(run_dir + "/42/logging.log.done")} ]; then
  mv {q(run_dir)} {q(backup)}
fi
nohup bash -lc "source ~/.bashrc >/dev/null 2>&1 || true && conda activate {CONDA_ENV} && cd {q(REMOTE_REPO)} && exec python -m fraudGT.main --cfg {q(cfg)} --repeat 1 --gpu {gpu}$resume_args" > {q(stdout_log)} 2>&1 < /dev/null &
echo $! > {q(pid_file)}
printf "%s" "$(cat {q(pid_file)})"
'
"""
    code, out, err = run_remote(cli, cmd, timeout=90)
    if code != 0:
        raise RuntimeError(err or out or f"launch failed {dataset}")
    log(f"launch {dataset} gpu={gpu} pid={out.strip()}")


def main():
    log("classsplit subgraph family daemon start")
    cli = None
    while True:
        try:
            if cli is None:
                cli = connect()
                log("connected to remote host")
            kill_stale_queue(cli)
            raw = fetch_state(cli)
            state = {
                "active_datasets": sorted(raw.get("active", {}).keys()),
                "active_gpus": raw.get("active_gpus", []),
                "states": raw["states"],
            }
            STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False))
            write_summary(state)
            status_parts = []
            for item in META:
                item_state = state["states"][item["dataset"]]
                peak = item_state["raw_peak"]
                peak_text = "-" if peak is None else f"{peak:.4f}"
                status_parts.append(
                    f"{item['dataset']}:{peak_text}@{item_state['raw_peak_epoch']}"
                )
            status = " ".join(status_parts)
            log(
                "status "
                + status
                + f" active={state['active_datasets']} active_gpus={state['active_gpus']}"
            )
            if all(state["states"][item["dataset"]]["complete"] for item in META):
                log("all datasets complete; daemon exit")
                return
            active_gpus = set(state["active_gpus"])
            for gpu in (0, 1):
                if gpu in active_gpus:
                    continue
                item = next_pending(state, gpu)
                if item is not None:
                    launch(cli, item)
                    time.sleep(5)
            time.sleep(POLL_SECONDS)
        except Exception as exc:
            log(f"loop error: {exc}; retry in {POLL_SECONDS}s")
            try:
                if cli is not None:
                    cli.close()
            except Exception:
                pass
            cli = None
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
