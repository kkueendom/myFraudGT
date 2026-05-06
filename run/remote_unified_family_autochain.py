#!/usr/bin/env python3
import argparse
import ast
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import paramiko

from unified_family_queue import FAMILIES, THRESHOLDS, read_config_meta


HOST = "10.168.1.101"
USER = "yyk"
PASSWORD = "tj654478"
REMOTE_REPO = "/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining"
CONDA_ENV = "fraudgt_dual_gate"

GPU_ORDER = {
    1: ["Small-LI", "Small-HI", "Large-LI"],
    0: ["Medium-HI", "Large-HI", "Medium-LI"],
}

DEFAULT_CHAIN = [
    "supportmixconsisdualprotoboundresid_full240",
    "supportmixconsisdeltafusiondualprotoboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardscaleboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardscaleclassrouteboundresid_full240",
]

LOG_PATH = Path("/Users/kun/remote_unified_family_autochain.log")
SUMMARY_MD = Path("/Users/kun/Desktop/remote_unified_family_autochain_status.md")


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
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


def wait_for_pid_release(pid_file, poll_seconds):
    pid_path = Path(pid_file)
    last_msg = None
    while True:
        if not pid_path.exists():
            log(f"pid file released: {pid_file}")
            return
        raw_pid = pid_path.read_text().strip()
        if not raw_pid.isdigit():
            msg = f"waiting for ready pid file {pid_file}: {raw_pid!r}"
            if msg != last_msg:
                log(msg)
                last_msg = msg
            time.sleep(poll_seconds)
            continue
        pid = int(raw_pid)
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if not proc.stdout.strip():
            log(f"pid {pid} exited; continue")
            return
        msg = f"waiting on local pid {pid}"
        if msg != last_msg:
            log(msg)
            last_msg = msg
        time.sleep(poll_seconds)


def family_meta(family):
    items = []
    for cfg_path in FAMILIES[family]:
        out_dir, dataset = read_config_meta(cfg_path)
        gpu = 1 if dataset in GPU_ORDER[1] else 0
        run_dir = str(Path(out_dir) / f"{Path(cfg_path).stem}-gpu{gpu}")
        items.append(
            {
                "dataset": dataset,
                "cfg": cfg_path,
                "gpu": gpu,
                "run_dir": run_dir,
                "target": THRESHOLDS[dataset],
                "max_epoch": 240,
            }
        )
    return items


def fetch_state(cli, family):
    payload = json.dumps(family_meta(family))
    cmd = f"""
python3 - <<'PY'
import ast
import json
import os
import subprocess

items = json.loads({payload!r})
ps = subprocess.run(['ps', '-eo', 'pid,args='], capture_output=True, text=True, check=False).stdout.splitlines()
active = {{}}
active_gpus = set()
for line in ps:
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    command = parts[1]
    if not (command.startswith('python -m fraudGT.main ') or command.startswith('python3 -m fraudGT.main ')):
        continue
    for item in items:
        if item['cfg'] in command:
            active[item['dataset']] = command
            active_gpus.add(int(item['gpu']))
            break
res = {{'active': active, 'active_gpus': sorted(active_gpus), 'states': {{}}}}
for item in items:
    log_path = os.path.join(item['run_dir'], '42', 'logging.log')
    st = {{
        'exists': os.path.exists(log_path),
        'epoch_max': -1,
        'raw_peak': None,
        'raw_peak_epoch': None,
        'done': False,
    }}
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
PY
"""
    code, out, err = run_remote(cli, cmd, timeout=240)
    if code != 0:
        raise RuntimeError(err or out or "fetch_state failed")
    return json.loads(out.strip())


def prepare_family(cli, family):
    items = family_meta(family)
    payload = json.dumps(items)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = f"""
python3 - <<'PY'
import json
import os
import subprocess

items = json.loads({payload!r})
stamp = {stamp!r}
for item in items:
    run_dir = item['run_dir']
    if not os.path.isdir(run_dir):
        continue
    backup = f"{{run_dir}}_fresh_{{stamp}}"
    subprocess.run(['mv', run_dir, backup], check=True)
    print(f"moved {{run_dir}} -> {{backup}}")
PY
"""
    code, out, err = run_remote(cli, cmd, timeout=240)
    if code != 0:
        raise RuntimeError(err or out or f"prepare_family failed: {family}")
    if out.strip():
        for line in out.strip().splitlines():
            log(line)


def write_summary(family, state):
    lines = [
        "# Remote Unified Family Autochain",
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
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


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


def launch(cli, item):
    cfg = item["cfg"]
    gpu = item["gpu"]
    run_dir = item["run_dir"]
    tag = item["dataset"].lower().replace("-", "")
    stdout_log = f"{REMOTE_REPO}/.last_remote_unified_family_{tag}_gpu{gpu}_stdout"
    pid_file = f"{REMOTE_REPO}/.last_remote_unified_family_{tag}_gpu{gpu}_pid"
    backup = f"{run_dir}_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    q = shlex.quote
    cmd = f"""
bash -lc 'set -e
if [ -d {q(run_dir)} ]; then
  mv {q(run_dir)} {q(backup)}
fi
nohup bash -lc "source ~/.bashrc >/dev/null 2>&1 || true && conda activate {CONDA_ENV} && cd {q(REMOTE_REPO)} && exec python -m fraudGT.main --cfg {q(cfg)} --repeat 1 --gpu {gpu}" > {q(stdout_log)} 2>&1 < /dev/null &
echo $! > {q(pid_file)}
printf "%s" "$(cat {q(pid_file)})"
'
"""
    code, out, err = run_remote(cli, cmd, timeout=90)
    if code != 0:
        raise RuntimeError(err or out or f"launch failed: {item['dataset']}")
    log(f"launch {item['dataset']} gpu={gpu} pid={out.strip()}")


def run_family(cli, family, poll_seconds):
    log(f"family {family} start")
    prepare_family(cli, family)
    while True:
        state = fetch_state(cli, family)
        write_summary(family, state)
        status_parts = []
        for item in family_meta(family):
            st = state["states"][item["dataset"]]
            peak = "-" if st["raw_peak"] is None else f"{st['raw_peak']:.4f}"
            status_parts.append(f"{item['dataset']}:{peak}@{st['raw_peak_epoch']}")
        log(
            "status "
            + " ".join(status_parts)
            + f" active={sorted(state['active'])} active_gpus={state['active_gpus']}"
        )

        if all(state["states"][item["dataset"]]["complete"] for item in family_meta(family)):
            all_pass = all(state["states"][item["dataset"]]["passed"] for item in family_meta(family))
            log(f"family {family} complete all_pass={all_pass}")
            return all_pass

        active_gpus = set(state["active_gpus"])
        for gpu in (0, 1):
            if gpu in active_gpus:
                continue
            item = next_pending(state, family, gpu)
            if item is not None:
                launch(cli, item)
                time.sleep(5)

        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_CHAIN,
    )
    parser.add_argument("--wait-pid-file")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    LOG_PATH.write_text("")
    log("remote unified family autochain start")
    if args.wait_pid_file:
        wait_for_pid_release(args.wait_pid_file, args.poll_seconds)

    cli = None
    try:
        cli = connect()
        log("connected to remote host")
        for family in args.families:
            ok = run_family(cli, family, args.poll_seconds)
            if ok:
                log(f"family {family} satisfied all targets; stop chain")
                return 0
        log("autochain exhausted all configured families without satisfying all targets")
        return 0
    finally:
        if cli is not None:
            cli.close()


if __name__ == "__main__":
    raise SystemExit(main())
