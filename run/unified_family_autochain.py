#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_FAMILY_CHAIN = [
    "supportmixconsisclassmixprotoboundclasssplitresid_full240",
    "supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240",
    "supportmixconsisclassmixprotoboundclasssplitsubgraphrouteprotoboundresid_full240",
    "supportmixconsisdualprotoboundresid_full240",
    "supportmixconsisdeltafusiondualprotoboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardscaleboundresid_full240",
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardscaleclassrouteboundresid_full240",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def parse_all_pass(queue_log: Path, family: str):
    target = f"family {family} all_pass="
    all_pass = None
    if not queue_log.exists():
        return None
    for line in queue_log.read_text(errors="ignore").splitlines():
        if target not in line:
            continue
        suffix = line.split(target, 1)[1].strip()
        if suffix.startswith("True"):
            all_pass = True
        elif suffix.startswith("False"):
            all_pass = False
    return all_pass


def run_family(family: str, gpu: int, poll_seconds: int, max_used_mem_mib: int, repo_root: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    queue_log = repo_root / "logs" / f"autochain_{family}_{ts}.log"
    cmd = [
        sys.executable,
        "run/unified_family_queue.py",
        "--family",
        family,
        "--gpu",
        str(gpu),
        "--poll-seconds",
        str(poll_seconds),
        "--max-used-mem-mib",
        str(max_used_mem_mib),
    ]
    log(f"starting family={family} log={queue_log}")
    with queue_log.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (repo_root / ".last_unified_autochain_pid").write_text(f"{proc.pid}\n")
        (repo_root / ".last_unified_autochain_log").write_text(f"{queue_log}\n")
        status = proc.wait()
    all_pass = parse_all_pass(queue_log, family)
    log(
        f"finished family={family} status={status} "
        f"all_pass={all_pass} log={queue_log}"
    )
    return status, all_pass, queue_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-used-mem-mib", type=int, default=1000)
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILY_CHAIN,
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    for family in args.families:
        status, all_pass, _ = run_family(
            family,
            args.gpu,
            args.poll_seconds,
            args.max_used_mem_mib,
            repo_root,
        )
        if status != 0:
            log(f"family={family} exited with status={status}; continuing to next family")
            continue
        if all_pass:
            log(f"family={family} satisfied all +2 raw-peak targets; stopping chain")
            return 0
    log("autochain exhausted all configured families without satisfying all +2 targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
