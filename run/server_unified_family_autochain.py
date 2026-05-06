#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

from server_unified_family_supervisor import FAMILIES
from server_unified_family_supervisor import family_meta
from server_unified_family_supervisor import fetch_state
from server_unified_family_supervisor import log
from server_unified_family_supervisor import run_family


def process_cmdline(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def wait_for_pid_release(pid_file, cmd_substr, poll_seconds):
    pid_path = Path(pid_file)
    last_msg = None
    while True:
        if not pid_path.exists():
            log(f"pid file {pid_file} missing; continue")
            return
        raw_pid = pid_path.read_text().strip()
        if not raw_pid.isdigit():
            msg = f"pid file {pid_file} not ready: {raw_pid!r}"
            if msg != last_msg:
                log(msg)
                last_msg = msg
            time.sleep(poll_seconds)
            continue
        pid = int(raw_pid)
        cmdline = process_cmdline(pid)
        if not cmdline:
            log(f"wait pid {pid} exited")
            return
        if cmd_substr and cmd_substr not in cmdline:
            log(f"wait pid {pid} command changed; continue")
            return
        msg = f"waiting on pid {pid}: {cmdline}"
        if msg != last_msg:
            log(msg)
            last_msg = msg
        time.sleep(poll_seconds)


def family_complete(state, family):
    return all(
        state["states"][item["dataset"]]["complete"]
        for item in family_meta(family)
    )


def family_passed(state, family):
    return all(
        state["states"][item["dataset"]]["passed"]
        for item in family_meta(family)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        required=True,
        choices=sorted(FAMILIES),
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--wait-pid-file",
        default="",
        help="Optional existing supervisor pid file to wait on before continuing.",
    )
    parser.add_argument(
        "--wait-cmd-substr",
        default="",
        help="Optional command substring to validate the waited pid.",
    )
    parser.add_argument(
        "--resume-family",
        default="",
        choices=[""] + sorted(FAMILIES),
        help="Optional family that is already running under another supervisor.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    chain = list(args.families)

    if args.wait_pid_file:
        wait_for_pid_release(
            args.wait_pid_file,
            args.wait_cmd_substr,
            args.poll_seconds,
        )

    if args.resume_family:
        state = fetch_state(args.resume_family)
        if family_complete(state, args.resume_family):
            ok = family_passed(state, args.resume_family)
            log(
                f"resume family {args.resume_family} complete all_pass={ok}"
            )
            if ok:
                return
        else:
            log(
                f"resume family {args.resume_family} incomplete after wait; "
                "taking over"
            )
            ok = run_family(repo_root, args.resume_family, args.poll_seconds)
            if ok:
                return
        chain = [family for family in chain if family != args.resume_family]

    for family in chain:
        ok = run_family(repo_root, family, args.poll_seconds)
        if ok:
            log(f"chain satisfied by family {family}")
            return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
