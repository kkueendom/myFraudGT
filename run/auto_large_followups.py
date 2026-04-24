#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TEST_F1_INLINE = re.compile(r"test:\s*\{.*?'f1':\s*([0-9.eE+-]+)")
TEST_F1_SUMMARY = re.compile(r"test_f1:\s*([0-9.eE+-]+)")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def read_text(path):
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="ignore")


def wait_for_queue_event(queue_log, event):
    pattern = re.compile(rf"{event} .* gpu=(\d+)")
    last_size = -1
    while True:
        text = read_text(queue_log)
        match = pattern.search(text)
        if match:
            return int(match.group(1))
        p = Path(queue_log)
        size = p.stat().st_size if p.exists() else 0
        if size != last_size:
            log(f"waiting for '{event}' in {queue_log}")
            last_size = size
        time.sleep(60)


def wait_for_queue_finish(queue_log):
    pattern = re.compile(r"finish .* status=(\d+) gpu=(\d+)")
    last_size = -1
    while True:
        text = read_text(queue_log)
        match = pattern.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
        p = Path(queue_log)
        size = p.stat().st_size if p.exists() else 0
        if size != last_size:
            log(f"waiting for finish in {queue_log}")
            last_size = size
        time.sleep(60)


def parse_best_test_f1(run_log):
    text = read_text(run_log).replace("\r", "\n")
    matches = []
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        match = TEST_F1_INLINE.search(line)
        if match:
            matches.append(float(match.group(1)))
    if not matches:
        clean_text = ANSI_ESCAPE.sub("", text)
        matches = [float(x) for x in TEST_F1_SUMMARY.findall(clean_text)]
    if not matches:
        return 0.0
    return max(matches)


def compute_run_dir(cfg_path, out_dir, name_tag, gpu):
    run_name = Path(cfg_path).stem
    if name_tag:
        run_name += f"-{name_tag}"
    run_name += f"-gpu{gpu}"
    return str(Path(out_dir) / run_name)


def acquire_gpu(lock_prefix="/tmp/fraudgt_gpu_", max_used_mem_mib=1000, poll_seconds=300):
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            idx_s, mem_s = [part.strip() for part in line.split(",")]
            idx = int(idx_s)
            mem = int(mem_s)
            if mem >= max_used_mem_mib:
                continue
            lock_dir = f"{lock_prefix}{idx}.lock"
            try:
                os.mkdir(lock_dir)
                return idx, lock_dir
            except FileExistsError:
                continue
        log("no free GPU, sleeping")
        time.sleep(poll_seconds)


def run_experiment(cfg_path, name_tag, out_dir, run_log, extra_overrides):
    gpu, lock_dir = acquire_gpu()
    out_run_dir = compute_run_dir(cfg_path, out_dir, name_tag, gpu)
    Path(run_log).parent.mkdir(parents=True, exist_ok=True)
    log(f"launching {name_tag} on gpu {gpu}")
    cmd = [
        "python",
        "-m",
        "fraudGT.main",
        "--cfg",
        cfg_path,
        "--repeat",
        "1",
        "out_dir",
        out_dir,
        "name_tag",
        name_tag,
    ]
    cmd.extend(extra_overrides)
    status = 1
    try:
        with open(run_log, "a") as fh:
            fh.write(f"\n[{now()}] START auto follow-up gpu={gpu}\n")
            fh.flush()
            status = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            fh.write(f"\n[{now()}] FINISH auto follow-up status={status} gpu={gpu}\n")
            fh.flush()
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass
    return status, gpu, out_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--primary-queue-log", required=True)
    parser.add_argument("--primary-run-log", required=True)
    parser.add_argument("--primary-label", required=True)
    parser.add_argument("--fallback-cfg", required=True)
    parser.add_argument("--fallback-name-tag", required=True)
    parser.add_argument("--fallback-out-dir", required=True)
    parser.add_argument("--fallback-run-log", required=True)
    parser.add_argument("--second-fallback-cfg")
    parser.add_argument("--second-fallback-name-tag")
    parser.add_argument("--second-fallback-out-dir")
    parser.add_argument("--second-fallback-run-log")
    parser.add_argument("--pretrained-source-queue-log")
    parser.add_argument("--pretrained-source-cfg")
    parser.add_argument("--pretrained-source-out-dir")
    parser.add_argument("--pretrained-source-name-tag")
    args = parser.parse_args()

    log(f"monitoring primary run {args.primary_label}")
    primary_status, _ = wait_for_queue_finish(args.primary_queue_log)
    primary_best = parse_best_test_f1(args.primary_run_log)
    log(f"primary finished status={primary_status} best_test_f1={primary_best:.6f}")
    if primary_best >= args.threshold:
        log("primary already meets target, no follow-up needed")
        return 0

    extra_overrides = []
    if args.pretrained_source_queue_log:
        if not all(
            [
                args.pretrained_source_cfg,
                args.pretrained_source_out_dir,
                args.pretrained_source_name_tag,
            ]
        ):
            raise ValueError("pretrained source args are incomplete")
        source_status, source_gpu = wait_for_queue_finish(args.pretrained_source_queue_log)
        source_out_dir = compute_run_dir(
            args.pretrained_source_cfg,
            args.pretrained_source_out_dir,
            args.pretrained_source_name_tag,
            source_gpu,
        )
        cfg_file = Path(source_out_dir) / "config.yaml"
        ckpt_dir = Path(source_out_dir) / "42" / "ckpt"
        if not cfg_file.exists() or not ckpt_dir.exists():
            raise FileNotFoundError(
                f"pretrained source incomplete: status={source_status} dir={source_out_dir}"
            )
        extra_overrides.extend(["pretrained.dir", source_out_dir])
        log(f"using pretrained source {source_out_dir}")

    stages = [
        {
            "label": "fallback",
            "cfg": args.fallback_cfg,
            "name_tag": args.fallback_name_tag,
            "out_dir": args.fallback_out_dir,
            "run_log": args.fallback_run_log,
            "extra_overrides": extra_overrides,
        }
    ]
    if any(
        [
            args.second_fallback_cfg,
            args.second_fallback_name_tag,
            args.second_fallback_out_dir,
            args.second_fallback_run_log,
        ]
    ):
        if not all(
            [
                args.second_fallback_cfg,
                args.second_fallback_name_tag,
                args.second_fallback_out_dir,
                args.second_fallback_run_log,
            ]
        ):
            raise ValueError("second fallback args are incomplete")
        stages.append(
            {
                "label": "second_fallback",
                "cfg": args.second_fallback_cfg,
                "name_tag": args.second_fallback_name_tag,
                "out_dir": args.second_fallback_out_dir,
                "run_log": args.second_fallback_run_log,
                "extra_overrides": [],
            }
        )

    for stage in stages:
        status, _, out_dir = run_experiment(
            stage["cfg"],
            stage["name_tag"],
            stage["out_dir"],
            stage["run_log"],
            stage["extra_overrides"],
        )
        best = parse_best_test_f1(stage["run_log"])
        log(
            f"{stage['label']} finished status={status} best_test_f1={best:.6f} "
            f"out_dir={out_dir}"
        )
        if best >= args.threshold:
            log(f"{stage['label']} meets target")
            return 0

    log("all configured stages finished without meeting target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
