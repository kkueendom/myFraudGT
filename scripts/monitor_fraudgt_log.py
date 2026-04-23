#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path


EPOCH_RE = re.compile(
    r"> Epoch (?P<epoch>\d+):.*Best so far: epoch (?P<best_epoch>\d+)"
    r".*val_f1: (?P<val>[0-9.]+).*test_f1: (?P<test>[0-9.]+)"
)


def parse_latest(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    latest = None
    for line in text.splitlines():
        match = EPOCH_RE.search(line)
        if match:
            latest = {
                "epoch": int(match.group("epoch")),
                "best_epoch": int(match.group("best_epoch")),
                "val_f1": float(match.group("val")),
                "test_f1": float(match.group("test")),
            }
    return latest


def format_status(status, total_epochs):
    epoch = status["epoch"]
    best_epoch = status["best_epoch"]
    val_f1 = status["val_f1"]
    test_f1 = status["test_f1"]
    parts = [
        f"epoch={epoch}",
        f"best_epoch={best_epoch}",
        f"best_val_f1={val_f1:.4f}",
        f"best_test_f1={test_f1:.4f}",
    ]
    if total_epochs is not None and epoch > 0:
        remaining = max(total_epochs - (epoch + 1), 0)
        progress = (epoch + 1) / float(total_epochs)
        parts.append(f"progress={progress:.2%}")
        parts.append(f"remaining_epochs={remaining}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Monitor FraudGT autorun logs.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--sleep-secs", type=int, default=300)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--max-polls", type=int, default=None)
    args = parser.parse_args()

    polls = 0
    previous_epoch = None
    while True:
        status = parse_latest(args.log_path)
        if status is None:
            print(f"log_missing={args.log_path}", flush=True)
        elif status["epoch"] != previous_epoch:
            print(format_status(status, args.total_epochs), flush=True)
            previous_epoch = status["epoch"]
        else:
            print(f"no_new_epoch | epoch={status['epoch']}", flush=True)

        polls += 1
        if not args.wait:
            break
        if args.max_polls is not None and polls >= args.max_polls:
            break
        time.sleep(max(args.sleep_secs, 1))


if __name__ == "__main__":
    main()
