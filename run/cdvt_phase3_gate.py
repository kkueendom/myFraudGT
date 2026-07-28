#!/usr/bin/env python3
"""Require the frozen Phase 2 gate before any CDVT follow-up training."""

import argparse
import json
from pathlib import Path

from run.cdvt_phase2_summary import build_summary


def require_phase2_gate(phase1_root, phase2_root):
    summary = build_summary(Path(phase1_root), Path(phase2_root))
    gate = summary["phase2_gate"]
    if not gate["advance_to_phase3"]:
        val = summary["val_selected_summary"]
        raise RuntimeError(
            "Phase 2 gate failed: "
            f"wins={val['wins']}, mean_delta={val['mean_delta']:+.5f}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = require_phase2_gate(args.phase1_root, args.phase2_root)
    print(json.dumps({
        "advance_to_phase3": True,
        "mean_delta": summary["val_selected_summary"]["mean_delta"],
        "wins": summary["val_selected_summary"]["wins"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
