#!/usr/bin/env python3
"""Fast go/no-go audit for the Motif-Aware Graph Transformer.

Recursively finds every completed/partial run under the given roots (any dir
containing `val/stats.json`), computes the paper-consistent **val-selected test
F1** (`test_at_val`) and the diagnostic raw-best test F1, and prints a
motif-vs-baseline comparison per dataset.

Usage:
    python run/motif_quickcheck_audit.py --motif <motif_out_dir> --baseline <baseline_out_dir>
"""

import argparse
import json
import re
from pathlib import Path

DATASET_RE = re.compile(r"AML-(Small|Medium|Large)-(HI|LI)")


def rows(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            out.append(row)
    return out


def summarize_run(seed_dir):
    val_rows = rows(seed_dir / "val" / "stats.json")
    test_rows = rows(seed_dir / "test" / "stats.json")
    if not val_rows or not test_rows:
        return None
    test_by_epoch = {int(r["epoch"]): r for r in test_rows}
    best_val = max(val_rows, key=lambda r: float(r.get("f1", 0.0)))
    val_epoch = int(best_val["epoch"])
    test_at_val = test_by_epoch.get(val_epoch)
    raw_best = max(test_rows, key=lambda r: float(r.get("f1", 0.0)))
    return {
        "val_epoch": val_epoch,
        "val_f1": float(best_val.get("f1", 0.0)),
        "test_at_val_f1": float(test_at_val.get("f1", 0.0)) if test_at_val else None,
        "raw_best_test_f1": float(raw_best.get("f1", 0.0)),
        "last_epoch": max(int(r["epoch"]) for r in val_rows),
    }


def collect(root):
    """dataset -> best summary found under root (recursively)."""
    found = {}
    root = Path(root)
    for stats_path in root.rglob("val/stats.json"):
        seed_dir = stats_path.parent.parent  # dir holding val/ test/ train/
        m = DATASET_RE.search(str(seed_dir))
        if not m:
            continue
        dataset = f"{m.group(1)}-{m.group(2)}"
        summary = summarize_run(seed_dir)
        if summary is None:
            continue
        # keep the run that trained furthest for this dataset
        if dataset not in found or summary["last_epoch"] > found[dataset]["last_epoch"]:
            summary["run_dir"] = str(seed_dir)
            found[dataset] = summary
    return found


def fmt(x):
    return f"{x:.5f}" if isinstance(x, float) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", required=True, help="motif-on out_dir root")
    ap.add_argument("--baseline", required=True, help="motif-off out_dir root")
    args = ap.parse_args()

    motif = collect(args.motif)
    base = collect(args.baseline)
    datasets = sorted(set(motif) | set(base))

    print("| Dataset | motif Test@Val | base Test@Val | Δ Test@Val | motif raw-best | base raw-best | motif last-ep |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    deltas = []
    for ds in datasets:
        m = motif.get(ds)
        b = base.get(ds)
        mv = m["test_at_val_f1"] if m else None
        bv = b["test_at_val_f1"] if b else None
        delta = (mv - bv) if (mv is not None and bv is not None) else None
        if delta is not None:
            deltas.append(delta)
        print("| {ds} | {mv} | {bv} | {d} | {mr} | {br} | {le} |".format(
            ds=ds, mv=fmt(mv), bv=fmt(bv),
            d=(f"{delta:+.5f}" if delta is not None else "-"),
            mr=fmt(m["raw_best_test_f1"] if m else None),
            br=fmt(b["raw_best_test_f1"] if b else None),
            le=(m["last_epoch"] if m else "-"),
        ))
    print()
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        print(f"mean Δ Test@Val (motif - baseline): {mean_d:+.5f} over {len(deltas)} datasets; "
              f"motif wins {wins}/{len(deltas)}")
    else:
        print("No matched motif/baseline pairs found yet.")


if __name__ == "__main__":
    main()
