#!/usr/bin/env python3
"""Audit raw peak test F1 against the six AML +0.02 targets.

Default roots are the active novel/protected experiment families only. Use
--include-reference to include historical baselines for diagnosis without
confusing them with the current completion state.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


TARGETS = {
    "Small-HI": 0.8013,
    "Small-LI": 0.5101,
    "Medium-HI": 0.7993,
    "Medium-LI": 0.4806,
    "Large-HI": 0.7734,
    "Large-LI": 0.4143,
}

ACTIVE_ROOTS = [
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_screen180_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertconsisboundresid_screen180_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertkeepboundresid_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgate_seed43_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgate_seed44_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgate_seed45_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgate_seed46_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgate_seed47_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertkeep_seed43_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertkeep_seed44_full240_main",
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertkeep_seed45_full240_main",
    "unified_classsplitsubgraphroute_smallli_seed43_full240_main",
    "unified_classsplitsubgraphroute_smallli_seed44_full240_main",
    "unified_classsplitsubgraphroute_smallli_seed45_full240_main",
    "classmixprotobound_largehi_seed43_full240_main",
    "classmixprotobound_largehi_seed44_full240_main",
    "classmixprotobound_largehi_seed45_full240_main",
    "classmixprotobound_largehi_selftune_seed42_120_main",
    "classmixprotobound_largeli_seed43_full240_main",
    "classmixprotobound_largeli_seed44_full240_main",
    "classmixprotobound_largeli_seed45_full240_main",
    "classmixprotobound_largeli_selftune_seed42_120_main",
]

REFERENCE_ROOTS = [
    "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid_full240_main",
    "unified_supportmixconsisclassmixprotoboundresid_full240_main",
    "flowsketch_rawpeak_screen_v1",
    "flowsketch_scorecalib_rawpeak_screen_v1",
    "flowsketch_delta_rawpeak_screen_v1",
    "flowsketch_delta_scorecalib_rawpeak_screen_v1",
    "classsplit_subgraph_flowsketch_rawpeak_screen_v1",
]


def dataset_from_run(run_name):
    for dataset in TARGETS:
        if dataset in run_name:
            return dataset
    return None


def iter_rows(stats_path):
    if not stats_path.exists():
        return
    for line in stats_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in row and "f1" in row:
            yield int(row["epoch"]), float(row["f1"])


def stats_state(stats_path):
    rows = list(iter_rows(stats_path))
    if not rows:
        return None
    best_epoch, best_f1 = max(rows, key=lambda item: item[1])
    last_epoch, last_f1 = rows[-1]
    return {
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "last_epoch": last_epoch,
        "last_f1": last_f1,
    }


def active_commands():
    result = subprocess.run(
        ["ps", "-eo", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        line for line in result.stdout.splitlines()
        if "fraudGT.main" in line
    ]


def active_for_run(run_dir, active):
    run_name = run_dir.name
    if any(run_name in cmd for cmd in active):
        return True
    match = re.match(r"^(?P<cfg_stem>.+)-gpu(?P<gpu>\d+)$", run_name)
    if not match:
        return False
    cfg_stem = match.group("cfg_stem")
    gpu = match.group("gpu")
    gpu_needles = (f"--gpu {gpu}", f"--gpu={gpu}")
    return any(cfg_stem in cmd and any(needle in cmd for needle in gpu_needles) for cmd in active)


def collect(root_paths):
    active = active_commands()
    rows = []
    for root in root_paths:
        for stats_path in sorted(root.glob("*/*/test/stats.json")):
            if not stats_path.parents[1].name.isdigit():
                continue
            run_dir = stats_path.parents[2]
            dataset = dataset_from_run(run_dir.name)
            if dataset is None:
                continue
            state = stats_state(stats_path)
            if state is None:
                continue
            is_active = active_for_run(run_dir, active)
            rows.append({
                "dataset": dataset,
                "root": root.name,
                "run_dir": str(run_dir),
                "active": is_active,
                **state,
            })
    return rows


def best_by_dataset(rows):
    best = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in best or row["best_f1"] > best[dataset]["best_f1"]:
            best[dataset] = row
    return best


def active_by_dataset(rows):
    active = {}
    for row in rows:
        if not row["active"]:
            continue
        dataset = row["dataset"]
        if dataset not in active or row["last_epoch"] > active[dataset]["last_epoch"]:
            active[dataset] = row
    return active


def print_table(best, active):
    header = (
        "dataset", "status", "best", "best_epoch", "last", "last_epoch",
        "target", "active_last", "active_epoch", "active_root", "root"
    )
    print("\t".join(header))
    passed = 0
    for dataset, target in TARGETS.items():
        row = best.get(dataset)
        active_row = active.get(dataset)
        active_last = "-" if active_row is None else f"{active_row['last_f1']:.5f}"
        active_epoch = "-" if active_row is None else str(active_row["last_epoch"])
        active_root = "-" if active_row is None else active_row["root"]
        if row is None:
            values = (
                dataset, "MISS", "-", "-", "-", "-", f"{target:.4f}",
                active_last, active_epoch, active_root, "-"
            )
        else:
            ok = row["best_f1"] >= target
            passed += int(ok)
            values = (
                dataset,
                "PASS" if ok else "FAIL",
                f"{row['best_f1']:.5f}",
                str(row["best_epoch"]),
                f"{row['last_f1']:.5f}",
                str(row["last_epoch"]),
                f"{target:.4f}",
                active_last,
                active_epoch,
                active_root,
                row["root"],
            )
        print("\t".join(values))
    print(f"summary\t{passed}/6")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/e/yyk/FraudGT_multi6",
        help="remote/local results base containing run roots",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="include historical reference roots for diagnosis",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 unless all six datasets pass",
    )
    args = parser.parse_args()

    root_names = list(ACTIVE_ROOTS)
    if args.include_reference:
        root_names.extend(REFERENCE_ROOTS)
    root_paths = [Path(args.base) / name for name in root_names]
    rows = collect(root_paths)
    passed = print_table(best_by_dataset(rows), active_by_dataset(rows))
    if args.strict and passed < len(TARGETS):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
