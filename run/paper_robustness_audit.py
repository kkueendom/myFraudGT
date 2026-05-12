#!/usr/bin/env python3
"""Audit seed robustness for the scale-adaptive paper result.

Reference rows are the already selected main-table seeds. Queued rows are the
additional seed44/45 runs needed to finish the formal robustness package.
"""

import argparse
import json
from pathlib import Path


TARGETS = {
    "Small-HI": 0.8013,
    "Large-HI": 0.7734,
    "Large-LI": 0.4143,
}

REFERENCE_RUNS = [
    {
        "dataset": "Small-HI",
        "seed": "42",
        "done_epoch": 179,
        "structure": "full_dual_uncert_subgraph_gate",
        "root": "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_screen180_main",
        "run": "AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180-gpu1",
    },
    {
        "dataset": "Large-HI",
        "seed": "43",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largehi_seed43_full240_main",
        "run": "AML-Large-HI-ClassMixProtoBoundResid240Seed43-gpu0",
    },
    {
        "dataset": "Large-LI",
        "seed": "43",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largeli_seed43_full240_main",
        "run": "AML-Large-LI-ClassMixProtoBoundResid240Seed43-gpu1",
    },
]

QUEUED_RUNS = [
    {
        "dataset": "Small-HI",
        "seed": "44",
        "done_epoch": 179,
        "structure": "full_dual_uncert_subgraph_gate",
        "root": "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed44_screen180_main",
        "run": "AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44-gpu0",
    },
    {
        "dataset": "Small-HI",
        "seed": "45",
        "done_epoch": 179,
        "structure": "full_dual_uncert_subgraph_gate",
        "root": "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed45_screen180_main",
        "run": "AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed45-gpu1",
    },
    {
        "dataset": "Large-HI",
        "seed": "44",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largehi_seed44_full240_main",
        "run": "AML-Large-HI-ClassMixProtoBoundResid240Seed44-gpu0",
    },
    {
        "dataset": "Large-HI",
        "seed": "45",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largehi_seed45_full240_main",
        "run": "AML-Large-HI-ClassMixProtoBoundResid240Seed45-gpu1",
    },
    {
        "dataset": "Large-LI",
        "seed": "44",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largeli_seed44_full240_main",
        "run": "AML-Large-LI-ClassMixProtoBoundResid240Seed44-gpu0",
    },
    {
        "dataset": "Large-LI",
        "seed": "45",
        "done_epoch": 239,
        "structure": "scale_fallback_classmix_proto_bound_resid",
        "root": "classmixprotobound_largeli_seed45_full240_main",
        "run": "AML-Large-LI-ClassMixProtoBoundResid240Seed45-gpu1",
    },
]

SUPPLEMENTAL_RUNS = [
    {
        "dataset": "Small-HI",
        "seed": "44-halfbatch",
        "done_epoch": 179,
        "structure": "full_gate_resource_adapted_halfbatch",
        "root": "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed44_halfbatch_screen180_main",
        "run": "AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44-gpu0",
        "seed_dir": "44",
    },
    {
        "dataset": "Small-HI",
        "seed": "45-halfbatch",
        "done_epoch": 179,
        "structure": "full_gate_resource_adapted_halfbatch",
        "root": "unified_supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid_seed45_halfbatch_screen180_main",
        "run": "AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed45-gpu1",
        "seed_dir": "45",
    },
]


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
    train_rows = list(iter_rows(stats_path.parents[1] / "train" / "stats.json"))
    train_last_epoch = train_rows[-1][0] if train_rows else None
    return {
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "last_epoch": last_epoch,
        "last_f1": last_f1,
        "train_last_epoch": train_last_epoch,
    }


def print_row(base, spec):
    target = TARGETS[spec["dataset"]]
    stats_path = (
        Path(base)
        / spec["root"]
        / spec["run"]
        / spec.get("seed_dir", spec["seed"])
        / "test"
        / "stats.json"
    )
    state = stats_state(stats_path)
    if state is None:
        values = (
            spec["dataset"],
            spec["seed"],
            "MISS",
            "-",
            "-",
            "-",
            "-",
            f"{target:.4f}",
            "-",
            "no",
            spec["structure"],
            spec["root"],
        )
        print("\t".join(values))
        return {"status": "MISS", "complete": "no"}

    margin = state["best_f1"] - target
    status = "PASS" if margin >= 0 else "FAIL"
    complete = (
        state["train_last_epoch"] is not None
        and state["train_last_epoch"] >= spec["done_epoch"]
    )
    values = (
        spec["dataset"],
        spec["seed"],
        status,
        f"{state['best_f1']:.5f}",
        str(state["best_epoch"]),
        f"{state['last_f1']:.5f}",
        str(state["last_epoch"]),
        f"{target:.4f}",
        f"{margin:+.5f}",
        "yes" if complete else "no",
        spec["structure"],
        spec["root"],
    )
    print("\t".join(values))
    return {"status": status, "complete": "yes" if complete else "no"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/e/yyk/FraudGT_multi6")
    parser.add_argument("--strict-complete", action="store_true")
    args = parser.parse_args()

    header = (
        "dataset",
        "seed",
        "status",
        "best",
        "best_epoch",
        "last",
        "last_epoch",
        "target",
        "margin",
        "complete",
        "structure",
        "root",
    )
    print("\t".join(header))

    print("reference_runs")
    for spec in REFERENCE_RUNS:
        print_row(args.base, spec)

    print("queued_runs")
    rows_by_dataset = {dataset: [] for dataset in TARGETS}
    pass_count = 0
    complete_count = 0
    for spec in QUEUED_RUNS:
        row = print_row(args.base, spec)
        rows_by_dataset[spec["dataset"]].append(row)
        pass_count += int(row["status"] == "PASS")
        complete_count += int(row["complete"] == "yes")

    dataset_complete = 0
    dataset_all_pass = 0
    for dataset, rows in rows_by_dataset.items():
        complete = all(row["complete"] == "yes" for row in rows)
        all_pass = all(row["status"] == "PASS" for row in rows)
        dataset_complete += int(complete)
        dataset_all_pass += int(all_pass)
        print(
            f"{dataset}_summary\tcomplete={int(complete)}/1\t"
            f"all_pass={int(all_pass)}/1"
        )
    print(f"seed_complete_summary\t{complete_count}/{len(QUEUED_RUNS)}")
    print(f"seed_pass_summary\t{pass_count}/{len(QUEUED_RUNS)}")
    print(f"dataset_complete_summary\t{dataset_complete}/{len(TARGETS)}")
    print(f"dataset_all_pass_summary\t{dataset_all_pass}/{len(TARGETS)}")

    print("supplemental_resource_adapted_runs")
    for spec in SUPPLEMENTAL_RUNS:
        print_row(args.base, spec)

    if args.strict_complete and dataset_complete < len(TARGETS):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
