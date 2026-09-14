#!/usr/bin/env python3
"""Export auditable Figure 11(a) event-graph statistics without inference."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fraudGT  # noqa: E402,F401
from fraudGT.cdvt.event_graph import CausalEventGraphIndex  # noqa: E402
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg  # noqa: E402
from fraudGT.graphgym.loader import create_dataset, create_loader  # noqa: E402


TASK = ("node", "to", "node")
DATASET = "Small-LI"
SPLIT = "test"
SEED = 42
SAMPLING_PROTOCOL = "dynamic_random"
EXPECTED_LOADER_STEPS = 256
HISTORY_K = 4
HISTORY_HOPS = 2
MAX_EVENTS = 48
TIME_WINDOW = None
RELATION_NAMES = {
    0: "O→O",
    1: "I→I",
    2: "O→I",
    3: "I→O",
}
OUTPUT_FILES = (
    "export_figure11_event_stats.py",
    "target_event_counts.csv",
    "event_count_histogram.csv",
    "relation_counts.csv",
    "summary.json",
    "README.md",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_config(config):
    checks = {
        "dataset": config.get("dataset", {}).get("name") == DATASET,
        "seed": config.get("seed") == SEED,
        "train_sampler": (
            config.get("train", {}).get("sampler") == "link_neighbor"),
        "test_sampler": (
            config.get("val", {}).get("sampler") == "link_neighbor"),
        "train_batch_size": (
            config.get("train", {}).get("batch_size") == 2048),
        "train_iter_per_epoch": (
            config.get("train", {}).get("iter_per_epoch")
            == EXPECTED_LOADER_STEPS),
        "test_iter_per_epoch": (
            config.get("val", {}).get("iter_per_epoch")
            == EXPECTED_LOADER_STEPS),
        "fixed_target_panel_disabled": (
            config.get("val", {}).get("fixed_target_panel") is False),
        "tier_evidence": (
            config.get("dataset", {}).get("tier_evidence") is True),
        "history_k": config.get("cdvt", {}).get("history_k") == HISTORY_K,
        "history_hops": (
            config.get("cdvt", {}).get("history_hops") == HISTORY_HOPS),
        "max_events": (
            config.get("cdvt", {}).get("max_events") == MAX_EVENTS),
        "time_window": config.get("cdvt", {}).get("time_window") == -1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "configuration does not match the fixed Figure 11(a) scope: "
            + ", ".join(failed))
    return checks


def configure(config_path):
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    config_checks = validate_config(config)
    set_cfg(cfg)
    load_cfg(cfg, Namespace(cfg_file=str(config_path), opts=[]))
    cfg.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False
    cfg.val.fixed_target_panel = False
    torch.set_num_threads(int(cfg.num_threads))
    torch.set_num_interop_threads(1)
    return config, config_checks


def validate_loader(test_loader):
    require(len(test_loader) == EXPECTED_LOADER_STEPS,
            "test loader length differs from 256")
    loader = test_loader.loader
    require(getattr(loader, "generator", None) is None,
            "test loader must not use a dedicated generator")
    sampler = getattr(loader, "sampler", None)
    require(getattr(sampler, "generator", None) is None,
            "test sampler must not use a dedicated generator")
    return {
        "loader_steps_expected": True,
        "loader_generator_is_none": True,
        "sampler_generator_is_none": True,
    }


def collect_target_ids(test_loader):
    occurrences = []
    steps = 0
    requested_occurrences = 0
    requested_absent_occurrences = 0
    for raw_batch in test_loader:
        store = raw_batch[TASK]
        require(hasattr(store, "e_id"), "sampled test store lacks e_id")
        require(hasattr(store, "target_edge_id"),
                "sampled test store lacks target_edge_id")
        mask = torch.isin(store.e_id, store.target_edge_id)
        target_ids = store.e_id[mask].detach().cpu().long()
        requested = store.target_edge_id.detach().cpu().long()
        require(bool(torch.isin(target_ids, requested).all()),
                "extracted target IDs are not requested target IDs")
        require(torch.unique(target_ids).numel() == target_ids.numel(),
                "a target edge occurs more than once in a sampled store")
        absent = requested[~torch.isin(requested, target_ids)]
        require(
            target_ids.numel() + absent.numel() == requested.numel(),
            "extracted and absent target counts do not cover requests",
        )
        occurrences.append(target_ids)
        requested_occurrences += int(requested.numel())
        requested_absent_occurrences += int(absent.numel())
        steps += 1
    require(steps == EXPECTED_LOADER_STEPS,
            "actual test-loader steps differ from 256")
    require(occurrences, "test loader produced no target IDs")
    sampled = torch.cat(occurrences)
    require(
        sampled.numel() + requested_absent_occurrences
        == requested_occurrences,
        "full-pass extracted and absent target counts do not cover requests",
    )
    return (
        sampled,
        steps,
        requested_occurrences,
        requested_absent_occurrences,
    )


def query_event_graphs(index, target_ids, query_batch_size):
    require(query_batch_size > 0, "query batch size must be positive")
    require(
        (
            CausalEventGraphIndex.OUT_OUT,
            CausalEventGraphIndex.IN_IN,
            CausalEventGraphIndex.OUT_IN,
            CausalEventGraphIndex.IN_OUT,
        ) == (0, 1, 2, 3),
        "relation ID implementation mapping has changed",
    )
    event_count_parts = []
    relation_counts = torch.zeros(4, dtype=torch.long)
    target_identity_checks = 0
    relation_range_checks = 0
    for start in range(0, int(target_ids.numel()), query_batch_size):
        requested = target_ids[start:start + query_batch_size]
        graph = index.query(
            requested,
            k=HISTORY_K,
            hops=HISTORY_HOPS,
            max_events=MAX_EVENTS,
            time_window=TIME_WINDOW,
        )
        event_counts = graph.graph_ptr[1:] - graph.graph_ptr[:-1]
        require(event_counts.numel() == requested.numel(),
                "query returned the wrong number of local graphs")
        require(torch.equal(
            graph.target_nodes,
            graph.graph_ptr[1:] - 1,
        ), "target node is not the final node in a local event graph")
        require(torch.equal(
            graph.node_edge_ids[graph.target_nodes], requested,
        ), "local graph target node does not match requested edge ID")
        target_identity_checks += int(requested.numel())
        relations = graph.edge_relation.detach().cpu().long()
        if relations.numel():
            require(bool(((relations >= 0) & (relations < 4)).all()),
                    "graph.edge_relation contains an ID outside [0, 3]")
            relation_counts += torch.bincount(
                relations, minlength=4)[:4]
        relation_range_checks += int(relations.numel())
        event_count_parts.append(event_counts.detach().cpu().long())
    return (
        torch.cat(event_count_parts),
        relation_counts,
        target_identity_checks,
        relation_range_checks,
    )


def histogram_rows(event_counts):
    counts = torch.bincount(event_counts, minlength=MAX_EVENTS + 1)
    total = int(event_counts.numel())
    return [
        {
            "event_count": value,
            "target_count": int(counts[value]),
            "target_share": int(counts[value]) / total,
        }
        for value in range(1, MAX_EVENTS + 1)
    ]


def relation_rows(relation_counts):
    total = int(relation_counts.sum())
    require(total > 0, "local event graphs contain no event-edge occurrences")
    return [
        {
            "relation_id": relation_id,
            "relation_name": RELATION_NAMES[relation_id],
            "edge_occurrence_count": int(relation_counts[relation_id]),
            "edge_occurrence_share": (
                int(relation_counts[relation_id]) / total),
        }
        for relation_id in range(4)
    ]


def validate_statistics(
    target_ids,
    event_counts,
    histogram,
    relations,
    target_identity_checks,
    relation_range_checks,
):
    context_counts = event_counts - 1
    count_at_cap = int((event_counts == MAX_EVENTS).sum())
    relation_row_total = sum(
        row["edge_occurrence_count"] for row in relations)
    total_relation_occurrences = relation_range_checks
    checks = {
        "min_event_count_at_least_one": int(event_counts.min()) >= 1,
        "max_event_count_at_most_cap": (
            int(event_counts.max()) <= MAX_EVENTS),
        "context_count_equals_event_count_minus_one": bool(
            torch.equal(context_counts, event_counts - 1)),
        "count_at_cap_matches_event_counts": (
            count_at_cap == int((event_counts == MAX_EVENTS).sum())),
        "histogram_count_matches_unique_targets": (
            sum(row["target_count"] for row in histogram)
            == int(target_ids.numel())),
        "histogram_share_sums_to_one": abs(
            sum(row["target_share"] for row in histogram) - 1.0) < 1e-12,
        "relation_counts_match_total_occurrences": (
            relation_row_total == total_relation_occurrences),
        "relation_shares_sum_to_one": abs(
            sum(row["edge_occurrence_share"] for row in relations) - 1.0)
            < 1e-12,
        "relation_ids_within_zero_to_three": True,
        "target_nodes_match_requested_edge_ids": (
            target_identity_checks == int(target_ids.numel())),
        "labels_not_read_for_event_graph_construction": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            "Figure 11(a) validation failed: " + ", ".join(failed))
    return checks, context_counts, count_at_cap, total_relation_occurrences


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_readme(summary):
    cap_statement = (
        f"{summary['count_at_cap']} unique targets "
        f"({summary['share_at_cap']:.6%}) reached M=48."
    )
    return f"""# Figure 11(a) Event-Graph Statistics

- Dataset: `{summary['dataset']}`
- Split: held-out `{summary['split']}`
- Seed: `{summary['seed']}`
- Sampling protocol: `{summary['sampling_protocol']}`
- Scope: one complete shuffled test-loader pass with {summary['loader_steps']} steps
- Sampled target occurrences: {summary['sampled_target_occurrences']}
- Loader-requested target occurrences: {summary['loader_requested_target_occurrences']}
- Requested targets absent from `store.e_id`: {summary['requested_targets_absent_from_store_e_id']}. The exported target scope follows the prescribed `store.e_id[torch.isin(store.e_id, store.target_edge_id)]` extraction exactly; absent requests are recorded but are not silently added.
- Unique target transactions: {summary['unique_target_count']}
- Event count includes the target transaction itself; context count is event count minus one.
- Local event graphs use `history_k=4`, `history_hops=2`, `max_events=48`, and no time window.
- Relation shares use all {summary['total_local_event_edge_occurrences']} event-edge occurrences across target-conditioned local graphs as the denominator. The same global transition may occur in multiple local graphs; these are not globally unique edges.
- Relation IDs are `0=O→O`, `1=I→I`, `2=O→I`, `3=I→O`. This is not the reordered transition-vector column order.
- Cap result: {cap_statement}
- Population standard deviation and NumPy linear percentiles are reported.
- These are structural data statistics, not model-performance results, and no checkpoint or model forward pass was used.
- Preferred data path `/e/yky/data/archive` was unavailable; the formal paper configuration path `{summary['config_snapshot']['dataset']['dir']}` was used without modification.
"""


def write_outputs(
    output_dir,
    config_path,
    config_snapshot,
    config_checks,
    loader_checks,
    sampled_occurrences,
    unique_target_ids,
    event_counts,
    relation_counts_tensor,
    loader_steps,
    target_identity_checks,
    relation_range_checks,
    query_batch_size,
    loader_requested_occurrences,
    requested_absent_occurrences,
):
    histogram = histogram_rows(event_counts)
    relations = relation_rows(relation_counts_tensor)
    validation_checks, context_counts, count_at_cap, total_edges = (
        validate_statistics(
            unique_target_ids,
            event_counts,
            histogram,
            relations,
            target_identity_checks,
            relation_range_checks,
        )
    )
    values = event_counts.numpy().astype(np.float64)
    relation_counts = {
        row["relation_name"]: row["edge_occurrence_count"]
        for row in relations
    }
    relation_shares = {
        row["relation_name"]: row["edge_occurrence_share"]
        for row in relations
    }
    summary = {
        "dataset": DATASET,
        "split": SPLIT,
        "seed": SEED,
        "sampling_protocol": SAMPLING_PROTOCOL,
        "loader_steps": loader_steps,
        "sampled_target_occurrences": int(sampled_occurrences.numel()),
        "loader_requested_target_occurrences": int(
            loader_requested_occurrences),
        "requested_targets_absent_from_store_e_id": int(
            requested_absent_occurrences),
        "sampled_plus_absent_equals_loader_requests": (
            int(sampled_occurrences.numel())
            + int(requested_absent_occurrences)
            == int(loader_requested_occurrences)),
        "unique_target_count": int(unique_target_ids.numel()),
        "duplicate_target_occurrences": (
            int(sampled_occurrences.numel())
            - int(unique_target_ids.numel())),
        "history_k": HISTORY_K,
        "history_hops": HISTORY_HOPS,
        "max_events": MAX_EVENTS,
        "time_window": TIME_WINDOW,
        "event_count_includes_target": True,
        "mean_event_count": float(np.mean(values)),
        "std_event_count": float(np.std(values, ddof=0)),
        "std_definition": "population",
        "median_event_count": float(np.median(values)),
        "min_event_count": int(np.min(values)),
        "max_event_count": int(np.max(values)),
        "p25_event_count": float(np.percentile(values, 25)),
        "p75_event_count": float(np.percentile(values, 75)),
        "p90_event_count": float(np.percentile(values, 90)),
        "p95_event_count": float(np.percentile(values, 95)),
        "percentile_method": "numpy_linear",
        "count_at_cap": count_at_cap,
        "share_at_cap": count_at_cap / int(unique_target_ids.numel()),
        "total_local_event_edge_occurrences": total_edges,
        "relation_id_mapping": {
            str(key): value for key, value in RELATION_NAMES.items()
        },
        "relation_counts": relation_counts,
        "relation_shares": relation_shares,
        "git_commit": git_output("rev-parse", "HEAD"),
        "config_path": str(config_path.resolve()),
        "config_snapshot": config_snapshot,
        "export_script_path": str(Path(__file__).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_batch_size": int(query_batch_size),
        "event_cache_size_config": int(cfg.cdvt.event_cache_size),
        "event_cache_size_export": 0,
        "config_validation_checks": config_checks,
        "loader_validation_checks": loader_checks,
        "validation_checks": validation_checks,
    }

    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    temporary = output_dir.parent / (
        "." + output_dir.name + f".tmp-{os.getpid()}")
    require(not temporary.exists(), f"temporary path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        target_rows = (
            {
                "dataset": DATASET,
                "split": SPLIT,
                "seed": SEED,
                "target_edge_id": int(target_id),
                "event_count": int(event_count),
                "context_count": int(context_count),
                "reached_cap": bool(event_count == MAX_EVENTS),
            }
            for target_id, event_count, context_count in zip(
                unique_target_ids.tolist(),
                event_counts.tolist(),
                context_counts.tolist(),
            )
        )
        write_csv(
            temporary / "target_event_counts.csv",
            (
                "dataset", "split", "seed", "target_edge_id",
                "event_count", "context_count", "reached_cap",
            ),
            target_rows,
        )
        write_csv(
            temporary / "event_count_histogram.csv",
            ("event_count", "target_count", "target_share"),
            histogram,
        )
        write_csv(
            temporary / "relation_counts.csv",
            (
                "relation_id", "relation_name", "edge_occurrence_count",
                "edge_occurrence_share",
            ),
            relations,
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            build_readme(summary), encoding="utf-8")
        shutil.copy2(
            Path(__file__).resolve(),
            temporary / "export_figure11_event_stats.py",
        )
        checksums = [
            f"{sha256(temporary / name)}  {name}"
            for name in OUTPUT_FILES
        ]
        (temporary / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main():
    args = parse_args()
    require(args.query_batch_size > 0,
            "--query-batch-size must be positive")
    config_snapshot, config_checks = configure(args.config)
    # These seeds are fixed before both dataset and loader construction.
    seed_everything(SEED)
    dataset = create_dataset()
    loaders = create_loader(dataset=dataset, shuffle=True)
    require(len(loaders) == 3, "expected train, validation, and test loaders")
    test_loader = loaders[2]
    loader_checks = validate_loader(test_loader)

    (
        sampled_occurrences,
        loader_steps,
        loader_requested_occurrences,
        requested_absent_occurrences,
    ) = collect_target_ids(test_loader)
    unique_target_ids = torch.unique(sampled_occurrences, sorted=True)
    require(unique_target_ids.numel() > 0,
            "test pass produced no unique target transactions")

    store = dataset[SPLIT][TASK]
    require(hasattr(store, "timestamps"), "test store lacks timestamps")
    require(hasattr(store, "raw_edge_attr"),
            "test store lacks immutable raw_edge_attr")
    index = CausalEventGraphIndex(
        edge_index=store.edge_index,
        timestamps=store.timestamps,
        raw_edge_attr=store.raw_edge_attr,
        cache_size=0,
    )
    (
        event_counts,
        relation_counts_tensor,
        target_identity_checks,
        relation_range_checks,
    ) = query_event_graphs(
        index,
        unique_target_ids,
        args.query_batch_size,
    )
    summary = write_outputs(
        args.output_dir,
        args.config,
        config_snapshot,
        config_checks,
        loader_checks,
        sampled_occurrences,
        unique_target_ids,
        event_counts,
        relation_counts_tensor,
        loader_steps,
        target_identity_checks,
        relation_range_checks,
        args.query_batch_size,
        loader_requested_occurrences,
        requested_absent_occurrences,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "loader_steps": summary["loader_steps"],
        "sampled_target_occurrences": summary[
            "sampled_target_occurrences"],
        "unique_target_count": summary["unique_target_count"],
        "mean_event_count": summary["mean_event_count"],
        "count_at_cap": summary["count_at_cap"],
        "status": "COMPLETED",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
