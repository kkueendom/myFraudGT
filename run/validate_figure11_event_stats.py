#!/usr/bin/env python3
"""Independently read back and validate Figure 11(a) export artifacts."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


DATASET = "Small-LI"
SPLIT = "test"
SEED = 42
LOADER_STEPS = 256
BATCH_SIZE = 2048
MAX_EVENTS = 48
RELATION_NAMES = {0: "O→O", 1: "I→I", 2: "O→I", 3: "I→O"}
CONTENT_FILES = (
    "export_figure11_event_stats.py",
    "target_event_counts.csv",
    "event_count_histogram.csv",
    "relation_counts.csv",
    "summary.json",
    "README.md",
)
ALL_FILES = set(CONTENT_FILES) | {"SHA256SUMS"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-config", type=Path, required=True)
    parser.add_argument("--expected-export-commit", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_linear(values, percentile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def close(left, right, tolerance=1e-12):
    return math.isclose(float(left), float(right), rel_tol=tolerance,
                        abs_tol=tolerance)


def read_target_counts(path):
    event_counts = []
    target_ids = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == [
            "dataset", "split", "seed", "target_edge_id", "event_count",
            "context_count", "reached_cap",
        ], "target_event_counts.csv headers differ")
        for row in reader:
            require(row["dataset"] == DATASET, "target row dataset differs")
            require(row["split"] == SPLIT, "target row split differs")
            require(int(row["seed"]) == SEED, "target row seed differs")
            target_id = int(row["target_edge_id"])
            require(target_id not in target_ids, "duplicate target edge ID")
            target_ids.add(target_id)
            event_count = int(row["event_count"])
            context_count = int(row["context_count"])
            require(1 <= event_count <= MAX_EVENTS,
                    "event count outside [1, 48]")
            require(context_count == event_count - 1,
                    "context count does not equal event count minus one")
            require(row["reached_cap"] in {"True", "False"},
                    "invalid reached_cap value")
            require((row["reached_cap"] == "True")
                    == (event_count == MAX_EVENTS),
                    "reached_cap does not match event count")
            event_counts.append(event_count)
    require(event_counts, "target_event_counts.csv is empty")
    return event_counts


def read_histogram(path, event_counts):
    observed = [0] * (MAX_EVENTS + 1)
    for count in event_counts:
        observed[count] += 1
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == [
            "event_count", "target_count", "target_share",
        ], "event_count_histogram.csv headers differ")
        rows = list(reader)
    require(len(rows) == MAX_EVENTS, "histogram must contain 48 rows")
    for expected_count, row in enumerate(rows, start=1):
        require(int(row["event_count"]) == expected_count,
                "histogram bins are not exactly 1 through 48")
        require(int(row["target_count"]) == observed[expected_count],
                "histogram count differs from target CSV")
        require(close(row["target_share"],
                      observed[expected_count] / len(event_counts)),
                "histogram share differs from target CSV")
    require(sum(int(row["target_count"]) for row in rows)
            == len(event_counts), "histogram target total differs")
    require(close(sum(float(row["target_share"]) for row in rows), 1.0),
            "histogram shares do not sum to one")
    return observed


def read_relations(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == [
            "relation_id", "relation_name", "edge_occurrence_count",
            "edge_occurrence_share",
        ], "relation_counts.csv headers differ")
        rows = list(reader)
    require(len(rows) == 4, "relation CSV must contain four rows")
    counts = {}
    shares = {}
    for expected_id, row in enumerate(rows):
        relation_id = int(row["relation_id"])
        require(relation_id == expected_id, "relation IDs differ")
        name = row["relation_name"]
        require(name == RELATION_NAMES[relation_id],
                "relation ID/name mapping differs")
        count = int(row["edge_occurrence_count"])
        share = float(row["edge_occurrence_share"])
        require(count >= 0 and 0.0 <= share <= 1.0,
                "invalid relation count/share")
        counts[name] = count
        shares[name] = share
    total = sum(counts.values())
    require(total > 0, "relation occurrence total is zero")
    for name in counts:
        require(close(shares[name], counts[name] / total),
                "relation share differs from count total")
    require(close(sum(shares.values()), 1.0),
            "relation shares do not sum to one")
    return counts, shares, total


def validate_summary(summary, event_counts, histogram, relation_counts,
                     relation_shares, relation_total, args):
    required = {
        "dataset", "split", "seed", "sampling_protocol", "loader_steps",
        "sampled_target_occurrences", "unique_target_count",
        "duplicate_target_occurrences", "history_k", "history_hops",
        "max_events", "time_window", "event_count_includes_target",
        "mean_event_count", "std_event_count", "median_event_count",
        "min_event_count", "max_event_count", "p25_event_count",
        "p75_event_count", "p90_event_count", "p95_event_count",
        "count_at_cap", "share_at_cap",
        "total_local_event_edge_occurrences", "relation_counts",
        "relation_shares", "git_commit", "config_path", "config_snapshot",
        "export_script_path", "created_at",
    }
    require(required <= set(summary), "summary.json lacks required fields")
    require(summary["dataset"] == DATASET and summary["split"] == SPLIT,
            "summary dataset/split differs")
    require(summary["seed"] == SEED, "summary seed differs")
    require(summary["sampling_protocol"] == "dynamic_random",
            "summary sampling protocol differs")
    require(summary["loader_steps"] == LOADER_STEPS,
            "summary loader steps differ")
    require(summary["loader_requested_target_occurrences"]
            == LOADER_STEPS * BATCH_SIZE,
            "loader request total differs from 256 x 2048")
    require(summary["sampled_target_occurrences"]
            + summary["requested_targets_absent_from_store_e_id"]
            == summary["loader_requested_target_occurrences"],
            "sampled plus absent does not cover loader requests")
    require(summary["sampled_plus_absent_equals_loader_requests"] is True,
            "loader request coverage flag is false")
    require(summary["unique_target_count"] == len(event_counts),
            "summary unique target count differs from target CSV")
    require(summary["duplicate_target_occurrences"]
            == summary["sampled_target_occurrences"] - len(event_counts),
            "summary duplicate target count differs")
    require((summary["history_k"], summary["history_hops"],
             summary["max_events"], summary["time_window"])
            == (4, 2, 48, None), "event graph settings differ")
    require(summary["event_count_includes_target"] is True,
            "event count target inclusion flag is false")
    mean = sum(event_counts) / len(event_counts)
    std = math.sqrt(sum((value - mean) ** 2 for value in event_counts)
                    / len(event_counts))
    require(close(summary["mean_event_count"], mean), "mean differs")
    require(close(summary["std_event_count"], std, 1e-11), "std differs")
    require(close(summary["median_event_count"],
                  percentile_linear(event_counts, 50)), "median differs")
    require(summary["min_event_count"] == min(event_counts), "min differs")
    require(summary["max_event_count"] == max(event_counts), "max differs")
    for name, percentile in (("p25_event_count", 25),
                             ("p75_event_count", 75),
                             ("p90_event_count", 90),
                             ("p95_event_count", 95)):
        require(close(summary[name], percentile_linear(event_counts, percentile)),
                f"{name} differs")
    cap_count = histogram[MAX_EVENTS]
    require(summary["count_at_cap"] == cap_count, "cap count differs")
    require(close(summary["share_at_cap"], cap_count / len(event_counts)),
            "cap share differs")
    require(summary["total_local_event_edge_occurrences"] == relation_total,
            "relation total differs")
    require(summary["relation_counts"] == relation_counts,
            "summary relation counts differ")
    for name in relation_shares:
        require(close(summary["relation_shares"][name], relation_shares[name]),
                "summary relation shares differ")
    require(summary["git_commit"] == args.expected_export_commit,
            "export commit differs")
    require(Path(summary["config_path"]) == args.expected_config.resolve(),
            "formal config path differs")
    snapshot = summary["config_snapshot"]
    require(snapshot["dataset"]["name"] == DATASET,
            "config snapshot dataset differs")
    require(snapshot["seed"] == SEED, "config snapshot seed differs")
    require(snapshot["train"]["batch_size"] == BATCH_SIZE,
            "config snapshot batch size differs")
    require(snapshot["train"]["iter_per_epoch"] == LOADER_STEPS,
            "config snapshot train steps differ")
    require(snapshot["val"]["iter_per_epoch"] == LOADER_STEPS,
            "config snapshot test steps differ")
    require(snapshot["val"]["fixed_target_panel"] is False,
            "fixed target panel is enabled")
    require((snapshot["cdvt"]["history_k"],
             snapshot["cdvt"]["history_hops"],
             snapshot["cdvt"]["max_events"],
             snapshot["cdvt"]["time_window"])
            == (4, 2, 48, -1), "config snapshot event settings differ")
    require(all(summary["config_validation_checks"].values()),
            "an exporter config validation failed")
    require(all(summary["loader_validation_checks"].values()),
            "an exporter loader validation failed")
    require(all(summary["validation_checks"].values()),
            "an exporter statistics validation failed")


def validate_checksums(output_dir):
    expected = {}
    for line in (output_dir / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    require(set(expected) == set(CONTENT_FILES),
            "SHA256SUMS file set differs")
    actual = {name: sha256(output_dir / name) for name in CONTENT_FILES}
    require(actual == expected, "a content SHA-256 differs")
    return actual


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    require(output_dir.is_dir(), "output directory does not exist")
    actual_files = {path.name for path in output_dir.iterdir()
                    if path.is_file()}
    require(actual_files == ALL_FILES, "output directory file set differs")
    event_counts = read_target_counts(output_dir / "target_event_counts.csv")
    histogram = read_histogram(
        output_dir / "event_count_histogram.csv", event_counts)
    relation_counts, relation_shares, relation_total = read_relations(
        output_dir / "relation_counts.csv")
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8"))
    validate_summary(summary, event_counts, histogram, relation_counts,
                     relation_shares, relation_total, args)
    checksums = validate_checksums(output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    require(sha256(repo_root / "run/export_figure11_event_stats.py")
            == checksums["export_figure11_event_stats.py"],
            "packaged export script differs from repository script")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_export_commit], cwd=repo_root,
        text=True).strip()
    require(source_commit == args.expected_export_commit,
            "expected export commit is unavailable in repository")
    print(json.dumps({
        "files_validated": len(ALL_FILES),
        "target_rows_validated": len(event_counts),
        "relation_occurrences_validated": relation_total,
        "sha256_entries_validated": len(checksums),
        "status": "PASS",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
