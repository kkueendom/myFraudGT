#!/usr/bin/env python3
"""Compare M1, v3, v4 no-gate, and v4 router at an 80-epoch budget."""

import argparse
import json
from pathlib import Path


DATASETS = ["Small-HI", "Small-LI"]
SEED = 42
EPOCH_LIMIT = 79


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            output.append(row)
    return output


def find_test_stats(roots, dataset):
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob(f"**/{SEED}/test/stats.json"):
            if f"AML-{dataset}" in str(path) or dataset.lower() in str(path).lower():
                matches.append(path)
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)


def summarize(roots, dataset):
    paths = find_test_stats(roots, dataset)
    if not paths:
        return None
    path = paths[0]
    eligible = [
        row for row in rows(path)
        if int(row.get("epoch", -1)) <= EPOCH_LIMIT
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda row: float(row.get("f1", 0.0)))
    return {
        "epoch": int(best["epoch"]),
        "f1": float(best.get("f1", 0.0)),
        "path": str(path),
    }


def existing_roots(repo_names, result_name):
    candidates = []
    for repo in repo_names:
        candidates.append(Path(repo) / "results" / result_name)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v4-repo",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--old-repos",
        nargs="+",
        default=[
            "/e/yky/FraudGT_feature_evidence_gate_decoder",
            "/e/yyk/FraudGT_evidence_gate_decoder",
        ],
    )
    args = parser.parse_args()

    v4_repo = Path(args.v4_repo)
    screen_root = v4_repo / "results" / "evidence_gate_v4_screen"
    variant_roots = {
        "M1 proto": existing_roots(args.old_repos, "evidence_gate_m1_proto"),
        "v3 convex gate": (
            existing_roots(args.old_repos, "evidence_gate_v3")
            + existing_roots(args.old_repos, "evidence_gate_v3_full_small")
        ),
        "v4 no-gate": [
            screen_root / "nogate-small-hi",
            screen_root / "nogate-small-li",
        ],
        "v4 residual router": [
            screen_root / "router-small-hi",
            screen_root / "router-small-li",
        ],
    }

    summaries = {}
    print("| Variant | Dataset | Raw-best test F1 (epoch <= 79) | Epoch |")
    print("|---|---|---:|---:|")
    for variant, roots in variant_roots.items():
        for dataset in DATASETS:
            summary = summarize(roots, dataset)
            summaries[(variant, dataset)] = summary
            if summary is None:
                print(f"| {variant} | {dataset} | - | - |")
            else:
                print(
                    f"| {variant} | {dataset} | {summary['f1']:.5f} | "
                    f"{summary['epoch']} |"
                )

    print()
    print("| Variant | Mean raw-best F1 | Mean delta vs M1 |")
    print("|---|---:|---:|")
    m1_values = [
        summaries[("M1 proto", dataset)]["f1"]
        for dataset in DATASETS
        if summaries[("M1 proto", dataset)] is not None
    ]
    m1_mean = sum(m1_values) / len(m1_values) if len(m1_values) == 2 else None
    for variant in variant_roots:
        values = [
            summaries[(variant, dataset)]["f1"]
            for dataset in DATASETS
            if summaries[(variant, dataset)] is not None
        ]
        if len(values) != 2:
            print(f"| {variant} | - | - |")
            continue
        mean_f1 = sum(values) / len(values)
        delta = mean_f1 - m1_mean if m1_mean is not None else None
        delta_text = f"{delta:+.5f}" if delta is not None else "-"
        print(f"| {variant} | {mean_f1:.5f} | {delta_text} |")


if __name__ == "__main__":
    main()
