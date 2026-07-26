#!/usr/bin/env python3
"""Directional GTF1C v2 real-graph development screen."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.gtf1c import paired_policy_statistics  # noqa: E402
from run.gtf1c_phase0_real_graph import (  # noqa: E402
    CONDITIONS,
    METHODS,
    NEGATIVE_REGIMES,
    POSITIVE_REGIMES,
    REGIMES,
    apply_policy,
    candidate_policy_thresholds,
    certification_results,
    empty_accumulator,
    finite_or_none,
    generate_scores,
    load_dataset,
    move_view,
    policy_family_statistics,
    repository_commit,
    summarize_accumulator,
    verify_repository,
)


V2_REGIMES = copy.deepcopy(REGIMES)
for config in V2_REGIMES.values():
    config["offsets"] = {
        "selection": 0.0,
        "certification": 0.0,
        "evaluation": 0.0,
    }
V2_REGIMES["cpse_remove_fragility"]["offsets"].update({
    "certification": 2.0,
    "evaluation": 2.0,
})
V2_REGIMES["alignment_drift"]["offsets"].update({
    "certification": 4.0,
    "evaluation": 4.0,
})
V2_REGIMES["rare_duplicate"]["offsets"].update({
    "certification": 2.0,
    "evaluation": 2.0,
})


def generate_v2_scores(view, config, role, generator):
    scores = generate_scores(view, config, role, generator)
    offset = float(config["offsets"][role])
    if offset:
        for direction in scores.values():
            for condition in direction:
                direction[condition] = direction[condition] + offset
    return scores


def _best_threshold(
        view,
        scores,
        add_thresholds,
        remove_thresholds,
        min_changes,
):
    delta, changed = policy_family_statistics(
        view, scores, "normal", add_thresholds, remove_thresholds)
    valid = changed >= int(min_changes)
    if not valid.any():
        return -1
    ranked = torch.where(
        valid, delta, torch.full_like(delta, -math.inf))
    return int(ranked.argmax())


def directional_candidates(
        view, scores, quantiles, config, min_changes):
    add_family, remove_family = candidate_policy_thresholds(
        view, scores, quantiles, config)
    count = int(quantiles.numel())
    add_index = _best_threshold(
        view,
        scores,
        add_family[:count],
        remove_family[:count],
        min_changes,
    )
    remove_index = _best_threshold(
        view,
        scores,
        add_family[count:2 * count],
        remove_family[count:2 * count],
        min_changes,
    )
    add_threshold = (
        float(add_family[add_index])
        if add_index >= 0 else math.inf
    )
    remove_threshold = (
        float(remove_family[count + remove_index])
        if remove_index >= 0 else math.inf
    )
    candidates = (
        {
            "name": "add",
            "add_threshold": add_threshold,
            "remove_threshold": math.inf,
        },
        {
            "name": "remove",
            "add_threshold": math.inf,
            "remove_threshold": remove_threshold,
        },
        {
            "name": "joint",
            "add_threshold": add_threshold,
            "remove_threshold": remove_threshold,
        },
    )
    return candidates


def certify_candidates(
        view,
        scores,
        condition,
        candidates,
        min_changes,
        delta,
        practical_delta,
):
    results = []
    candidate_delta = float(delta) / len(candidates)
    for candidate in candidates:
        routed = apply_policy(
            view,
            scores,
            condition,
            candidate["add_threshold"],
            candidate["remove_threshold"],
        )
        results.append(certification_results(
            view,
            routed,
            min_changes,
            candidate_delta,
            practical_delta,
        ))
    return results


def choose_candidate(method, certifications):
    ranked = []
    for index, result in enumerate(certifications):
        if not result["qualifications"][method]:
            continue
        if method in ("row_net", "gtprc_row_harm"):
            score = float(result["statistics"]["net"])
        elif method == "iid_paired_f1":
            score = float(result["iid_f1"]["lower_bound"])
        elif method == "time_block_paired_f1":
            score = float(result["time_f1"]["lower_bound"])
        else:
            score = min(
                float(result["time_f1"]["lower_bound"]),
                float(result["graph_f1"]["lower_bound"]),
            )
        ranked.append((
            score,
            result["statistics"]["changed"],
            -index,
            index,
        ))
    return max(ranked)[-1] if ranked else -1


def evaluate_candidate(view, scores, condition, candidates, index):
    if index < 0:
        routed = view["base"]
    else:
        candidate = candidates[index]
        routed = apply_policy(
            view,
            scores,
            condition,
            candidate["add_threshold"],
            candidate["remove_threshold"],
        )
    return paired_policy_statistics(
        view["labels"], view["base"], routed)


def oracle_candidate(
        view,
        scores,
        candidates,
        min_changes,
        practical_delta,
):
    rows = [
        evaluate_candidate(view, scores, "normal", candidates, index)
        for index in range(len(candidates))
    ]
    valid = [
        (row["paired_f1_delta"], row["changed"], -index, index)
        for index, row in enumerate(rows)
        if row["changed"] >= int(min_changes)
        and row["paired_f1_delta"] >= float(practical_delta)
    ]
    if not valid:
        return -1, 0, 0.0
    index = max(valid)[-1]
    return index, rows[index]["changed"], rows[index]["paired_f1_delta"]


def run_dataset_v2(
        dataset,
        views,
        regime,
        seed,
        replicates,
        policy_count,
        min_changes,
        delta,
        practical_delta,
        device,
):
    config = V2_REGIMES[regime]
    methods = {method: empty_accumulator() for method in METHODS}
    controls = {
        condition: empty_accumulator()
        for condition in ("shuffled", "harmful")
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    quantiles = torch.linspace(
        0.50, 0.9995, int(policy_count), device=device)
    gpu_views = {
        fold: move_view(view, device)
        for fold, view in views.items()
    }
    records = []
    oracle_opportunities = 0

    for replicate in range(int(replicates)):
        for evaluation_fold in range(3):
            selection_fold = (evaluation_fold + 1) % 3
            certification_fold = (evaluation_fold + 2) % 3
            selection_view = gpu_views[selection_fold]
            certification_view = gpu_views[certification_fold]
            evaluation_view = gpu_views[evaluation_fold]
            selection_scores = generate_v2_scores(
                selection_view, config, "selection", generator)
            certification_scores = generate_v2_scores(
                certification_view, config, "certification", generator)
            evaluation_scores = generate_v2_scores(
                evaluation_view, config, "evaluation", generator)
            candidates = directional_candidates(
                selection_view,
                selection_scores,
                quantiles,
                config,
                min_changes,
            )
            normal_certifications = certify_candidates(
                certification_view,
                certification_scores,
                "normal",
                candidates,
                min_changes,
                delta,
                practical_delta,
            )
            chosen = {
                method: choose_candidate(
                    method, normal_certifications)
                for method in METHODS
            }
            oracle_index, oracle_changed, oracle_delta = oracle_candidate(
                evaluation_view,
                evaluation_scores,
                candidates,
                min_changes,
                practical_delta,
            )
            oracle_opportunities += int(oracle_index >= 0)

            evaluations = {}
            for method in METHODS:
                evaluation = evaluate_candidate(
                    evaluation_view,
                    evaluation_scores,
                    "normal",
                    candidates,
                    chosen[method],
                )
                evaluations[method] = evaluation
                row = methods[method]
                qualified = chosen[method] >= 0
                row["trials"] += 1
                row["qualified"] += int(qualified)
                row["violations"] += int(
                    qualified
                    and evaluation["paired_f1_delta"] <= 0.0)
                row["practical_failures"] += int(
                    qualified
                    and evaluation["paired_f1_delta"]
                    < float(practical_delta))
                row["evaluation_deltas"].append(
                    evaluation["paired_f1_delta"] if qualified else 0.0)
                row["evaluation_changes"].append(
                    evaluation["changed"] if qualified else 0)
                row["oracle_fractions"].append(
                    min(
                        1.0,
                        evaluation["changed"]
                        / max(oracle_changed, 1),
                    )
                    if qualified and oracle_changed > 0
                    else 0.0
                )

            gtf1c_index = chosen["gtf1c_graph_time"]
            for condition in ("shuffled", "harmful"):
                control_certifications = certify_candidates(
                    certification_view,
                    certification_scores,
                    condition,
                    candidates,
                    min_changes,
                    delta,
                    practical_delta,
                )
                qualified = (
                    gtf1c_index >= 0
                    and control_certifications[gtf1c_index][
                        "qualifications"
                    ]["gtf1c_graph_time"]
                )
                evaluation = evaluate_candidate(
                    evaluation_view,
                    evaluation_scores,
                    condition,
                    candidates,
                    gtf1c_index if qualified else -1,
                )
                row = controls[condition]
                row["trials"] += 1
                row["qualified"] += int(qualified)
                row["violations"] += int(
                    qualified
                    and evaluation["paired_f1_delta"] <= 0.0)
                row["practical_failures"] += int(
                    qualified
                    and evaluation["paired_f1_delta"]
                    < float(practical_delta))
                row["evaluation_deltas"].append(
                    evaluation["paired_f1_delta"] if qualified else 0.0)
                row["evaluation_changes"].append(
                    evaluation["changed"] if qualified else 0)
                row["oracle_fractions"].append(0.0)

            records.append({
                "dataset": dataset,
                "replicate": replicate,
                "evaluation_fold": evaluation_fold,
                "selection_fold": selection_fold,
                "certification_fold": certification_fold,
                "candidates": [
                    {
                        "name": candidate["name"],
                        "add_threshold": finite_or_none(
                            candidate["add_threshold"]),
                        "remove_threshold": finite_or_none(
                            candidate["remove_threshold"]),
                    }
                    for candidate in candidates
                ],
                "chosen_candidate": chosen,
                "oracle_candidate": oracle_index,
                "oracle_changed": oracle_changed,
                "oracle_delta": oracle_delta,
                "candidate_certifications": [
                    {
                        "statistics": result["statistics"],
                        "row_break_upper": result["row_break_upper"],
                        "grouped_break_upper": (
                            result["grouped_break_upper"]),
                        "iid_f1_lcb": finite_or_none(
                            result["iid_f1"]["lower_bound"]),
                        "time_f1_lcb": finite_or_none(
                            result["time_f1"]["lower_bound"]),
                        "graph_f1_lcb": finite_or_none(
                            result["graph_f1"]["lower_bound"]),
                        "qualifications": result["qualifications"],
                    }
                    for result in normal_certifications
                ],
                "normal_evaluations": evaluations,
            })

    trials = int(replicates) * 3
    return {
        "dataset": dataset,
        "trials": trials,
        "oracle_opportunities": oracle_opportunities,
        "oracle_opportunity_rate": oracle_opportunities / trials,
        "methods": {
            method: summarize_accumulator(row)
            for method, row in methods.items()
        },
        "controls": {
            condition: summarize_accumulator(row)
            for condition, row in controls.items()
        },
        "fold_diagnostics": {
            str(fold): {
                "rows": int(view["labels"].numel()),
                "positives": int(view["labels"].sum()),
                "base": paired_policy_statistics(
                    view["labels"], view["base"], view["base"]),
                "time_group_count": int(view["time_groups"].max()) + 1,
                "graph_time_group_count": (
                    int(view["graph_groups"].max()) + 1),
            }
            for fold, view in views.items()
        },
    }, records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    commit = verify_repository(args.expected_commit)
    spec = json.loads(args.spec.read_text())
    task = spec["tasks"][args.task_index]
    regime = task["regime"]
    if regime not in V2_REGIMES:
        raise ValueError("unknown GTF1C v2 regime")
    input_root = args.input_root or Path(spec["input_root"])
    replicates = args.replicates or int(spec["replicates"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    dataset_results = {}
    all_records = []
    source_manifests = {}
    for offset, dataset in enumerate(("Small-LI", "Large-LI")):
        views, manifests = load_dataset(
            input_root, dataset, spec["block_count"])
        result, records = run_dataset_v2(
            dataset=dataset,
            views=views,
            regime=regime,
            seed=int(task["seed"]) + offset * 10000,
            replicates=replicates,
            policy_count=spec["policy_count"],
            min_changes=spec["min_changes"],
            delta=spec["delta"],
            practical_delta=spec["practical_delta"],
            device=device,
        )
        dataset_results[dataset] = result
        all_records.extend(records)
        source_manifests[dataset] = manifests

    output_dir = args.output_root / (
        f"{regime}_seed{task['seed']}_{commit[:8]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "gtf1c_v2_trials.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in all_records
        )
    )
    manifest = {
        "experiment": "GTF1C_Phase0_v2_directional",
        "method_version": 2,
        "regime": regime,
        "regime_type": (
            "positive" if regime in POSITIVE_REGIMES else "negative"),
        "regime_config": V2_REGIMES[regime],
        "seed": task["seed"],
        "git_commit": commit,
        "sampling_protocol": "dynamic_random",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "replicates": replicates,
        "policy_count_per_direction": spec["policy_count"],
        "locked_candidate_count": 3,
        "candidate_delta": spec["delta"] / 3.0,
        "block_count": spec["block_count"],
        "min_changes": spec["min_changes"],
        "delta": spec["delta"],
        "practical_delta": spec["practical_delta"],
        "device": str(device),
        "source_oof_root": str(input_root),
        "source_manifests": source_manifests,
        "datasets": dataset_results,
        "trial_records": str(records_path),
        "runtime_seconds": time.time() - started,
    }
    manifest_path = output_dir / "gtf1c_phase0_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "regime": regime,
        "datasets": {
            dataset: {
                "oracle_rate": row["oracle_opportunity_rate"],
                "gtf1c": row["methods"]["gtf1c_graph_time"],
            }
            for dataset, row in dataset_results.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
