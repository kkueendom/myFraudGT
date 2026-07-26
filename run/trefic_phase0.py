#!/usr/bin/env python3
"""TREFIC real-graph development screen on locked GTF1C v2 regimes."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.gtf1c import paired_policy_statistics  # noqa: E402
from fraudGT.evidence.trefic import (  # noqa: E402
    combined_ratio_upper,
    ratio_envelope_certification,
)
from run.gtf1c_phase0_real_graph import (  # noqa: E402
    NEGATIVE_REGIMES,
    POSITIVE_REGIMES,
    apply_policy,
    empty_accumulator,
    finite_or_none,
    load_dataset,
    move_view,
    repository_commit,
    summarize_accumulator,
    verify_repository,
)
from run.gtf1c_phase0_v2 import (  # noqa: E402
    V2_REGIMES,
    certify_candidates,
    choose_candidate,
    directional_candidates,
    evaluate_candidate,
    generate_v2_scores,
    oracle_candidate,
)


REFERENCE_METHODS = (
    "row_net",
    "gtprc_row_harm",
    "iid_paired_f1",
    "time_block_paired_f1",
    "gtf1c_graph_time",
)
METHODS = REFERENCE_METHODS + ("trefic",)


def certify_trefic_candidates(
        view,
        scores,
        condition,
        candidates,
        rho_upper,
        min_changes,
        delta,
        practical_delta,
):
    endpoint_delta = float(delta) / (len(candidates) * 2)
    results = []
    for candidate in candidates:
        routed = apply_policy(
            view,
            scores,
            condition,
            candidate["add_threshold"],
            candidate["remove_threshold"],
        )
        results.append(ratio_envelope_certification(
            view,
            routed,
            rho_upper=rho_upper,
            min_changes=min_changes,
            delta=endpoint_delta,
            practical_delta=practical_delta,
        ))
    return results


def choose_trefic_candidate(certifications):
    ranked = [
        (
            float(result["worst_lower_bound"]),
            result["statistics"]["changed"],
            -index,
            index,
        )
        for index, result in enumerate(certifications)
        if result["qualified"]
    ]
    return max(ranked)[-1] if ranked else -1


def _record_accumulator(
        accumulator,
        qualified,
        evaluation,
        practical_delta,
        oracle_changed,
):
    accumulator["trials"] += 1
    accumulator["qualified"] += int(qualified)
    accumulator["violations"] += int(
        qualified and evaluation["paired_f1_delta"] <= 0.0)
    accumulator["practical_failures"] += int(
        qualified
        and evaluation["paired_f1_delta"] < float(practical_delta))
    accumulator["evaluation_deltas"].append(
        evaluation["paired_f1_delta"] if qualified else 0.0)
    accumulator["evaluation_changes"].append(
        evaluation["changed"] if qualified else 0)
    accumulator["oracle_fractions"].append(
        min(1.0, evaluation["changed"] / max(oracle_changed, 1))
        if qualified and oracle_changed > 0
        else 0.0
    )


def _serializable_trefic(result):
    return {
        "statistics": result["statistics"],
        "directional_counts": result["directional_counts"],
        "rho_upper": result["rho_upper"],
        "worst_lower_bound": finite_or_none(
            result["worst_lower_bound"]),
        "qualified": result["qualified"],
        "endpoint_results": {
            name: {
                "rho": endpoint["rho"],
                "sum": endpoint["sum"],
                "exact_sum": endpoint["exact_sum"],
                "time": {
                    **endpoint["time"],
                    "standard_error": finite_or_none(
                        endpoint["time"]["standard_error"]),
                    "lower_bound": finite_or_none(
                        endpoint["time"]["lower_bound"]),
                },
                "graph": {
                    **endpoint["graph"],
                    "standard_error": finite_or_none(
                        endpoint["graph"]["standard_error"]),
                    "lower_bound": finite_or_none(
                        endpoint["graph"]["lower_bound"]),
                },
            }
            for name, endpoint in result["endpoint_results"].items()
        },
    }


def run_dataset_trefic(
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

            reference_certifications = certify_candidates(
                certification_view,
                certification_scores,
                "normal",
                candidates,
                min_changes,
                delta,
                practical_delta,
            )
            rho = combined_ratio_upper(
                (selection_view, certification_view), delta=delta)
            trefic_certifications = certify_trefic_candidates(
                certification_view,
                certification_scores,
                "normal",
                candidates,
                rho["upper"],
                min_changes,
                delta,
                practical_delta,
            )
            chosen = {
                method: choose_candidate(
                    method, reference_certifications)
                for method in REFERENCE_METHODS
            }
            chosen["trefic"] = choose_trefic_candidate(
                trefic_certifications)

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
                _record_accumulator(
                    methods[method],
                    chosen[method] >= 0,
                    evaluation,
                    practical_delta,
                    oracle_changed,
                )

            trefic_index = chosen["trefic"]
            control_records = {}
            for condition in ("shuffled", "harmful"):
                control_certifications = certify_trefic_candidates(
                    certification_view,
                    certification_scores,
                    condition,
                    candidates,
                    rho["upper"],
                    min_changes,
                    delta,
                    practical_delta,
                )
                qualified = (
                    trefic_index >= 0
                    and control_certifications[trefic_index]["qualified"]
                )
                evaluation = evaluate_candidate(
                    evaluation_view,
                    evaluation_scores,
                    condition,
                    candidates,
                    trefic_index if qualified else -1,
                )
                _record_accumulator(
                    controls[condition],
                    qualified,
                    evaluation,
                    practical_delta,
                    oracle_changed=0,
                )
                control_records[condition] = {
                    "qualified": qualified,
                    "evaluation": evaluation,
                    "certification": (
                        _serializable_trefic(
                            control_certifications[trefic_index])
                        if trefic_index >= 0
                        else None
                    ),
                }

            records.append({
                "dataset": dataset,
                "replicate": replicate,
                "evaluation_fold": evaluation_fold,
                "selection_fold": selection_fold,
                "certification_fold": certification_fold,
                "rho_envelope": {
                    "lower": 0.0,
                    **rho,
                },
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
                "trefic_certifications": [
                    _serializable_trefic(result)
                    for result in trefic_certifications
                ],
                "normal_evaluations": evaluations,
                "control_evaluations": control_records,
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
        raise ValueError("unknown TREFIC regime")
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
        result, records = run_dataset_trefic(
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
    records_path = output_dir / "trefic_trials.jsonl"
    records_path.write_text("".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in all_records
    ))
    manifest = {
        "experiment": "TREFIC_Phase0_ratio_envelope",
        "method_version": 1,
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
        "reference_candidate_delta": spec["delta"] / 3.0,
        "trefic_endpoint_delta": spec["delta"] / 6.0,
        "rho_lower": 0.0,
        "rho_upper_method": "one_sided_wilson_selection_plus_certification",
        "rho_delta": spec["delta"],
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
    manifest_path = output_dir / "trefic_phase0_manifest.json"
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "regime": regime,
        "datasets": {
            dataset: {
                "oracle_rate": row["oracle_opportunity_rate"],
                "trefic": row["methods"]["trefic"],
            }
            for dataset, row in dataset_results.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
