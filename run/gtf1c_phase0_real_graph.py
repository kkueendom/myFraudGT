#!/usr/bin/env python3
"""Real-graph semi-synthetic stress benchmark for GTF1C."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.gtf1c import (  # noqa: E402
    graph_time_groups,
    grouped_break_upper,
    paired_f1_lcb,
    paired_policy_statistics,
    time_block_groups,
)
from fraudGT.evidence.gtprc import row_wilson_upper  # noqa: E402


POSITIVE_REGIMES = {
    "iid_positive",
    "entity_positive",
    "temporal_positive",
    "graph_time_positive",
}
NEGATIVE_REGIMES = {
    "cpse_remove_fragility",
    "alignment_drift",
    "rare_duplicate",
}

REGIMES = {
    "iid_positive": {
        "signals": {"selection": 4.0, "certification": 4.0,
                    "evaluation": 4.0},
        "noise": 1.0,
        "graph_effect": 0.0,
        "time_effect": 0.0,
        "degree_effect": 0.0,
        "allow_add": True,
        "allow_remove": True,
    },
    "entity_positive": {
        "signals": {"selection": 4.0, "certification": 4.0,
                    "evaluation": 4.0},
        "noise": 1.0,
        "graph_effect": 1.4,
        "time_effect": 0.0,
        "degree_effect": 0.0,
        "allow_add": True,
        "allow_remove": True,
    },
    "temporal_positive": {
        "signals": {"selection": 4.0, "certification": 4.0,
                    "evaluation": 4.0},
        "noise": 1.0,
        "graph_effect": 0.0,
        "time_effect": 1.4,
        "degree_effect": 0.0,
        "allow_add": True,
        "allow_remove": True,
    },
    "graph_time_positive": {
        "signals": {"selection": 4.0, "certification": 4.0,
                    "evaluation": 4.0},
        "noise": 1.0,
        "graph_effect": 1.0,
        "time_effect": 1.0,
        "degree_effect": 0.0,
        "allow_add": True,
        "allow_remove": True,
    },
    "cpse_remove_fragility": {
        "signals": {"selection": 2.0, "certification": 0.15,
                    "evaluation": 0.0},
        "noise": 0.8,
        "graph_effect": 0.4,
        "time_effect": 0.0,
        "degree_effect": 0.0,
        "allow_add": False,
        "allow_remove": True,
    },
    "alignment_drift": {
        "signals": {"selection": 4.0, "certification": 0.2,
                    "evaluation": -0.5},
        "noise": 1.0,
        "graph_effect": 0.7,
        "time_effect": 0.7,
        "degree_effect": 0.0,
        "allow_add": True,
        "allow_remove": True,
    },
    "rare_duplicate": {
        "signals": {"selection": 2.0, "certification": 0.1,
                    "evaluation": 0.0},
        "noise": 0.8,
        "graph_effect": 1.6,
        "time_effect": 0.0,
        "degree_effect": 1.2,
        "allow_add": False,
        "allow_remove": True,
    },
}

METHODS = (
    "row_net",
    "gtprc_row_harm",
    "iid_paired_f1",
    "time_block_paired_f1",
    "gtf1c_graph_time",
)
CONDITIONS = ("normal", "shuffled", "harmful")


def repository_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def verify_repository(expected_commit):
    commit = repository_commit()
    if expected_commit and not commit.startswith(expected_commit):
        raise RuntimeError(
            f"repository commit {commit} does not match {expected_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("repository must be clean before execution")
    return commit


def _validate_source_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest["sampling_protocol"] != "dynamic_random":
        raise ValueError("source OOF protocol is not dynamic_random")
    if manifest["validation_loader_iterations"] != 0:
        raise ValueError("source OOF used validation loader")
    if manifest["test_loader_iterations"] != 0:
        raise ValueError("source OOF used test loader")
    if manifest["fraud_labels_used_for_cpse_training"]:
        raise ValueError("source CPSE training used fraud labels")
    if not manifest["held_out_edges_excluded_from_cpse_loss"]:
        raise ValueError("held-out edge entered CPSE loss")
    return manifest


def _degree_signal(source, destination):
    entities = torch.cat((source, destination))
    _, inverse, counts = torch.unique(
        entities, return_inverse=True, return_counts=True)
    count = source.numel()
    degree = (
        counts[inverse[:count]] + counts[inverse[count:]]
    ).to(torch.float32).log1p()
    return (degree - degree.mean()) / degree.std(
        unbiased=False).clamp_min(1e-6)


def load_dataset(root, dataset, block_count):
    manifests = sorted(root.glob(
        f"{dataset}_fold*/cpse_phase0b_manifest.json"))
    if len(manifests) != 3:
        raise ValueError(f"{dataset} does not have three source manifests")
    payloads = []
    source_manifests = []
    seen_folds = set()
    for manifest_path in manifests:
        manifest = _validate_source_manifest(manifest_path)
        payload = torch.load(
            manifest["cpse_oof_features"], map_location="cpu")
        fold = int(payload["fold"])
        if fold in seen_folds:
            raise ValueError("duplicate source OOF fold")
        seen_folds.add(fold)
        if payload["sampling_protocol"] != "dynamic_random":
            raise ValueError("payload protocol is not dynamic_random")
        payloads.append(payload)
        source_manifests.append(str(manifest_path))
    if seen_folds != {0, 1, 2}:
        raise ValueError("source OOF fold set is incomplete")

    edge_sets = [set(row["edge_ids"].tolist()) for row in payloads]
    if any(
        edge_sets[left] & edge_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("edge ID appears in more than one OOF fold")

    views = {}
    for row in payloads:
        fold = int(row["fold"])
        labels = row["labels"].bool()
        base = row["a2_scores"] >= float(row["a2_threshold"])
        timestamps = row["timestamps"]
        time_groups = time_block_groups(timestamps, block_count)
        component_groups = graph_time_groups(
            row["source_ids"], row["destination_ids"], timestamps,
            block_count,
        )
        views[fold] = {
            "labels": labels,
            "base": base,
            "source": row["source_ids"],
            "destination": row["destination_ids"],
            "timestamps": timestamps,
            "edge_ids": row["edge_ids"],
            "time_groups": time_groups,
            "graph_groups": component_groups,
            "degree_signal": _degree_signal(
                row["source_ids"], row["destination_ids"]),
        }
    return views, source_manifests


def move_view(view, device):
    return {
        key: value.to(device)
        for key, value in view.items()
    }


def _mapped_effect(groups, generator):
    group_count = int(groups.max()) + 1 if groups.numel() else 0
    if group_count == 0:
        return torch.zeros_like(groups, dtype=torch.float32)
    values = torch.randn(
        group_count, device=groups.device, generator=generator)
    return values[groups]


def _shuffle_candidates(scores, candidate, generator):
    shuffled = scores.clone()
    indices = torch.where(candidate)[0]
    if indices.numel():
        order = torch.randperm(
            indices.numel(), device=scores.device, generator=generator)
        shuffled[indices] = scores[indices[order]]
    return shuffled


def generate_scores(view, config, role, generator):
    labels = view["labels"]
    base = view["base"]
    signal = float(config["signals"][role])
    graph_effect = _mapped_effect(view["graph_groups"], generator)
    time_effect = _mapped_effect(view["time_groups"], generator)
    shared = (
        float(config["graph_effect"]) * graph_effect
        + float(config["time_effect"]) * time_effect
        + float(config["degree_effect"]) * view["degree_signal"]
    )
    scores = {}
    for direction in ("add", "remove"):
        utility = labels if direction == "add" else ~labels
        normal = (
            signal * (2.0 * utility.float() - 1.0)
            + shared
            + float(config["noise"]) * torch.randn(
                labels.numel(), device=labels.device,
                generator=generator)
        )
        candidate = ~base if direction == "add" else base
        scores[direction] = {
            "normal": normal,
            "shuffled": _shuffle_candidates(
                normal, candidate, generator),
            "harmful": -normal,
        }
    return scores


def _direction_thresholds(scores, candidates, quantiles, allowed):
    if not allowed or not candidates.any():
        return torch.full_like(quantiles, math.inf)
    return torch.quantile(scores[candidates], quantiles)


def candidate_policy_thresholds(
        view, scores, quantiles, config):
    add = _direction_thresholds(
        scores["add"]["normal"], ~view["base"], quantiles,
        config["allow_add"])
    remove = _direction_thresholds(
        scores["remove"]["normal"], view["base"], quantiles,
        config["allow_remove"])
    infinity = torch.full_like(quantiles, math.inf)
    return (
        torch.cat((add, infinity, add)),
        torch.cat((infinity, remove, remove)),
    )


def apply_policy(view, scores, condition, add_threshold, remove_threshold):
    base = view["base"]
    routed = base.clone()
    add = (
        ~base
        & (scores["add"][condition] >= float(add_threshold))
    )
    remove = (
        base
        & (scores["remove"][condition] >= float(remove_threshold))
    )
    routed[add] = True
    routed[remove] = False
    return routed


def policy_family_statistics(
        view, scores, condition, add_thresholds, remove_thresholds):
    labels = view["labels"]
    base = view["base"]
    add = (
        ~base[None, :]
        & (
            scores["add"][condition][None, :]
            >= add_thresholds[:, None]
        )
    )
    remove = (
        base[None, :]
        & (
            scores["remove"][condition][None, :]
            >= remove_thresholds[:, None]
        )
    )
    routed = base[None, :].expand(
        add_thresholds.numel(), -1).clone()
    routed[add] = True
    routed[remove] = False
    tp = (routed & labels[None, :]).sum(dim=1).to(torch.float64)
    fp = (routed & ~labels[None, :]).sum(dim=1).to(torch.float64)
    fn = (~routed & labels[None, :]).sum(dim=1).to(torch.float64)
    denominator = (2.0 * tp + fp + fn).clamp_min(1.0)
    routed_f1 = 2.0 * tp / denominator
    base_stats = paired_policy_statistics(labels, base, base)
    delta = routed_f1 - base_stats["base_f1"]
    changed = (add | remove).sum(dim=1)
    return delta, changed


def select_policy(
        view, scores, add_thresholds, remove_thresholds,
        min_changes):
    delta, changed = policy_family_statistics(
        view, scores, "normal", add_thresholds, remove_thresholds)
    valid = changed >= int(min_changes)
    ranked = torch.where(
        valid, delta, torch.full_like(delta, -math.inf))
    if not valid.any():
        return -1
    return int(ranked.argmax())


def oracle_policy(
        view, scores, add_thresholds, remove_thresholds,
        min_changes, practical_delta):
    delta, changed = policy_family_statistics(
        view, scores, "normal", add_thresholds, remove_thresholds)
    valid = (
        (changed >= int(min_changes))
        & (delta >= float(practical_delta))
    )
    ranked = torch.where(
        valid, delta, torch.full_like(delta, -math.inf))
    if not valid.any():
        return -1, 0, 0.0
    index = int(ranked.argmax())
    return index, int(changed[index]), float(delta[index])


def _row_break_upper(active, broken, delta):
    selected = active[None, None, :]
    broken_rows = broken[None, :]
    return float(row_wilson_upper(
        selected, broken_rows, 1, delta)[0, 0])


def certification_results(
        view, routed, min_changes, delta, practical_delta):
    labels = view["labels"]
    base = view["base"]
    stats = paired_policy_statistics(labels, base, routed)
    active = base != routed
    broken = active & (base == labels) & (routed != labels)
    row_upper = _row_break_upper(active, broken, delta)
    grouped_upper = grouped_break_upper(
        active, broken, view["graph_groups"], delta=delta)
    iid = paired_f1_lcb(labels, base, routed, delta=delta)
    time = paired_f1_lcb(
        labels, base, routed, view["time_groups"], delta=delta)
    graph = paired_f1_lcb(
        labels, base, routed, view["graph_groups"], delta=delta)
    common = stats["changed"] >= int(min_changes)
    f1_common = (
        common
        and stats["paired_f1_delta"] >= float(practical_delta)
    )
    qualifications = {
        "row_net": (
            common
            and stats["corrected"] > stats["broken"]
            and stats["corrected"] >= 1.5 * max(stats["broken"], 1)
            and row_upper <= 0.40
        ),
        "gtprc_row_harm": (
            common
            and stats["corrected"] > stats["broken"]
            and stats["corrected"] >= 1.5 * max(stats["broken"], 1)
            and grouped_upper <= 0.40
        ),
        "iid_paired_f1": (
            f1_common and iid["lower_bound"] > 0.0),
        "time_block_paired_f1": (
            f1_common and time["lower_bound"] > 0.0),
        "gtf1c_graph_time": (
            f1_common
            and time["lower_bound"] > 0.0
            and graph["lower_bound"] > 0.0
        ),
    }
    return {
        "statistics": stats,
        "row_break_upper": row_upper,
        "grouped_break_upper": grouped_upper,
        "iid_f1": iid,
        "time_f1": time,
        "graph_f1": graph,
        "qualifications": qualifications,
    }


def empty_accumulator():
    return {
        "trials": 0,
        "qualified": 0,
        "violations": 0,
        "practical_failures": 0,
        "evaluation_deltas": [],
        "evaluation_changes": [],
        "oracle_fractions": [],
    }


def summarize_accumulator(row):
    deltas = torch.tensor(
        row["evaluation_deltas"], dtype=torch.float64)
    changes = torch.tensor(
        row["evaluation_changes"], dtype=torch.float64)
    fractions = torch.tensor(
        row["oracle_fractions"], dtype=torch.float64)
    qualified = row["qualified"]
    return {
        "trials": row["trials"],
        "qualified": qualified,
        "qualification_rate": qualified / max(row["trials"], 1),
        "violations": row["violations"],
        "false_qualification_rate": (
            row["violations"] / max(row["trials"], 1)),
        "conditional_violation_rate": (
            row["violations"] / max(qualified, 1)),
        "practical_failures": row["practical_failures"],
        "evaluation_delta_mean": (
            float(deltas.mean()) if deltas.numel() else 0.0),
        "evaluation_delta_median": (
            float(deltas.median()) if deltas.numel() else 0.0),
        "evaluation_changes_mean": (
            float(changes.mean()) if changes.numel() else 0.0),
        "oracle_fraction_mean": (
            float(fractions.mean()) if fractions.numel() else 0.0),
        "oracle_fraction_median": (
            float(fractions.median()) if fractions.numel() else 0.0),
    }


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def run_dataset(
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
    config = REGIMES[regime]
    methods = {method: empty_accumulator() for method in METHODS}
    controls = {
        condition: empty_accumulator()
        for condition in ("shuffled", "harmful")
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    quantiles = torch.linspace(
        0.50, 0.9995, int(policy_count), device=device)
    records = []
    gpu_views = {
        fold: move_view(view, device)
        for fold, view in views.items()
    }
    oracle_opportunities = 0

    for replicate in range(int(replicates)):
        for evaluation_fold in range(3):
            selection_fold = (evaluation_fold + 1) % 3
            certification_fold = (evaluation_fold + 2) % 3
            selection_view = gpu_views[selection_fold]
            certification_view = gpu_views[certification_fold]
            evaluation_view = gpu_views[evaluation_fold]
            selection_scores = generate_scores(
                selection_view, config, "selection", generator)
            certification_scores = generate_scores(
                certification_view, config, "certification", generator)
            evaluation_scores = generate_scores(
                evaluation_view, config, "evaluation", generator)
            add_thresholds, remove_thresholds = (
                candidate_policy_thresholds(
                    selection_view, selection_scores, quantiles, config)
            )
            selected_index = select_policy(
                selection_view,
                selection_scores,
                add_thresholds,
                remove_thresholds,
                min_changes,
            )
            oracle_index, oracle_changed, oracle_delta = oracle_policy(
                evaluation_view,
                evaluation_scores,
                add_thresholds,
                remove_thresholds,
                min_changes,
                practical_delta,
            )
            oracle_opportunities += int(oracle_index >= 0)
            if selected_index < 0:
                add_threshold = math.inf
                remove_threshold = math.inf
            else:
                add_threshold = float(add_thresholds[selected_index])
                remove_threshold = float(remove_thresholds[selected_index])

            condition_results = {}
            for condition in CONDITIONS:
                cert_routed = apply_policy(
                    certification_view,
                    certification_scores,
                    condition,
                    add_threshold,
                    remove_threshold,
                )
                eval_routed = apply_policy(
                    evaluation_view,
                    evaluation_scores,
                    condition,
                    add_threshold,
                    remove_threshold,
                )
                certification = certification_results(
                    certification_view,
                    cert_routed,
                    min_changes,
                    delta,
                    practical_delta,
                )
                evaluation = paired_policy_statistics(
                    evaluation_view["labels"],
                    evaluation_view["base"],
                    eval_routed,
                )
                condition_results[condition] = {
                    "certification": certification,
                    "evaluation": evaluation,
                }

            normal = condition_results["normal"]
            for method in METHODS:
                row = methods[method]
                qualified = bool(
                    normal["certification"]["qualifications"][method])
                evaluation = normal["evaluation"]
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

            for condition in ("shuffled", "harmful"):
                row = controls[condition]
                result = condition_results[condition]
                qualified = bool(
                    result["certification"]["qualifications"][
                        "gtf1c_graph_time"
                    ]
                )
                evaluation = result["evaluation"]
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

            cert = normal["certification"]
            records.append({
                "dataset": dataset,
                "replicate": replicate,
                "evaluation_fold": evaluation_fold,
                "selection_fold": selection_fold,
                "certification_fold": certification_fold,
                "selected_policy_index": selected_index,
                "add_threshold": finite_or_none(add_threshold),
                "remove_threshold": finite_or_none(remove_threshold),
                "oracle_policy_index": oracle_index,
                "oracle_changed": oracle_changed,
                "oracle_delta": oracle_delta,
                "certification_statistics": cert["statistics"],
                "row_break_upper": cert["row_break_upper"],
                "grouped_break_upper": cert["grouped_break_upper"],
                "iid_f1_lcb": finite_or_none(
                    cert["iid_f1"]["lower_bound"]),
                "time_f1_lcb": finite_or_none(
                    cert["time_f1"]["lower_bound"]),
                "graph_f1_lcb": finite_or_none(
                    cert["graph_f1"]["lower_bound"]),
                "qualifications": cert["qualifications"],
                "normal_evaluation": normal["evaluation"],
                "shuffled_evaluation": (
                    condition_results["shuffled"]["evaluation"]),
                "harmful_evaluation": (
                    condition_results["harmful"]["evaluation"]),
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
    if regime not in REGIMES:
        raise ValueError("unknown GTF1C regime")
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
        result, records = run_dataset(
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
    records_path = output_dir / "gtf1c_trials.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in all_records
        )
    )
    manifest = {
        "experiment": "GTF1C_Phase0_real_graph_stress",
        "regime": regime,
        "regime_type": (
            "positive" if regime in POSITIVE_REGIMES else "negative"),
        "regime_config": REGIMES[regime],
        "seed": task["seed"],
        "git_commit": commit,
        "sampling_protocol": "dynamic_random",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "replicates": replicates,
        "policy_count": spec["policy_count"],
        "policy_family_size": 3 * int(spec["policy_count"]),
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
