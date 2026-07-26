#!/usr/bin/env python3
"""Controlled graph-time dependence validation for GTPRC."""

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

from fraudGT.evidence.gtprc import (  # noqa: E402
    gather_policy,
    grouped_empirical_bernstein_upper,
    grouped_hoeffding_upper,
    row_wilson_upper,
    select_max_coverage_policy,
)


REGIMES = {
    "iid": {
        "graph_group_size": 1,
        "entity_group_size": 1,
        "time_group_size": 1,
        "shared_strength": 0.0,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -2.15,
        "break_logit": -1.55,
        "duplicate_strength": 0.0,
    },
    "entity_cluster": {
        "graph_group_size": 32,
        "entity_group_size": 32,
        "time_group_size": 1,
        "shared_strength": 1.1,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -2.15,
        "break_logit": -1.55,
        "duplicate_strength": 0.0,
    },
    "temporal_autocorrelation": {
        "graph_group_size": 32,
        "entity_group_size": 1,
        "time_group_size": 32,
        "shared_strength": 1.1,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -2.15,
        "break_logit": -1.55,
        "duplicate_strength": 0.0,
    },
    "entity_temporal": {
        "graph_group_size": 64,
        "entity_group_size": 8,
        "time_group_size": 8,
        "shared_strength": 1.3,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -2.15,
        "break_logit": -1.55,
        "duplicate_strength": 0.0,
    },
    "prevalence_drift": {
        "graph_group_size": 32,
        "entity_group_size": 4,
        "time_group_size": 16,
        "shared_strength": 0.9,
        "time_drift": 1.1,
        "alignment_drift": 0.0,
        "correct_logit": -2.35,
        "break_logit": -1.65,
        "duplicate_strength": 0.0,
    },
    "alignment_drift": {
        "graph_group_size": 32,
        "entity_group_size": 4,
        "time_group_size": 16,
        "shared_strength": 0.9,
        "time_drift": 0.0,
        "alignment_drift": 1.3,
        "correct_logit": -2.15,
        "break_logit": -1.55,
        "duplicate_strength": 0.0,
    },
    "rare_duplicate": {
        "graph_group_size": 64,
        "entity_group_size": 16,
        "time_group_size": 16,
        "shared_strength": 1.35,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -3.05,
        "break_logit": -2.05,
        "duplicate_strength": 0.9,
    },
}

V2_REGIMES = {
    "iid": {
        "graph_group_size": 1,
        "entity_group_size": 1,
        "time_group_size": 1,
        "shared_strength": 0.0,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -1.75,
        "break_logit": -0.15,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "entity_cluster": {
        "graph_group_size": 128,
        "entity_group_size": 128,
        "time_group_size": 1,
        "shared_strength": 1.55,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -1.75,
        "break_logit": -0.15,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "temporal_autocorrelation": {
        "graph_group_size": 128,
        "entity_group_size": 1,
        "time_group_size": 128,
        "shared_strength": 1.55,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -1.75,
        "break_logit": -0.15,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "entity_temporal": {
        "graph_group_size": 256,
        "entity_group_size": 16,
        "time_group_size": 16,
        "shared_strength": 1.75,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -1.75,
        "break_logit": -0.15,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "prevalence_drift": {
        "graph_group_size": 128,
        "entity_group_size": 8,
        "time_group_size": 32,
        "shared_strength": 1.35,
        "time_drift": 1.15,
        "alignment_drift": 0.0,
        "correct_logit": -1.85,
        "break_logit": -0.20,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "alignment_drift": {
        "graph_group_size": 128,
        "entity_group_size": 8,
        "time_group_size": 32,
        "shared_strength": 1.35,
        "time_drift": 0.0,
        "alignment_drift": 1.10,
        "correct_logit": -1.75,
        "break_logit": -0.15,
        "duplicate_strength": 0.0,
        "score_correct_weight": 1.55,
        "score_break_weight": 1.25,
        "score_noise": 1.35,
    },
    "rare_duplicate": {
        "graph_group_size": 256,
        "entity_group_size": 32,
        "time_group_size": 32,
        "shared_strength": 1.85,
        "time_drift": 0.0,
        "alignment_drift": 0.0,
        "correct_logit": -2.65,
        "break_logit": -0.45,
        "duplicate_strength": 0.9,
        "score_correct_weight": 1.70,
        "score_break_weight": 1.30,
        "score_noise": 1.25,
    },
}


METHOD_GROUP_KEYS = {
    "row_iid": None,
    "time_block": "time_group_size",
    "entity_block": "entity_group_size",
    "gtprc_graph_time": "graph_group_size",
}


def repository_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def verify_repository(expected_commit):
    commit = repository_commit()
    if expected_commit and not commit.startswith(expected_commit):
        raise RuntimeError(
            "repository commit {} does not match {}".format(
                commit, expected_commit))
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("repository must be clean before execution")
    return commit


def expand_group_effect(
        batch_size, row_count, group_size, device, generator):
    group_count = row_count // int(group_size)
    effects = torch.randn(
        batch_size,
        group_count,
        device=device,
        generator=generator,
    )
    return effects.repeat_interleave(int(group_size), dim=1)


def simulate_potential_outcomes(
        config, batch_size, row_count, device, generator):
    group_size = int(config["graph_group_size"])
    if row_count % group_size != 0:
        raise ValueError("row_count must be divisible by graph group size")
    shared = expand_group_effect(
        batch_size, row_count, group_size, device, generator)
    position = torch.linspace(
        -1.0, 1.0, row_count, device=device)[None, :]
    individual = torch.randn(
        batch_size, row_count, device=device, generator=generator)
    duplicate = float(config["duplicate_strength"]) * shared

    break_logit = (
        float(config["break_logit"])
        + float(config["shared_strength"]) * shared
        + float(config["time_drift"]) * position
        + 0.35 * individual
        + 0.25 * duplicate
    )
    correct_logit = (
        float(config["correct_logit"])
        - 0.55 * float(config["shared_strength"]) * shared
        - 0.50 * float(config["time_drift"]) * position
        - 0.20 * individual
        - 0.15 * duplicate
    )
    p_break = torch.sigmoid(break_logit).clamp(max=0.65)
    p_correct = torch.sigmoid(correct_logit).clamp(max=0.45)
    total = p_break + p_correct
    scale = torch.maximum(total / 0.92, torch.ones_like(total))
    p_break = p_break / scale
    p_correct = p_correct / scale

    draw = torch.rand(
        batch_size, row_count, device=device, generator=generator)
    broken = draw < p_break
    corrected = (draw >= p_break) & (draw < p_break + p_correct)

    score_noise = torch.randn(
        batch_size, row_count, device=device, generator=generator)
    alignment = (
        1.0
        - float(config["alignment_drift"])
        * (position + 1.0) / 2.0
    )
    aligned_score = (
        alignment * (
            float(config.get("score_correct_weight", 2.7))
            * corrected.to(torch.float32)
            - float(config.get("score_break_weight", 2.2))
            * broken.to(torch.float32)
        )
        - 0.35 * shared
        + float(config.get("score_noise", 0.85)) * score_noise
    )
    permutation = torch.randperm(
        row_count, device=device, generator=generator)
    shuffled_score = aligned_score[:, permutation]
    harmful_score = -aligned_score
    return {
        "broken": broken,
        "corrected": corrected,
        "aligned": aligned_score,
        "shuffled": shuffled_score,
        "harmful": harmful_score,
    }


def candidate_thresholds(scores, quantiles):
    return torch.quantile(
        scores, quantiles.to(scores.device), dim=1).transpose(0, 1)


def policy_tensors(scores, broken, corrected, thresholds):
    selected = scores[:, None, :] >= thresholds[:, :, None]
    selected_count = selected.sum(dim=-1).to(torch.float64)
    coverage = selected_count / scores.shape[1]
    break_count = (
        selected & broken[:, None, :]
    ).sum(dim=-1).to(torch.float64)
    correct_count = (
        selected & corrected[:, None, :]
    ).sum(dim=-1).to(torch.float64)
    net = (correct_count - break_count) / scores.shape[1]
    return selected, coverage, net


def method_upper(
        method, selected, broken, config, family_size, delta,
        stress_version):
    group_key = METHOD_GROUP_KEYS[method]
    if group_key is None:
        return row_wilson_upper(
            selected, broken, family_size, delta)
    bound = (
        grouped_hoeffding_upper
        if int(stress_version) == 2
        else grouped_empirical_bernstein_upper
    )
    return bound(
        selected,
        broken,
        int(config[group_key]),
        family_size,
        delta,
    )


def evaluate_indices(scores, broken, corrected, thresholds, indices):
    chosen_threshold = gather_policy(thresholds, indices, default=math.inf)
    selected = scores >= chosen_threshold[:, None]
    selected_count = selected.sum(dim=-1).to(torch.float64)
    break_count = (
        selected & broken
    ).sum(dim=-1).to(torch.float64)
    correct_count = (
        selected & corrected
    ).sum(dim=-1).to(torch.float64)
    risk = break_count / selected_count.clamp_min(1.0)
    coverage = selected_count / scores.shape[1]
    net_count = correct_count - break_count
    return {
        "risk": torch.where(
            indices >= 0, risk, torch.zeros_like(risk)),
        "coverage": torch.where(
            indices >= 0, coverage, torch.zeros_like(coverage)),
        "net_count": torch.where(
            indices >= 0, net_count, torch.zeros_like(net_count)),
        "qualified": indices >= 0,
    }


def oracle_indices(scores, broken, corrected, thresholds, alpha, min_coverage):
    selected, coverage, net = policy_tensors(
        scores, broken, corrected, thresholds)
    selected_count = selected.sum(dim=-1).to(torch.float64)
    break_count = (
        selected & broken[:, None, :]
    ).sum(dim=-1).to(torch.float64)
    risk = break_count / selected_count.clamp_min(1.0)
    return select_max_coverage_policy(
        coverage, net, risk, alpha, min_coverage)


def empty_accumulator(methods):
    return {
        method: {
            "replicates": 0,
            "qualified": 0,
            "violations": 0,
            "coverage": [],
            "oracle_fraction": [],
            "net_count": [],
        }
        for method in methods
    }


def run_simulation(
        regime,
        seed,
        replicates,
        batch_size,
        calibration_rows,
        test_rows,
        policy_count,
        alpha,
        delta,
        min_coverage,
        device,
        stress_version=1):
    regime_table = V2_REGIMES if int(stress_version) == 2 else REGIMES
    config = regime_table[regime]
    methods = list(METHOD_GROUP_KEYS)
    accumulator = empty_accumulator(methods)
    control_qualified = {"shuffled": 0, "harmful": 0}
    completed = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    quantile_start = 0.0 if int(stress_version) == 2 else 0.50
    quantiles = torch.linspace(
        quantile_start, 0.995, policy_count, device=device)

    while completed < replicates:
        current = min(batch_size, replicates - completed)
        calibration = simulate_potential_outcomes(
            config, current, calibration_rows, device, generator)
        test = simulate_potential_outcomes(
            config, current, test_rows, device, generator)
        thresholds = candidate_thresholds(
            calibration["aligned"], quantiles)
        selected, coverage, net = policy_tensors(
            calibration["aligned"],
            calibration["broken"],
            calibration["corrected"],
            thresholds,
        )
        oracle = oracle_indices(
            test["aligned"],
            test["broken"],
            test["corrected"],
            thresholds,
            alpha,
            min_coverage,
        )
        oracle_eval = evaluate_indices(
            test["aligned"],
            test["broken"],
            test["corrected"],
            thresholds,
            oracle,
        )

        for method in methods:
            upper = method_upper(
                method,
                selected,
                calibration["broken"],
                config,
                policy_count,
                delta,
                stress_version,
            )
            indices = select_max_coverage_policy(
                coverage, net, upper, alpha, min_coverage)
            evaluated = evaluate_indices(
                test["aligned"],
                test["broken"],
                test["corrected"],
                thresholds,
                indices,
            )
            qualified = evaluated["qualified"]
            violation = qualified & (evaluated["risk"] > alpha)
            oracle_coverage = oracle_eval["coverage"]
            fraction = torch.where(
                oracle_coverage > 0,
                evaluated["coverage"] / oracle_coverage,
                torch.ones_like(oracle_coverage),
            ).clamp(max=1.0)
            row = accumulator[method]
            row["replicates"] += current
            row["qualified"] += int(qualified.sum().item())
            row["violations"] += int(violation.sum().item())
            row["coverage"].extend(
                evaluated["coverage"].detach().cpu().tolist())
            row["oracle_fraction"].extend(
                fraction.detach().cpu().tolist())
            row["net_count"].extend(
                evaluated["net_count"].detach().cpu().tolist())

        for control in ("shuffled", "harmful"):
            control_thresholds = candidate_thresholds(
                calibration[control], quantiles)
            c_selected, c_coverage, c_net = policy_tensors(
                calibration[control],
                calibration["broken"],
                calibration["corrected"],
                control_thresholds,
            )
            c_upper = method_upper(
                "gtprc_graph_time",
                c_selected,
                calibration["broken"],
                config,
                policy_count,
                delta,
                stress_version,
            )
            c_indices = select_max_coverage_policy(
                c_coverage, c_net, c_upper, alpha, min_coverage)
            control_qualified[control] += int(
                (c_indices >= 0).sum().item())
        completed += current

    summaries = {}
    for method, row in accumulator.items():
        coverage = torch.tensor(row.pop("coverage"), dtype=torch.float64)
        oracle_fraction = torch.tensor(
            row.pop("oracle_fraction"), dtype=torch.float64)
        net_count = torch.tensor(row.pop("net_count"), dtype=torch.float64)
        summaries[method] = {
            **row,
            "qualification_rate": row["qualified"] / row["replicates"],
            "violation_rate": row["violations"] / row["replicates"],
            "coverage_mean": float(coverage.mean().item()),
            "coverage_median": float(coverage.median().item()),
            "oracle_fraction_mean": float(
                oracle_fraction.mean().item()),
            "oracle_fraction_median": float(
                oracle_fraction.median().item()),
            "net_count_mean": float(net_count.mean().item()),
        }
    controls = {
        key: {
            "qualified": value,
            "false_qualification_rate": value / replicates,
        }
        for key, value in control_qualified.items()
    }
    return summaries, controls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=sorted(REGIMES), required=True)
    parser.add_argument("--stress-version", type=int, choices=(1, 2),
                        default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--replicates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-rows", type=int, default=8192)
    parser.add_argument("--test-rows", type=int, default=16384)
    parser.add_argument("--policy-count", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    commit = verify_repository(args.expected_commit)
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.time()
    summaries, controls = run_simulation(
        regime=args.regime,
        seed=args.seed,
        replicates=args.replicates,
        batch_size=args.batch_size,
        calibration_rows=args.calibration_rows,
        test_rows=args.test_rows,
        policy_count=args.policy_count,
        alpha=args.alpha,
        delta=args.delta,
        min_coverage=args.min_coverage,
        device=device,
        stress_version=args.stress_version,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment": "gtprc_phase0a_controlled_dependence",
        "regime": args.regime,
        "stress_version": args.stress_version,
        "seed": args.seed,
        "git_commit": commit,
        "experiment_protocol": "controlled_graph_time_simulation",
        "downstream_sampling_protocol": "dynamic_random",
        "replicates": args.replicates,
        "batch_size": args.batch_size,
        "calibration_rows": args.calibration_rows,
        "test_rows": args.test_rows,
        "policy_count": args.policy_count,
        "alpha": args.alpha,
        "delta": args.delta,
        "min_coverage": args.min_coverage,
        "device": str(device),
        "regime_config": (
            V2_REGIMES if args.stress_version == 2 else REGIMES
        )[args.regime],
        "methods": summaries,
        "controls": controls,
        "runtime_seconds": time.time() - started,
    }
    path = args.output_dir / "gtprc_phase0a_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(path),
        "regime": args.regime,
        "gtprc": summaries["gtprc_graph_time"],
        "controls": controls,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
