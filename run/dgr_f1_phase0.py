#!/usr/bin/env python3
"""Controlled development screen for DGR-F1."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.dgr_f1 import (  # noqa: E402
    DECISION_HARM,
    DECISION_IMPROVEMENT,
    DECISION_INSUFFICIENT,
    dgr_f1_trajectory,
    iid_paired_f1_interval,
    mean_only_trajectory,
    one_stream_decision,
    replication_only_trajectory,
)
from run.gtf1c_phase0_real_graph import verify_repository  # noqa: E402


SCENARIO_TARGETS = {
    "stable_iid_improvement": DECISION_IMPROVEMENT,
    "stable_graph_improvement": DECISION_IMPROVEMENT,
    "stable_temporal_improvement": DECISION_IMPROVEMENT,
    "positive_mean_low_replication": DECISION_INSUFFICIENT,
    "base_ratio_sign_reversal": DECISION_INSUFFICIENT,
    "null_intervention": DECISION_INSUFFICIENT,
    "stable_harm": DECISION_HARM,
}
METHODS = (
    "one_stream",
    "row_iid",
    "mean_only",
    "replication_only",
    "dgr_f1",
)


def _poisson_counts(mean, factor, generator):
    rates = torch.full_like(
        factor, float(mean), dtype=torch.float64) * factor
    return torch.poisson(rates.clamp_min(0.0), generator=generator).long()


def _lognormal_factor(shape, sigma, generator, device):
    noise = torch.randn(
        shape, dtype=torch.float64, device=device, generator=generator)
    return torch.exp(float(sigma) * noise - 0.5 * float(sigma) ** 2)


def _binomial(counts, probability, generator):
    if not torch.is_tensor(probability):
        probability = torch.full_like(
            counts, float(probability), dtype=torch.float64)
    else:
        probability = probability.to(
            dtype=torch.float64, device=counts.device)
    return torch.binomial(
        counts.to(torch.float64),
        probability.clamp(0.0, 1.0),
        generator=generator,
    ).long()


def _scaled_probability(base, latent, strength):
    return (
        float(base) * torch.exp(float(strength) * latent)
    ).clamp(0.0, min(1.0, float(base) * 8.0))


def simulate_streams(
        template,
        scenario,
        shape,
        generator,
        device,
):
    """Generate feasible stream-level confusion and intervention counts."""
    parameters = scenario["parameters"]
    count_sigma = float(parameters["count_log_sd"])
    common = _lognormal_factor(
        shape, count_sigma, generator, device)
    tp = _poisson_counts(
        template["tp"], common, generator).clamp_min(0)
    fp = _poisson_counts(
        template["fp"],
        _lognormal_factor(shape, count_sigma, generator, device),
        generator,
    )
    fn = _poisson_counts(
        template["fn"],
        _lognormal_factor(shape, count_sigma, generator, device),
        generator,
    )
    tn = torch.full(
        shape, int(template["tn"]), dtype=torch.long, device=device)

    mode = scenario["mode"]
    latent = torch.randn(
        shape, dtype=torch.float64, device=device, generator=generator)
    if mode == "ratio_reversal":
        high = torch.rand(
            shape, dtype=torch.float64, device=device,
            generator=generator) < float(parameters["high_ratio_probability"])
        multipliers = torch.where(
            high,
            torch.full_like(latent, float(parameters["tp_high_multiplier"])),
            torch.full_like(latent, float(parameters["tp_low_multiplier"])),
        )
        tp = _poisson_counts(
            template["tp"], multipliers, generator)
        fp = fp.clamp_min(1)
        remove_corrected = torch.round(
            float(parameters["remove_corrected_rate"]) * fp).long()
        remove_broken = torch.minimum(
            torch.full_like(tp, int(parameters["remove_broken_count"])),
            tp,
        )
        add_corrected = torch.zeros_like(tp)
        add_broken = torch.zeros_like(tp)
    elif mode == "null":
        add_corrected = _binomial(
            fn, float(parameters["add_corrected_rate"]), generator)
        remove_corrected = _binomial(
            fp, float(parameters["remove_corrected_rate"]), generator)
        remove_broken = torch.zeros_like(tp)
        denominator = (tp + fp + fn).to(torch.float64)
        rho = torch.where(
            denominator > 0.0,
            tp.to(torch.float64) / denominator,
            torch.zeros_like(denominator),
        )
        target_breaks = torch.where(
            rho > 0.0,
            torch.round(
                (
                    add_corrected.to(torch.float64)
                    + rho * remove_corrected.to(torch.float64)
                ) / rho.clamp_min(1e-12)
            ),
            torch.zeros_like(rho),
        ).long()
        add_corrected = torch.where(
            rho > 0.0, add_corrected, torch.zeros_like(add_corrected))
        add_broken = torch.minimum(target_breaks, tn)
    elif mode == "mixture":
        positive = torch.rand(
            shape, dtype=torch.float64, device=device,
            generator=generator) < float(parameters["positive_probability"])
        add_rate = torch.where(
            positive,
            torch.full_like(latent, float(parameters["positive_add_rate"])),
            torch.full_like(latent, float(parameters["negative_add_rate"])),
        )
        remove_corrected_rate = torch.where(
            positive,
            torch.full_like(
                latent, float(parameters["positive_remove_corrected_rate"])),
            torch.full_like(
                latent, float(parameters["negative_remove_corrected_rate"])),
        )
        remove_broken_rate = torch.where(
            positive,
            torch.full_like(
                latent, float(parameters["positive_remove_broken_rate"])),
            torch.full_like(
                latent, float(parameters["negative_remove_broken_rate"])),
        )
        add_broken_rate = torch.where(
            positive,
            torch.full_like(
                latent, float(parameters["positive_add_broken_rate"])),
            torch.full_like(
                latent, float(parameters["negative_add_broken_rate"])),
        )
        add_corrected = _binomial(fn, add_rate, generator)
        add_broken = _binomial(tn, add_broken_rate, generator)
        remove_corrected = _binomial(
            fp, remove_corrected_rate, generator)
        remove_broken = _binomial(tp, remove_broken_rate, generator)
    else:
        dependence = float(parameters.get("dependence_strength", 0.0))
        add_rate = _scaled_probability(
            parameters["add_corrected_rate"], latent, dependence)
        add_broken_rate = _scaled_probability(
            parameters["add_broken_rate"], -latent, dependence)
        remove_corrected_rate = _scaled_probability(
            parameters["remove_corrected_rate"], latent, dependence)
        remove_broken_rate = _scaled_probability(
            parameters["remove_broken_rate"], -latent, dependence)
        if mode == "temporal":
            drift = torch.randn(
                shape, dtype=torch.float64, device=device,
                generator=generator)
            tp = _poisson_counts(
                template["tp"],
                torch.exp(
                    float(parameters["ratio_drift_strength"]) * drift
                    - 0.5 * float(
                        parameters["ratio_drift_strength"]) ** 2
                ),
                generator,
            )
            add_rate = (
                add_rate
                * torch.exp(
                    -float(parameters["utility_drift_strength"]) * drift)
            ).clamp(0.0, 1.0)
        add_corrected = _binomial(fn, add_rate, generator)
        add_broken = _binomial(tn, add_broken_rate, generator)
        remove_corrected = _binomial(
            fp, remove_corrected_rate, generator)
        remove_broken = _binomial(tp, remove_broken_rate, generator)

    routed_tp = tp + add_corrected - remove_broken
    routed_fp = fp + add_broken - remove_corrected
    routed_fn = fn - add_corrected + remove_broken
    base_denominator = (2 * tp + fp + fn).to(torch.float64)
    routed_denominator = (
        2 * routed_tp + routed_fp + routed_fn).to(torch.float64)
    base_f1 = torch.where(
        base_denominator > 0.0,
        2.0 * tp.to(torch.float64) / base_denominator,
        torch.zeros_like(base_denominator),
    )
    routed_f1 = torch.where(
        routed_denominator > 0.0,
        2.0 * routed_tp.to(torch.float64) / routed_denominator,
        torch.zeros_like(routed_denominator),
    )
    denominator = (tp + fp + fn).to(torch.float64)
    rho = torch.where(
        denominator > 0.0,
        tp.to(torch.float64) / denominator,
        torch.zeros_like(denominator),
    )
    utility = (
        add_corrected.to(torch.float64)
        - remove_broken.to(torch.float64)
        + rho * (
            remove_corrected.to(torch.float64)
            - add_broken.to(torch.float64)
        )
    )
    delta = routed_f1 - base_f1
    identity_failure = (
        ((delta > 1e-12) != (utility > 1e-12))
        | ((delta < -1e-12) != (utility < -1e-12))
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "add_corrected": add_corrected,
        "add_broken": add_broken,
        "remove_corrected": remove_corrected,
        "remove_broken": remove_broken,
        "rho": rho,
        "utility": utility,
        "delta": delta,
        "identity_failure": identity_failure,
    }


def population_target(deltas, epsilon, pi0):
    deltas = np.asarray(deltas, dtype=np.float64)
    mean = float(deltas.mean())
    positive = float((deltas >= float(epsilon)).mean())
    harmful = float((deltas <= -float(epsilon)).mean())
    if mean > 0.0 and positive >= float(pi0):
        decision = DECISION_IMPROVEMENT
    elif mean < 0.0 and harmful >= float(pi0):
        decision = DECISION_HARM
    else:
        decision = DECISION_INSUFFICIENT
    return {
        "mean_delta": mean,
        "practical_improvement_probability": positive,
        "practical_harm_probability": harmful,
        "decision": decision,
    }


def _iid_decision(counts, index, delta):
    interval = iid_paired_f1_interval(
        tp=int(counts["tp"][index]),
        fp=int(counts["fp"][index]),
        fn=int(counts["fn"][index]),
        tn=int(counts["tn"][index]),
        add_corrected=int(counts["add_corrected"][index]),
        add_broken=int(counts["add_broken"][index]),
        remove_corrected=int(counts["remove_corrected"][index]),
        remove_broken=int(counts["remove_broken"][index]),
        delta=delta,
    )
    if interval["lower"] > 0.0:
        return DECISION_IMPROVEMENT
    if interval["upper"] < 0.0:
        return DECISION_HARM
    return DECISION_INSUFFICIENT


def run_template(
        template_name,
        template,
        scenario,
        seed,
        spec,
        device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    reference = simulate_streams(
        template,
        scenario,
        (int(spec["reference_streams"]),),
        generator,
        device,
    )
    reference_deltas = reference["delta"].cpu().numpy()
    target = population_target(
        reference_deltas, spec["epsilon"], spec["pi0"])
    expected = SCENARIO_TARGETS[scenario["name"]]
    if target["decision"] != expected:
        raise RuntimeError(
            f"{scenario['name']} {template_name} generated "
            f"{target['decision']} instead of {expected}"
        )

    replicate_count = int(spec["replicates"])
    stream_count = int(spec["streams"])
    generated = simulate_streams(
        template,
        scenario,
        (replicate_count, stream_count),
        generator,
        device,
    )
    identity_failures = int(generated["identity_failure"].sum())
    deltas = generated["delta"].cpu().numpy()
    cpu_counts = {
        key: value.cpu().numpy()
        for key, value in generated.items()
        if key in {
            "tp", "fp", "fn", "tn",
            "add_corrected", "add_broken",
            "remove_corrected", "remove_broken",
        }
    }
    decisions = {
        method: {
            DECISION_IMPROVEMENT: 0,
            DECISION_HARM: 0,
            DECISION_INSUFFICIENT: 0,
            "stopping_checkpoints": [],
        }
        for method in METHODS
    }
    simultaneous_mean_coverage = 0
    records = []
    endpoint_delta = (
        float(spec["delta"])
        / (len(spec["checkpoints"]) * 3)
    )
    for replicate in range(replicate_count):
        values = deltas[replicate]
        method_rows = {}
        method_rows["one_stream"] = {
            "decision": one_stream_decision(
                values[0], spec["epsilon"]),
            "stopping_checkpoint": 1,
        }
        method_rows["row_iid"] = {
            "decision": _iid_decision(
                {
                    key: value[replicate]
                    for key, value in cpu_counts.items()
                },
                0,
                spec["delta"],
            ),
            "stopping_checkpoint": 1,
        }
        method_rows["mean_only"] = mean_only_trajectory(
            values,
            checkpoints=spec["checkpoints"],
            delta=spec["delta"],
        )
        method_rows["replication_only"] = replication_only_trajectory(
            values,
            checkpoints=spec["checkpoints"],
            epsilon=spec["epsilon"],
            pi0=spec["pi0"],
            delta=spec["delta"],
        )
        dgr = dgr_f1_trajectory(
            values,
            checkpoints=spec["checkpoints"],
            epsilon=spec["epsilon"],
            pi0=spec["pi0"],
            delta=spec["delta"],
        )
        method_rows["dgr_f1"] = {
            "decision": dgr["decision"],
            "stopping_checkpoint": dgr["stopping_checkpoint"],
        }
        covered = all(
            row["mean"]["lower"] <= target["mean_delta"]
            <= row["mean"]["upper"]
            for row in dgr["trajectory"]
        )
        simultaneous_mean_coverage += int(covered)
        for method, row in method_rows.items():
            decisions[method][row["decision"]] += 1
            if row["stopping_checkpoint"] is not None:
                decisions[method]["stopping_checkpoints"].append(
                    int(row["stopping_checkpoint"]))
        records.append({
            "template": template_name,
            "replicate": replicate,
            "target": expected,
            "stream_delta_mean": float(values.mean()),
            "stream_delta_std": float(values.std(ddof=1)),
            "method_decisions": method_rows,
            "dgr_mean_intervals": [
                {
                    "checkpoint": row["checkpoint"],
                    "lower": row["mean"]["lower"],
                    "upper": row["mean"]["upper"],
                }
                for row in dgr["trajectory"]
            ],
            "simultaneous_mean_covered": covered,
        })

    summaries = {}
    for method, row in decisions.items():
        stopping = row.pop("stopping_checkpoints")
        summaries[method] = {
            "decision_counts": row,
            "improvement_rate": (
                row[DECISION_IMPROVEMENT] / replicate_count),
            "harm_rate": row[DECISION_HARM] / replicate_count,
            "insufficient_rate": (
                row[DECISION_INSUFFICIENT] / replicate_count),
            "median_stopping_checkpoint": (
                float(np.median(stopping)) if stopping else None),
        }
    return {
        "template": template_name,
        "template_counts": template,
        "population_target": target,
        "registered_target": expected,
        "identity_failures": identity_failures,
        "reference_identity_failures": int(
            reference["identity_failure"].sum()),
        "simultaneous_mean_coverage_rate": (
            simultaneous_mean_coverage / replicate_count),
        "endpoint_delta": endpoint_delta,
        "methods": summaries,
    }, records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--reference-streams", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    commit = verify_repository(args.expected_commit)
    spec = json.loads(args.spec.read_text())
    if args.replicates is not None:
        spec["replicates"] = int(args.replicates)
    if args.reference_streams is not None:
        spec["reference_streams"] = int(args.reference_streams)
    task = spec["tasks"][args.task_index]
    scenario = {
        **spec["scenarios"][task["scenario"]],
        "name": task["scenario"],
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    templates = {}
    records = []
    for offset, (name, template) in enumerate(
            spec["templates"].items()):
        result, rows = run_template(
            name,
            template,
            scenario,
            int(task["seed"]) + offset * 10000,
            spec,
            device,
        )
        templates[name] = result
        records.extend(rows)

    output_dir = args.output_root / (
        f"{task['scenario']}_seed{task['seed']}_{commit[:8]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "dgr_f1_trials.jsonl"
    records_path.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in records
    ))
    manifest = {
        "experiment": "DGR_F1_Phase0_controlled_development",
        "method_version": 1,
        "scenario": task["scenario"],
        "registered_target": SCENARIO_TARGETS[task["scenario"]],
        "scenario_config": scenario,
        "seed": task["seed"],
        "git_commit": commit,
        "sampling_protocol": "dynamic_random_templates_no_loader_access",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "replicates": spec["replicates"],
        "streams_per_replicate": spec["streams"],
        "reference_streams": spec["reference_streams"],
        "checkpoints": spec["checkpoints"],
        "epsilon": spec["epsilon"],
        "pi0": spec["pi0"],
        "delta": spec["delta"],
        "endpoint_delta": (
            spec["delta"] / (len(spec["checkpoints"]) * 3)),
        "device": str(device),
        "templates": templates,
        "trial_records": str(records_path),
        "runtime_seconds": time.time() - started,
    }
    manifest_path = output_dir / "dgr_f1_phase0_manifest.json"
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "scenario": task["scenario"],
        "templates": {
            name: {
                "target": row["population_target"],
                "dgr_f1": row["methods"]["dgr_f1"],
            }
            for name, row in templates.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
