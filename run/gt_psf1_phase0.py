#!/usr/bin/env python3
"""Untouched synthetic development screen for GT-psF1."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudGT.evidence.gt_psf1 import (  # noqa: E402
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    allocation_plan,
    dependency_adjacency,
    dependency_diagnostics,
    graph_diffusion_kernel,
)


EFFECTS = {
    "null": 0.0,
    "improvement": 1.0,
    "harm": -1.0,
}
METHODS = (
    "row_iid",
    "target_iid",
    "endpoint_dyadic",
    "graph_hac",
    "gt_psf1",
)


def verify_repository(expected_commit=None):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("tracked repository state is dirty")
    if expected_commit and head != expected_commit:
        raise RuntimeError(
            "repository commit does not match the requested commit")
    return head


def _scenario_structure(target_count, scenario, device):
    target = torch.arange(target_count, device=device)
    if scenario == "iid_deterministic":
        source = 2 * target
        destination = source + 1
        timestamps = target.to(torch.float64) * 100.0
        temporal_window = None
        graph_order = 1
        time_scale = None
    elif scenario == "endpoint_dyadic":
        block = target // 8
        source = block * 1000
        destination = block * 1000 + target.remainder(8) + 1
        timestamps = block.to(torch.float64) * 100.0
        temporal_window = None
        graph_order = 1
        time_scale = None
    elif scenario == "two_hop_spillover":
        block = target // 16
        position = target.remainder(16)
        source = block * 1000 + position
        destination = source + 1
        timestamps = block.to(torch.float64) * 100.0 + position
        temporal_window = None
        graph_order = 2
        time_scale = None
    elif scenario == "temporal_ar1":
        block = target // 12
        position = target.remainder(12)
        source = 2 * target
        destination = source + 1
        timestamps = block.to(torch.float64) * 100.0 + position
        temporal_window = 3.0
        graph_order = 2
        time_scale = 3.0
    elif scenario in (
        "low_sampler_variance",
        "high_sampler_variance",
    ):
        block = target // 8
        position = target.remainder(8)
        source = block * 1000
        destination = block * 1000 + position + 1
        timestamps = block.to(torch.float64) * 100.0 + position
        temporal_window = 2.0
        graph_order = 2
        time_scale = 3.0
    elif scenario == "dense_hub_out_of_scope":
        source = torch.zeros_like(target)
        destination = target + 1
        timestamps = target.to(torch.float64)
        temporal_window = None
        graph_order = 1
        time_scale = None
    else:
        raise ValueError("unknown scenario: {}".format(scenario))
    adjacency = dependency_adjacency(
        source,
        destination,
        timestamps=timestamps,
        temporal_window=temporal_window,
    )
    kernel = graph_diffusion_kernel(
        adjacency,
        timestamps=timestamps,
        graph_order=graph_order,
        time_scale=time_scale,
    )
    endpoint = dependency_adjacency(source, destination)
    dyadic_kernel = endpoint.to(torch.float64)
    dyadic_kernel.fill_diagonal_(True)
    return {
        "source": source,
        "destination": destination,
        "timestamps": timestamps,
        "adjacency": adjacency,
        "kernel": kernel,
        "dyadic_kernel": dyadic_kernel,
    }


def build_topology(template, scenario, device):
    topology = _scenario_structure(
        int(template["targets"]), scenario, device)
    count = topology["kernel"].shape[0]
    jitter = torch.eye(
        count, dtype=torch.float64, device=device)
    topology["factor"] = torch.linalg.cholesky(
        topology["kernel"] + 1e-6 * jitter)
    topology["diagnostics"] = dependency_diagnostics(
        topology["adjacency"], topology["kernel"])
    return topology


def _batched_value_gradient(moments):
    def value_gradient(values):
        tp, fp, fn = values.unbind(dim=-1)
        denominator = (2.0 * tp + fp + fn).clamp_min(1e-12)
        value = 2.0 * tp / denominator
        gradient = torch.stack((
            2.0 * (fp + fn) / denominator.square(),
            -2.0 * tp / denominator.square(),
            -2.0 * tp / denominator.square(),
        ), dim=-1)
        return value, gradient

    value_a, gradient_a = value_gradient(moments[..., :3])
    value_b, gradient_b = value_gradient(moments[..., 3:])
    return value_b - value_a, torch.cat((-gradient_a, gradient_b), dim=-1)


def _contributions(labels, predictions_a, predictions_b):
    labels = labels.bool()

    def rows(predictions):
        return torch.stack((
            predictions & labels,
            predictions & ~labels,
            ~predictions & labels,
        ), dim=-1).to(torch.float64)

    return torch.cat((rows(predictions_a), rows(predictions_b)), dim=-1)


def _graph_draws(experiments, topology, generator):
    count = topology["kernel"].shape[0]
    independent = torch.randn(
        (experiments, count),
        dtype=torch.float64,
        device=topology["kernel"].device,
        generator=generator,
    )
    return independent @ topology["factor"].T


def simulate_experiments(
        template,
        scenario,
        effect_multiplier,
        experiments,
        repeats,
        parameters,
        topology,
        generator,
):
    device = topology["kernel"].device
    target_count = topology["kernel"].shape[0]
    base_separation = float(parameters["base_separation"])
    effect = (
        float(parameters["alternative_separation_change"])
        * float(effect_multiplier)
    )
    target_noise_sd = float(parameters["target_noise_sd"])
    shared_fraction = float(
        parameters["shared_classifier_noise_fraction"])
    label_shock = float(parameters["label_logit_graph_shock"])
    dependence_sd, sampler_sd = parameters[
        "scenario_dependence_and_sampler_sd"][scenario]
    dependence_sd = float(dependence_sd)
    sampler_sd = float(sampler_sd)

    graph_label = _graph_draws(experiments, topology, generator)
    graph_advantage = _graph_draws(experiments, topology, generator)
    prevalence = float(template["fraud_prevalence"])
    base_logit = math.log(prevalence / (1.0 - prevalence))
    label_probability = torch.sigmoid(
        base_logit + label_shock * graph_label)
    labels_target = torch.rand(
        (experiments, target_count),
        dtype=torch.float64,
        device=device,
        generator=generator,
    ) < label_probability
    sign = labels_target.to(torch.float64) * 2.0 - 1.0

    common_target = torch.randn(
        (experiments, target_count),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    noise_a = torch.randn(
        (experiments, target_count),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    noise_b = torch.randn(
        (experiments, target_count),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    common_scale = math.sqrt(shared_fraction)
    independent_scale = math.sqrt(1.0 - shared_fraction)
    target_a = target_noise_sd * (
        common_scale * common_target + independent_scale * noise_a)
    target_b = target_noise_sd * (
        common_scale * common_target + independent_scale * noise_b)
    advantage = dependence_sd * graph_advantage * sign
    score_a = base_separation * sign - advantage + target_a
    score_b = (base_separation + effect) * sign + advantage + target_b

    if sampler_sd > 0.0:
        common_sampler = torch.randn(
            (experiments, target_count, repeats),
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        sampler_a = torch.randn(
            (experiments, target_count, repeats),
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        sampler_b = torch.randn(
            (experiments, target_count, repeats),
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        sampler_a = sampler_sd * (
            common_scale * common_sampler + independent_scale * sampler_a)
        sampler_b = sampler_sd * (
            common_scale * common_sampler + independent_scale * sampler_b)
    else:
        sampler_a = torch.zeros(
            (experiments, target_count, repeats),
            dtype=torch.float64,
            device=device,
        )
        sampler_b = torch.zeros_like(sampler_a)

    predictions_a = score_a[..., None] + sampler_a > 0.0
    predictions_b = score_b[..., None] + sampler_b > 0.0
    labels = labels_target[..., None].expand_as(predictions_a)
    contributions = _contributions(
        labels, predictions_a, predictions_b)
    target_means = contributions.mean(dim=2)
    moments = target_means.mean(dim=1)
    point, gradient = _batched_value_gradient(moments)
    target_influence = (
        (target_means - moments[:, None, :])
        * gradient[:, None, :]
    ).sum(dim=-1)

    pooled = contributions.reshape(
        experiments, target_count * repeats, 6)
    pooled_moments = pooled.mean(dim=1)
    row_point, row_gradient = _batched_value_gradient(pooled_moments)
    row_influence = (
        (pooled - pooled_moments[:, None, :])
        * row_gradient[:, None, :]
    ).sum(dim=-1)
    row_variance = row_influence.var(
        dim=1, unbiased=True) / (target_count * repeats)
    target_variance = target_influence.var(
        dim=1, unbiased=True) / target_count

    correction = target_count / (target_count - 1.0)
    graph_variance = torch.einsum(
        "bi,ij,bj->b",
        target_influence,
        topology["kernel"],
        target_influence,
    ) * correction / (target_count * target_count)
    dyadic_variance = torch.einsum(
        "bi,ij,bj->b",
        target_influence,
        topology["dyadic_kernel"],
        target_influence,
    ) * correction / (target_count * target_count)
    graph_variance = graph_variance.clamp_min(0.0)
    dyadic_variance = dyadic_variance.clamp_min(0.0)

    draw_deviation = (
        (
            contributions - target_means[:, :, None, :]
        ) * gradient[:, None, None, :]
    ).sum(dim=-1)
    if repeats > 1:
        within_per_draw = draw_deviation.var(
            dim=2, unbiased=True).mean(dim=1)
        within_current = (
            draw_deviation.var(dim=2, unbiased=True) / repeats
        ).sum(dim=1) / (target_count * target_count)
    else:
        within_per_draw = torch.full_like(point, float("nan"))
        within_current = torch.full_like(point, float("nan"))
    between_per_target = (
        target_count * (graph_variance - within_current)
    ).clamp_min(0.0)

    return {
        "point": point,
        "row_point": row_point,
        "variances": {
            "row_iid": row_variance,
            "target_iid": target_variance,
            "endpoint_dyadic": dyadic_variance,
            "graph_hac": graph_variance,
            "gt_psf1": graph_variance,
        },
        "between_per_target": between_per_target,
        "within_per_draw": within_per_draw,
        "positive_count_a": (
            2.0 * moments[:, 0] + moments[:, 1] + moments[:, 2]
        ) * target_count,
        "positive_count_b": (
            2.0 * moments[:, 3] + moments[:, 4] + moments[:, 5]
        ) * target_count,
    }


def _summarize_method(
        point,
        variance,
        truth,
        status,
        alpha,
        in_scope=None,
):
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    standard_error = variance.clamp_min(0.0).sqrt()
    lower = point - critical * standard_error
    upper = point + critical * standard_error
    valid = torch.isfinite(standard_error)
    if in_scope is not None:
        valid = valid & in_scope.bool()
    valid_count = int(valid.sum())
    if valid_count:
        coverage = float(
            ((lower[valid] <= truth) & (upper[valid] >= truth))
            .double().mean()
        )
        mean_standard_error = float(standard_error[valid].mean())
    else:
        coverage = None
        mean_standard_error = None
    return {
        "coverage": coverage,
        "false_or_true_improvement": float(
            ((lower > 0.0) & valid).double().mean()),
        "false_or_true_harm": float(
            ((upper < 0.0) & valid).double().mean()),
        "mean_point": float(point.mean()),
        "mean_standard_error": mean_standard_error,
        "in_scope_rate": valid_count / int(point.numel()),
        "out_of_scope_rate": 1.0 - valid_count / int(point.numel()),
        "status": status,
    }


def _reference_truth(
        template,
        scenario,
        effect_multiplier,
        experiments,
        repeats,
        parameters,
        topology,
        generator,
):
    if effect_multiplier == 0.0:
        return 0.0
    values = []
    remaining = int(experiments)
    while remaining:
        batch = min(512, remaining)
        result = simulate_experiments(
            template,
            scenario,
            effect_multiplier,
            batch,
            repeats,
            parameters,
            topology,
            generator,
        )
        values.append(result["point"])
        remaining -= batch
    return float(torch.cat(values).mean())


def _scope_status(topology):
    diagnostics = topology["diagnostics"]
    count = diagnostics["target_count"]
    reasons = []
    if diagnostics["max_degree"] > math.sqrt(count):
        reasons.append("max_degree")
    if diagnostics["max_component_fraction"] > 0.5:
        reasons.append("component_fraction")
    if diagnostics["effective_target_count"] < 30.0:
        reasons.append("effective_target_count")
    return (
        STATUS_OUT_OF_SCOPE if reasons else STATUS_OK,
        reasons,
    )


def _allocation_summary(
        reference_result,
        template,
        scenario,
        truth,
        experiments,
        parameters,
        topology,
        generator,
        replicate_grid,
        alpha,
):
    a = float(reference_result["between_per_target"].mean())
    b = float(reference_result["within_per_draw"].mean())
    plan = allocation_plan(
        a,
        b,
        target_cost=parameters["target_cost"],
        neighbor_cost=parameters["neighbor_cost"],
        replicate_grid=replicate_grid,
    )
    target_count = int(template["targets"])
    fractions = (0.25, 0.5, 0.75, 1.0)
    cells = []
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    for repeats in replicate_grid:
        for fraction in fractions:
            selected_count = max(32, int(target_count * fraction))
            selected_count -= selected_count % 8
            sub_template = dict(template)
            sub_template["targets"] = selected_count
            sub_topology = build_topology(
                sub_template, scenario, topology["kernel"].device)
            result = simulate_experiments(
                sub_template,
                scenario,
                1.0,
                experiments,
                repeats,
                parameters,
                sub_topology,
                generator,
            )
            standard_error = result["variances"]["gt_psf1"].sqrt()
            power = float(
                (
                    result["point"] - critical * standard_error > 0.0
                ).double().mean()
            )
            cost = selected_count * (
                float(parameters["target_cost"])
                + repeats * float(parameters["neighbor_cost"])
            )
            cells.append({
                "repeats": int(repeats),
                "targets": selected_count,
                "power": power,
                "cost": cost,
            })
    eligible = [cell for cell in cells if cell["power"] >= 0.80]
    empirical = min(
        eligible, key=lambda cell: cell["cost"]) if eligible else None
    selected_cells = [
        cell for cell in eligible
        if cell["repeats"] == plan["selected_replicates"]
    ]
    selected = min(
        selected_cells, key=lambda cell: cell["cost"]
    ) if selected_cells else None
    regret = math.inf
    if empirical is not None and selected is not None:
        regret = selected["cost"] / empirical["cost"] - 1.0
    return {
        "truth": truth,
        "between_per_target": a,
        "within_per_draw": b,
        "plan": plan,
        "cells": cells,
        "empirical_cheapest": empirical,
        "selected_feasible": selected,
        "cost_regret": regret,
    }


def run_task(args):
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    task = next(
        item for item in spec["tasks"]
        if int(item["task_id"]) == int(args.task_id)
    )
    commit = verify_repository(args.expected_commit)
    device = torch.device(
        args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    experiments = (
        int(args.experiments)
        if args.experiments is not None
        else int(spec["formal_experiments"])
    )
    reference_experiments = (
        int(args.reference_experiments)
        if args.reference_experiments is not None
        else int(spec["population_reference_experiments"])
    )
    parameters = spec["data_generating_parameters"]
    scenario = task["scenario"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    formal_generator = torch.Generator(device=device)
    formal_generator.manual_seed(int(task["seed"]))
    reference_generator = torch.Generator(device=device)
    reference_generator.manual_seed(int(task["reference_seed"]))
    result = {
        "schema_version": 1,
        "task": task,
        "commit": commit,
        "spec": str(spec_path),
        "spec_sha256": hashlib.sha256(
            spec_path.read_bytes()).hexdigest(),
        "device": str(device),
        "experiments": experiments,
        "reference_experiments": reference_experiments,
        "templates": {},
    }
    for template_name, template in spec["templates"].items():
        topology = build_topology(template, scenario, device)
        scope_status, scope_reasons = _scope_status(topology)
        template_result = {
            "topology": topology["diagnostics"],
            "scope_status": scope_status,
            "scope_reasons": scope_reasons,
            "effects": {},
        }
        reference_cache = {}
        for effect_name, direction in EFFECTS.items():
            truth = _reference_truth(
                template,
                scenario,
                direction,
                reference_experiments,
                int(spec["formal_replicates"]),
                parameters,
                topology,
                reference_generator,
            )
            reference_cache[effect_name] = truth
            formal = simulate_experiments(
                template,
                scenario,
                direction,
                experiments,
                int(spec["formal_replicates"]),
                parameters,
                topology,
                formal_generator,
            )
            method_results = {}
            point = formal["point"]
            low_positive = (
                (formal["positive_count_a"] < 10.0)
                | (formal["positive_count_b"] < 10.0)
            )
            for method in METHODS:
                method_point = (
                    formal["row_point"]
                    if method == "row_iid" else point
                )
                method_status = (
                    scope_status
                    if method == "gt_psf1" else STATUS_OK
                )
                in_scope = None
                if method == "gt_psf1":
                    in_scope = ~low_positive
                    if scope_status == STATUS_OUT_OF_SCOPE:
                        in_scope = torch.zeros_like(in_scope)
                method_results[method] = _summarize_method(
                    method_point,
                    formal["variances"][method],
                    truth,
                    method_status,
                    float(spec["one_sided_alpha"]),
                    in_scope=in_scope,
                )
            template_result["effects"][effect_name] = {
                "truth": truth,
                "methods": method_results,
                "low_positive_contribution_rate": float(
                    low_positive.double().mean()),
            }
        if scenario in (
            "low_sampler_variance",
            "high_sampler_variance",
        ):
            reference_pilot = simulate_experiments(
                template,
                scenario,
                1.0,
                max(512, reference_experiments // 4),
                int(spec["formal_replicates"]),
                parameters,
                topology,
                reference_generator,
            )
            template_result["allocation"] = _allocation_summary(
                reference_pilot,
                template,
                scenario,
                reference_cache["improvement"],
                max(64, experiments // 2),
                parameters,
                topology,
                formal_generator,
                tuple(spec["allocation_replicate_grid"]),
                float(spec["one_sided_alpha"]),
            )
        result["templates"][template_name] = template_result

    result["elapsed_seconds"] = time.time() - started
    result["completed"] = True
    (output / "phase0_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "task_id": task["task_id"],
        "scenario": scenario,
        "output": str(output),
        "elapsed_seconds": result["elapsed_seconds"],
        "completed": True,
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--spec",
        default=str(REPO_ROOT / "run" / "gt_psf1_phase0_spec.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--device")
    parser.add_argument("--experiments", type=int)
    parser.add_argument("--reference-experiments", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run_task(parse_args())
