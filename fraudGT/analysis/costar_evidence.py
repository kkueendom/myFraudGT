"""Pure utilities for paired COSTAR evidence diagnostics."""

from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve


VARIANT_KEYS = (
    "z_base",
    "a2_anchor",
    "costar_full",
    "evidence_only_total",
    "evidence_only_prototype",
    "evidence_only_costar",
    "evidence_off",
    "evidence_shuffled",
)


def best_f1_threshold(labels, margins):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(margins, -60.0, 60.0)))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def predictions(margins, threshold):
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(margins, -60.0, 60.0)))
    return (scores > float(threshold)).astype(np.int64)


def f1(labels, margins, threshold):
    return float(f1_score(
        np.asarray(labels).reshape(-1),
        predictions(margins, threshold),
        zero_division=0,
    ))


def build_variants(split, permutation):
    base = split["base_margin"].reshape(-1)
    anchor = split["anchor_margin"].reshape(-1)
    final = split["final_margin"].reshape(-1)
    prototype = split["prototype_margin_delta"].reshape(-1)
    costar = split["costar_margin_delta"].reshape(-1)
    total = split["total_evidence_margin_delta"].reshape(-1)
    return OrderedDict((
        ("z_base", base),
        ("a2_anchor", anchor),
        ("costar_full", final),
        ("evidence_only_total", total),
        ("evidence_only_prototype", prototype),
        ("evidence_only_costar", costar),
        ("evidence_off", base),
        ("evidence_shuffled", base + total[permutation]),
    ))


def evaluate_variants(val, test, val_permutation, test_permutation):
    val_variants = build_variants(val, val_permutation)
    test_variants = build_variants(test, test_permutation)
    result = OrderedDict()
    for key in VARIANT_KEYS:
        threshold = best_f1_threshold(val["labels"], val_variants[key])
        result[key] = {
            "threshold": threshold,
            "val_f1": f1(val["labels"], val_variants[key], threshold),
            "test_f1": f1(test["labels"], test_variants[key], threshold),
        }
    return result, val_variants, test_variants


def error_subset(labels, a2_margin, evidence_margin,
                 a2_threshold, evidence_threshold):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    a2_correct = predictions(a2_margin, a2_threshold) == labels
    evidence_correct = predictions(
        evidence_margin, evidence_threshold) == labels
    counts = OrderedDict((
        ("a2_wrong_evidence_right", int((~a2_correct & evidence_correct).sum())),
        ("a2_right_evidence_wrong", int((a2_correct & ~evidence_correct).sum())),
        ("both_wrong", int((~a2_correct & ~evidence_correct).sum())),
        ("both_right", int((a2_correct & evidence_correct).sum())),
    ))
    a2_wrong = int((~a2_correct).sum())
    a2_right = int(a2_correct.sum())
    counts["a2_wrong_total"] = a2_wrong
    counts["a2_right_total"] = a2_right
    counts["correction_rate_within_a2_errors"] = (
        counts["a2_wrong_evidence_right"] / max(a2_wrong, 1))
    counts["damage_rate_within_a2_correct"] = (
        counts["a2_right_evidence_wrong"] / max(a2_right, 1))
    counts["net_corrected_minus_broken"] = (
        counts["a2_wrong_evidence_right"] -
        counts["a2_right_evidence_wrong"])
    return counts


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: None for key in (
            "mean", "p00", "p25", "p50", "p75", "p90", "p95", "p99",
            "p100")}
    quantiles = np.quantile(
        finite, [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    return {
        "mean": float(finite.mean()),
        "p00": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "p100": float(quantiles[7]),
    }


def changed_decisions(labels, reference_margin, candidate_margin, threshold):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    reference = predictions(reference_margin, threshold)
    candidate = predictions(candidate_margin, threshold)
    changed = reference != candidate
    corrected = changed & (reference != labels) & (candidate == labels)
    broken = changed & (reference == labels) & (candidate != labels)
    changed_count = int(changed.sum())
    return {
        "sample_count": int(labels.size),
        "changed_count": changed_count,
        "changed_rate": changed_count / max(int(labels.size), 1),
        "corrected_count": int(corrected.sum()),
        "broken_count": int(broken.sum()),
        "other_changed_count": int((changed & ~corrected & ~broken).sum()),
    }


def contribution_audit(test, full_threshold):
    base = test["base_margin"].reshape(-1)
    anchor = test["anchor_margin"].reshape(-1)
    final = test["final_margin"].reshape(-1)
    prototype = test["prototype_margin_delta"].reshape(-1)
    correction = test["costar_margin_delta"].reshape(-1)
    total = test["total_evidence_margin_delta"].reshape(-1)
    denominator = np.maximum(np.abs(base), 1e-6)
    fallback = test["fallback_mask"].reshape(-1).astype(bool)
    correction_active = np.abs(correction) > 1e-4
    confidence_pass = test["router_consistency"].reshape(-1) >= 0.25
    return {
        "absolute_margins": {
            "z_base": distribution(np.abs(base)),
            "prototype_delta": distribution(np.abs(prototype)),
            "costar_delta": distribution(np.abs(correction)),
            "total_evidence_delta": distribution(np.abs(total)),
        },
        "weights": {
            "prototype_alpha": distribution(
                test["prototype_alpha"].reshape(-1)),
            "prototype_ready": distribution(
                test["prototype_ready"].reshape(-1)),
            "prototype_reliability": distribution(
                test["prototype_reliability"].reshape(-1)),
            "router_consistency": distribution(
                test["router_consistency"].reshape(-1)),
            "residual_weight": distribution(
                test["residual_weight"].reshape(-1)),
        },
        # COSTAR does not implement a hard deployment gate. These rates expose
        # its diagnostic conditions without falsely implying that correction
        # is actually disabled when a condition fails.
        "hard_gate_present": False,
        "applied_correction_rate": float(correction_active.mean()),
        "confidence_pass_rate": float(confidence_pass.mean()),
        "diagnostic_open_rate": float((~fallback).mean()),
        "diagnostic_fallback_rate": float(fallback.mean()),
        "relative_contribution": {
            "prototype_over_base": distribution(
                np.abs(prototype) / denominator),
            "costar_over_base": distribution(
                np.abs(correction) / denominator),
            "total_evidence_over_base": distribution(
                np.abs(total) / denominator),
        },
        "costar_vs_a2_decision_changes": changed_decisions(
            test["labels"], anchor, final, full_threshold),
        "full_vs_z_base_decision_changes": changed_decisions(
            test["labels"], base, final, full_threshold),
    }


def gradient_group_summary(named_parameters):
    groups = {
        "costar_router": [],
        "prototype_branch": [],
        "z_base_decoder": [],
        "encoder_and_other": [],
    }
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        if "costar_router_ema" in name:
            continue
        if "costar_router" in name:
            group = "costar_router"
        elif "eg_proto_head" in name or "eg_proto_alpha" in name:
            group = "prototype_branch"
        elif "layer_post_mp" in name:
            group = "z_base_decoder"
        else:
            group = "encoder_and_other"
        groups[group].append(parameter.grad.detach().float().reshape(-1).cpu())

    result = {}
    for name, tensors in groups.items():
        if not tensors:
            result[name] = {
                "numel": 0, "l2_norm": 0.0, "mean_abs": 0.0,
                "max_abs": 0.0, "nonzero_fraction": 0.0,
            }
            continue
        values = torch.cat(tensors)
        result[name] = {
            "numel": int(values.numel()),
            "l2_norm": float(values.norm()),
            "mean_abs": float(values.abs().mean()),
            "max_abs": float(values.abs().max()),
            "nonzero_fraction": float((values != 0).float().mean()),
        }
    return result
