"""Temporal ratio-envelope certification for F1 interventions."""

import math
from statistics import NormalDist

import torch

from fraudGT.evidence.gtf1c import paired_policy_statistics


def directional_intervention_counts(labels, base, routed):
    """Count add/remove corrections and breaks for a binary policy."""
    labels = labels.bool()
    base = base.bool()
    routed = routed.bool()
    add = ~base & routed
    remove = base & ~routed
    return {
        "add_corrected": int((add & labels).sum()),
        "add_broken": int((add & ~labels).sum()),
        "remove_corrected": int((remove & ~labels).sum()),
        "remove_broken": int((remove & labels).sum()),
    }


def exact_f1_sign_utility(
        add_corrected,
        add_broken,
        remove_corrected,
        remove_broken,
        rho,
):
    """Return the exact numerator governing the sign of paired F1 change."""
    return (
        float(add_corrected)
        - float(remove_broken)
        + float(rho)
        * (float(remove_corrected) - float(add_broken))
    )


def base_f1_sensitivity_ratio(labels, base):
    """Return rho = TP / (TP + FP + FN) for the frozen base predictor."""
    labels = labels.bool()
    base = base.bool()
    tp = int((base & labels).sum())
    fp = int((base & ~labels).sum())
    fn = int((~base & labels).sum())
    total = tp + fp + fn
    return 0.0 if total == 0 else tp / total


def wilson_upper(successes, trials, delta=0.05):
    """One-sided Wilson upper confidence bound for a binomial ratio."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    z = NormalDist().inv_cdf(1.0 - float(delta))
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = proportion + z2 / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z2 / (4.0 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)


def combined_ratio_upper(views, delta=0.05):
    """Estimate rho upper bound from frozen-base confusion across views."""
    tp = fp = fn = 0
    for view in views:
        labels = view["labels"].bool()
        base = view["base"].bool()
        tp += int((base & labels).sum())
        fp += int((base & ~labels).sum())
        fn += int((~base & labels).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "denominator": tp + fp + fn,
        "point": (
            tp / (tp + fp + fn)
            if tp + fp + fn > 0
            else 0.0
        ),
        "upper": wilson_upper(tp, tp + fp + fn, delta=delta),
        "delta": float(delta),
    }


def directional_utility_rows(labels, base, routed, rho):
    """Construct additive rows whose sum is the exact F1 sign utility."""
    labels = labels.bool()
    base = base.bool()
    routed = routed.bool()
    rows = torch.zeros(
        labels.numel(), dtype=torch.float64, device=labels.device)
    add = ~base & routed
    remove = base & ~routed
    rows[add & labels] = 1.0
    rows[add & ~labels] = -float(rho)
    rows[remove & ~labels] = float(rho)
    rows[remove & labels] = -1.0
    return rows


def additive_utility_lcb(utility, groups=None, delta=0.05):
    """One-sided lower bound for mean additive intervention utility."""
    utility = utility.to(torch.float64)
    count = int(utility.numel())
    point = float(utility.mean()) if count else 0.0
    if count < 2:
        return {
            "point": point,
            "standard_error": math.inf,
            "lower_bound": -math.inf,
            "group_count": count,
        }

    centered = utility - utility.mean()
    if groups is None:
        group_count = count
        standard_error = float(utility.std(unbiased=True) / math.sqrt(count))
    else:
        groups = groups.to(utility.device)
        if groups.numel() != count:
            raise ValueError("group vector does not match utility rows")
        _, inverse = torch.unique(groups, return_inverse=True)
        group_count = int(inverse.max()) + 1 if inverse.numel() else 0
        if group_count < 2:
            standard_error = math.inf
        else:
            sums = torch.zeros(
                group_count, dtype=torch.float64, device=utility.device)
            sums.scatter_add_(0, inverse, centered)
            variance = (
                group_count / (group_count - 1.0)
                * sums.square().sum()
                / (count * count)
            )
            standard_error = float(torch.sqrt(variance))

    z = NormalDist().inv_cdf(1.0 - float(delta))
    lower_bound = (
        -math.inf
        if not math.isfinite(standard_error)
        else point - z * standard_error
    )
    return {
        "point": point,
        "standard_error": standard_error,
        "lower_bound": lower_bound,
        "group_count": group_count,
    }


def ratio_envelope_certification(
        view,
        routed,
        rho_upper,
        min_changes,
        delta,
        practical_delta,
):
    """Certify an intervention at both endpoints of [0, rho_upper]."""
    labels = view["labels"]
    base = view["base"]
    statistics = paired_policy_statistics(labels, base, routed)
    counts = directional_intervention_counts(labels, base, routed)
    endpoints = {}
    for name, rho in (("lower", 0.0), ("upper", float(rho_upper))):
        utility = directional_utility_rows(labels, base, routed, rho)
        endpoints[name] = {
            "rho": rho,
            "sum": float(utility.sum()),
            "exact_sum": exact_f1_sign_utility(
                counts["add_corrected"],
                counts["add_broken"],
                counts["remove_corrected"],
                counts["remove_broken"],
                rho,
            ),
            "time": additive_utility_lcb(
                utility, view["time_groups"], delta=delta),
            "graph": additive_utility_lcb(
                utility, view["graph_groups"], delta=delta),
        }

    lower_bounds = [
        endpoints[endpoint][grouping]["lower_bound"]
        for endpoint in ("lower", "upper")
        for grouping in ("time", "graph")
    ]
    qualified = (
        statistics["changed"] >= int(min_changes)
        and statistics["paired_f1_delta"] >= float(practical_delta)
        and all(bound > 0.0 for bound in lower_bounds)
    )
    return {
        "statistics": statistics,
        "directional_counts": counts,
        "rho_upper": float(rho_upper),
        "endpoint_results": endpoints,
        "worst_lower_bound": min(lower_bounds),
        "qualified": qualified,
    }
