"""Graph-time paired F1 certification utilities."""

import math
from statistics import NormalDist

import torch


def confusion_counts(labels, predictions):
    labels = labels.bool()
    predictions = predictions.bool()
    return {
        "tp": int((predictions & labels).sum()),
        "fp": int((predictions & ~labels).sum()),
        "fn": int((~predictions & labels).sum()),
        "tn": int((~predictions & ~labels).sum()),
    }


def f1_from_counts(tp, fp, fn):
    denominator = 2.0 * float(tp) + float(fp) + float(fn)
    return 0.0 if denominator <= 0.0 else 2.0 * float(tp) / denominator


def paired_policy_statistics(labels, base, routed):
    base_counts = confusion_counts(labels, base)
    routed_counts = confusion_counts(labels, routed)
    changed = base.bool() != routed.bool()
    corrected = changed & (base.bool() != labels.bool()) & (
        routed.bool() == labels.bool())
    broken = changed & (base.bool() == labels.bool()) & (
        routed.bool() != labels.bool())
    base_f1 = f1_from_counts(
        base_counts["tp"], base_counts["fp"], base_counts["fn"])
    routed_f1 = f1_from_counts(
        routed_counts["tp"], routed_counts["fp"], routed_counts["fn"])
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "broken": int(broken.sum()),
        "net": int(corrected.sum() - broken.sum()),
        "base_f1": base_f1,
        "routed_f1": routed_f1,
        "paired_f1_delta": routed_f1 - base_f1,
    }


def _f1_and_gradient(contributions):
    means = contributions.to(torch.float64).mean(dim=0)
    tp, fp, fn = means.unbind()
    denominator = 2.0 * tp + fp + fn
    if float(denominator) <= 0.0:
        return (
            torch.zeros(
                (), dtype=torch.float64, device=contributions.device),
            torch.zeros(
                3, dtype=torch.float64, device=contributions.device),
        )
    value = 2.0 * tp / denominator
    gradient = torch.stack((
        2.0 * (fp + fn) / denominator.square(),
        -2.0 * tp / denominator.square(),
        -2.0 * tp / denominator.square(),
    ))
    return value, gradient


def paired_f1_influence(labels, base, routed):
    labels = labels.bool()
    base = base.bool()
    routed = routed.bool()
    routed_rows = torch.stack((
        routed & labels,
        routed & ~labels,
        ~routed & labels,
    ), dim=1).to(torch.float64)
    base_rows = torch.stack((
        base & labels,
        base & ~labels,
        ~base & labels,
    ), dim=1).to(torch.float64)
    routed_f1, routed_gradient = _f1_and_gradient(routed_rows)
    base_f1, base_gradient = _f1_and_gradient(base_rows)
    influence = (
        (routed_rows - routed_rows.mean(dim=0)) @ routed_gradient
        - (base_rows - base_rows.mean(dim=0)) @ base_gradient
    )
    return float(routed_f1 - base_f1), influence


def paired_f1_lcb(labels, base, routed, groups=None, delta=0.05):
    point, influence = paired_f1_influence(labels, base, routed)
    count = int(influence.numel())
    if count < 2:
        return {
            "point": point,
            "standard_error": math.inf,
            "lower_bound": -math.inf,
            "group_count": count,
        }
    if groups is None:
        standard_error = float(
            influence.std(unbiased=True) / math.sqrt(count))
        group_count = count
    else:
        groups = groups.to(influence.device)
        if groups.numel() != count:
            raise ValueError("group vector does not match prediction rows")
        _, inverse = torch.unique(groups, return_inverse=True)
        group_count = int(inverse.max()) + 1 if inverse.numel() else 0
        if group_count < 2:
            standard_error = math.inf
        else:
            sums = torch.zeros(
                group_count, dtype=torch.float64,
                device=influence.device)
            sums.scatter_add_(0, inverse, influence)
            variance = (
                group_count / (group_count - 1.0)
                * sums.square().sum()
                / (count * count)
            )
            standard_error = float(torch.sqrt(variance))
    z = NormalDist().inv_cdf(1.0 - float(delta))
    lower_bound = (
        -math.inf if not math.isfinite(standard_error)
        else point - z * standard_error
    )
    return {
        "point": point,
        "standard_error": standard_error,
        "lower_bound": lower_bound,
        "group_count": group_count,
    }


def time_block_groups(timestamps, block_count=32):
    count = int(timestamps.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    order = torch.argsort(timestamps.cpu(), stable=True)
    groups = torch.empty(count, dtype=torch.long)
    groups[order] = (
        torch.arange(count) * int(block_count) // count
    ).clamp_max(int(block_count) - 1)
    return groups


def graph_time_groups(
        source, destination, timestamps, block_count=32):
    """Chronological blocks with shared-entity components within each block."""
    source = source.cpu()
    destination = destination.cpu()
    blocks = time_block_groups(timestamps, block_count)
    count = int(source.numel())
    groups = torch.empty(count, dtype=torch.long)
    next_group = 0
    for block_id in range(int(block_count)):
        indices = torch.where(blocks == block_id)[0]
        parent = list(range(indices.numel()))

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        last = {}
        index_list = indices.tolist()
        for local, global_index in enumerate(index_list):
            for entity in (
                int(source[global_index]), int(destination[global_index])
            ):
                if entity in last:
                    union(local, last[entity])
                last[entity] = local
        mapping = {}
        for local, global_index in enumerate(index_list):
            root = find(local)
            if root not in mapping:
                mapping[root] = next_group
                next_group += 1
            groups[global_index] = mapping[root]
    return groups


def grouped_break_upper(
        active, broken, groups, family_size=1, delta=0.05):
    active = active.bool()
    broken = broken.bool()
    groups = groups.to(active.device)
    active_groups = groups[active]
    if not active_groups.numel():
        return 1.0
    _, inverse = torch.unique(active_groups, return_inverse=True)
    counts = torch.bincount(inverse)
    broken_counts = torch.bincount(
        inverse, weights=broken[active].float())
    rates = broken_counts / counts
    mean = float(rates.mean())
    group_count = int(counts.numel())
    radius = math.sqrt(
        math.log(max(int(family_size), 1) / float(delta))
        / (2.0 * group_count)
    )
    return min(1.0, mean + radius)
