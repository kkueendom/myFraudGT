"""Graph-time grouped risk bounds for paired evidence intervention."""

import math
from statistics import NormalDist

import torch


def row_wilson_upper(selected, broken, family_size, delta):
    """Bonferroni-adjusted Wilson upper bound over selected rows."""
    selected_count = selected.sum(dim=-1).to(torch.float64)
    broken_count = (
        selected & broken[:, None, :]
    ).sum(dim=-1).to(torch.float64)
    p_hat = broken_count / selected_count.clamp_min(1.0)
    tail = delta / max(int(family_size), 1)
    z = NormalDist().inv_cdf(1.0 - tail)
    z2 = z * z
    denominator = 1.0 + z2 / selected_count.clamp_min(1.0)
    center = p_hat + z2 / (2.0 * selected_count.clamp_min(1.0))
    radius = z * torch.sqrt(
        (
            p_hat * (1.0 - p_hat)
            + z2 / (4.0 * selected_count.clamp_min(1.0))
        )
        / selected_count.clamp_min(1.0)
    )
    upper = (center + radius) / denominator
    return torch.where(
        selected_count > 0,
        upper.clamp(max=1.0),
        torch.ones_like(upper),
    )


def grouped_empirical_bernstein_upper(
        selected, broken, group_size, family_size, delta):
    """Upper bound over independent groups of within-group break rates.

    The bound is an empirical validation object, not a claimed theorem for
    arbitrary transaction graphs. Phase 0A tests its behavior when ``group``
    is the true dependency unit and when it is misspecified.
    """
    if selected.ndim != 3 or broken.ndim != 2:
        raise ValueError("expected selected [R,K,N] and broken [R,N]")
    replicate_count, policy_count, row_count = selected.shape
    if row_count % int(group_size) != 0:
        raise ValueError("row count must be divisible by group_size")
    group_count = row_count // int(group_size)
    grouped_selected = selected.reshape(
        replicate_count, policy_count, group_count, int(group_size))
    selected_count = grouped_selected.sum(dim=-1).to(torch.float64)
    grouped_broken = (
        grouped_selected
        & broken[:, None, :].expand(
            -1, policy_count, -1
        ).reshape(
            replicate_count, policy_count, group_count, int(group_size)
        )
    ).sum(dim=-1).to(torch.float64)

    active = selected_count > 0
    active_count = active.sum(dim=-1).to(torch.float64)
    rates = grouped_broken / selected_count.clamp_min(1.0)
    means = (
        rates * active.to(torch.float64)
    ).sum(dim=-1) / active_count.clamp_min(1.0)
    centered = rates - means[:, :, None]
    variances = (
        centered.square() * active.to(torch.float64)
    ).sum(dim=-1) / (active_count - 1.0).clamp_min(1.0)

    log_term = math.log(
        3.0 * max(int(family_size), 1) / float(delta))
    upper = (
        means
        + torch.sqrt(
            2.0 * variances * log_term
            / active_count.clamp_min(1.0)
        )
        + 3.0 * log_term / active_count.clamp_min(1.0)
    )
    valid = active_count >= 2
    return torch.where(
        valid,
        upper.clamp(max=1.0),
        torch.ones_like(upper),
    )


def grouped_hoeffding_upper(
        selected, broken, group_size, family_size, delta):
    """Bonferroni-adjusted Hoeffding bound over group break rates."""
    if selected.ndim != 3 or broken.ndim != 2:
        raise ValueError("expected selected [R,K,N] and broken [R,N]")
    replicate_count, policy_count, row_count = selected.shape
    if row_count % int(group_size) != 0:
        raise ValueError("row count must be divisible by group_size")
    group_count = row_count // int(group_size)
    grouped_selected = selected.reshape(
        replicate_count, policy_count, group_count, int(group_size))
    selected_count = grouped_selected.sum(dim=-1).to(torch.float64)
    grouped_broken = (
        grouped_selected
        & broken[:, None, :].expand(
            -1, policy_count, -1
        ).reshape(
            replicate_count, policy_count, group_count, int(group_size)
        )
    ).sum(dim=-1).to(torch.float64)
    active = selected_count > 0
    active_count = active.sum(dim=-1).to(torch.float64)
    rates = grouped_broken / selected_count.clamp_min(1.0)
    means = (
        rates * active.to(torch.float64)
    ).sum(dim=-1) / active_count.clamp_min(1.0)
    log_term = math.log(
        max(int(family_size), 1) / float(delta))
    upper = means + torch.sqrt(
        log_term / (2.0 * active_count.clamp_min(1.0)))
    return torch.where(
        active_count > 0,
        upper.clamp(max=1.0),
        torch.ones_like(upper),
    )


def select_max_coverage_policy(
        coverage, net_utility, upper_bound, alpha, min_coverage):
    """Select the highest-coverage policy satisfying locked constraints."""
    safe = (
        (upper_bound <= float(alpha))
        & (net_utility > 0)
        & (coverage >= float(min_coverage))
    )
    ranked = torch.where(
        safe,
        coverage,
        torch.full_like(coverage, -1.0),
    )
    indices = ranked.argmax(dim=-1)
    any_safe = safe.any(dim=-1)
    return torch.where(
        any_safe,
        indices,
        torch.full_like(indices, -1),
    )


def gather_policy(values, indices, default=0.0):
    """Gather one policy value per replicate, preserving abstentions."""
    if values.ndim != 2 or indices.ndim != 1:
        raise ValueError("expected values [R,K] and indices [R]")
    gathered = values.gather(
        1, indices.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(
        indices >= 0,
        gathered,
        torch.full_like(gathered, float(default)),
    )
