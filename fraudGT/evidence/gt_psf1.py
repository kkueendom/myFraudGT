"""Graph-time paired F1 inference under stochastic neighborhood sampling."""

import math
from statistics import NormalDist

import torch


STATUS_OK = "OK"
STATUS_OUT_OF_SCOPE = "OUT_OF_SCOPE"


def paired_contributions(labels, predictions_a, predictions_b):
    labels = labels.bool()
    predictions_a = predictions_a.bool()
    predictions_b = predictions_b.bool()
    if not (
        labels.shape == predictions_a.shape == predictions_b.shape
    ):
        raise ValueError("labels and predictions must have matching shapes")

    def rows(predictions):
        return torch.stack((
            predictions & labels,
            predictions & ~labels,
            ~predictions & labels,
        ), dim=-1).to(torch.float64)

    return torch.cat((rows(predictions_a), rows(predictions_b)), dim=-1)


def _f1_value_gradient(moments):
    tp, fp, fn = moments.unbind()
    denominator = 2.0 * tp + fp + fn
    if float(denominator) <= 0.0:
        return (
            torch.zeros((), dtype=torch.float64, device=moments.device),
            torch.zeros(3, dtype=torch.float64, device=moments.device),
        )
    value = 2.0 * tp / denominator
    gradient = torch.stack((
        2.0 * (fp + fn) / denominator.square(),
        -2.0 * tp / denominator.square(),
        -2.0 * tp / denominator.square(),
    ))
    return value, gradient


def paired_f1_value_gradient(moments):
    moments = moments.to(torch.float64)
    if moments.shape != (6,):
        raise ValueError("paired confusion moments must have length six")
    value_a, gradient_a = _f1_value_gradient(moments[:3])
    value_b, gradient_b = _f1_value_gradient(moments[3:])
    gradient = torch.cat((-gradient_a, gradient_b))
    return value_b - value_a, gradient


def _aggregate_targets(values, target_ids):
    if values.ndim != 2:
        raise ValueError("values must be a draw-by-feature matrix")
    target_ids = target_ids.to(device=values.device)
    if target_ids.ndim != 1 or target_ids.numel() != values.shape[0]:
        raise ValueError("target IDs must match the number of draws")
    unique_ids, inverse = torch.unique(
        target_ids, sorted=True, return_inverse=True)
    target_count = int(unique_ids.numel())
    counts = torch.bincount(inverse, minlength=target_count).to(torch.float64)
    sums = torch.zeros(
        (target_count, values.shape[1]),
        dtype=torch.float64,
        device=values.device,
    )
    sums.index_add_(0, inverse, values.to(torch.float64))
    return unique_ids, inverse, counts, sums / counts[:, None]


def paired_f1_target_summary(
        labels,
        predictions_a,
        predictions_b,
        target_ids,
):
    contributions = paired_contributions(
        labels, predictions_a, predictions_b)
    unique_ids, inverse, counts, target_means = _aggregate_targets(
        contributions, target_ids)
    moments = target_means.mean(dim=0)
    point, gradient = paired_f1_value_gradient(moments)
    target_influence = (target_means - moments) @ gradient
    draw_deviation = (
        contributions - target_means[inverse]
    ) @ gradient

    target_count = int(unique_ids.numel())
    within_sum_squares = torch.zeros(
        target_count, dtype=torch.float64, device=contributions.device)
    within_sum_squares.index_add_(0, inverse, draw_deviation.square())
    within_variance = torch.where(
        counts > 1.0,
        within_sum_squares / (counts - 1.0).clamp_min(1.0),
        torch.full_like(counts, float("nan")),
    )
    return {
        "point": float(point),
        "gradient": gradient,
        "moments": moments,
        "target_ids": unique_ids,
        "replicate_counts": counts,
        "target_means": target_means,
        "target_influence": target_influence,
        "draw_deviation": draw_deviation,
        "within_variance": within_variance,
        "draw_to_target": inverse,
    }


def dependency_adjacency(
        source,
        destination,
        timestamps=None,
        temporal_window=None,
):
    source = source.reshape(-1)
    destination = destination.reshape(-1)
    if source.numel() != destination.numel():
        raise ValueError("source and destination must have matching lengths")
    shared = (
        (source[:, None] == source[None, :])
        | (source[:, None] == destination[None, :])
        | (destination[:, None] == source[None, :])
        | (destination[:, None] == destination[None, :])
    )
    adjacency = shared
    if temporal_window is not None:
        if timestamps is None or timestamps.numel() != source.numel():
            raise ValueError("timestamps must match targets")
        temporal = (
            timestamps.reshape(-1, 1) - timestamps.reshape(1, -1)
        ).abs() <= float(temporal_window)
        adjacency = adjacency | temporal
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(False)
    return adjacency


def graph_diffusion_kernel(
        adjacency,
        timestamps=None,
        graph_order=2,
        graph_decay=0.35,
        time_scale=None,
):
    adjacency = adjacency.bool()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    count = adjacency.shape[0]
    device = adjacency.device
    dtype = torch.float64
    weights = adjacency.to(dtype)
    degree = weights.sum(dim=1).clamp_min(1.0)
    normalized = (
        weights / torch.sqrt(degree[:, None] * degree[None, :])
    )
    basis = torch.eye(count, dtype=dtype, device=device)
    propagation = basis.clone()
    for order in range(1, int(graph_order) + 1):
        propagation = propagation @ normalized
        basis = basis + float(graph_decay) ** order * propagation
    kernel = basis @ basis.T
    diagonal = torch.sqrt(
        torch.diagonal(kernel).clamp_min(torch.finfo(dtype).eps))
    kernel = kernel / (diagonal[:, None] * diagonal[None, :])

    if time_scale is not None:
        if timestamps is None or timestamps.numel() != count:
            raise ValueError("timestamps must match the adjacency")
        time_distance = (
            timestamps.reshape(-1, 1) - timestamps.reshape(1, -1)
        ).to(dtype)
        time_kernel = torch.exp(
            -0.5 * (time_distance / float(time_scale)).square())
        kernel = kernel * time_kernel
    return 0.5 * (kernel + kernel.T)


def _component_sizes(adjacency):
    adjacency = adjacency.bool().cpu()
    count = adjacency.shape[0]
    visited = [False] * count
    sizes = []
    for start in range(count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in torch.where(adjacency[current])[0].tolist():
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def dependency_diagnostics(adjacency, kernel):
    count = int(adjacency.shape[0])
    degree = adjacency.sum(dim=1).to(torch.float64)
    component_sizes = _component_sizes(adjacency)
    trace = float(torch.trace(kernel))
    squared_trace = float(kernel.square().sum())
    effective_count = (
        0.0 if squared_trace <= 0.0 else trace * trace / squared_trace
    )
    return {
        "target_count": count,
        "max_degree": int(degree.max()) if count else 0,
        "max_component_fraction": (
            0.0 if not component_sizes
            else max(component_sizes) / float(count)
        ),
        "effective_target_count": effective_count,
    }


def _variance_from_kernel(influence, kernel):
    count = int(influence.numel())
    if count < 2:
        return math.inf
    variance = float(influence @ kernel @ influence)
    variance *= count / (count - 1.0)
    return max(0.0, variance / (count * count))


def endpoint_dyadic_variance(influence, source, destination):
    influence = influence.to(torch.float64)
    source = source.to(influence.device)
    destination = destination.to(influence.device)
    if not (
        influence.numel() == source.numel() == destination.numel()
    ):
        raise ValueError("endpoint rows must match target influence")
    entities = torch.unique(torch.cat((source, destination)))
    meat = torch.zeros((), dtype=torch.float64, device=influence.device)
    memberships = torch.zeros_like(influence)
    for entity in entities:
        active = (source == entity) | (destination == entity)
        memberships += active.to(torch.float64)
        meat += influence[active].sum().square()
    meat -= ((memberships - 1.0).clamp_min(0.0) * influence.square()).sum()
    count = int(influence.numel())
    if count < 2:
        return math.inf
    return max(
        0.0,
        float(meat) * count / (count - 1.0) / (count * count),
    )


def _critical_value(alpha, alternative):
    if alternative == "two-sided":
        return NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    if alternative in ("greater", "less"):
        return NormalDist().inv_cdf(1.0 - float(alpha))
    raise ValueError("alternative must be two-sided, greater or less")


def _interval(point, variance, alpha=0.05):
    standard_error = math.sqrt(variance) if math.isfinite(variance) else math.inf
    critical = _critical_value(alpha, "two-sided")
    radius = critical * standard_error
    return {
        "point": float(point),
        "standard_error": standard_error,
        "lower": float(point) - radius,
        "upper": float(point) + radius,
    }


def allocation_plan(
        between_per_target,
        within_per_draw,
        target_cost=1.0,
        neighbor_cost=1.0,
        replicate_grid=(2, 4, 8, 16),
):
    a = max(0.0, float(between_per_target))
    b = max(0.0, float(within_per_draw))
    target_cost = float(target_cost)
    neighbor_cost = float(neighbor_cost)
    grid = tuple(sorted(set(int(value) for value in replicate_grid)))
    if (
        target_cost <= 0.0 or neighbor_cost <= 0.0
        or not grid or grid[0] < 1
    ):
        raise ValueError("costs and replicate grid must be positive")
    if a == 0.0 and b == 0.0:
        continuous = float(grid[0])
    elif a == 0.0:
        continuous = math.inf
    elif b == 0.0:
        continuous = 0.0
    else:
        continuous = math.sqrt(
            b * target_cost / (a * neighbor_cost))

    objective = {
        repeats: (
            a + b / repeats
        ) * (
            target_cost + repeats * neighbor_cost
        )
        for repeats in grid
    }
    selected = min(grid, key=lambda value: (objective[value], value))
    return {
        "between_per_target": a,
        "within_per_draw": b,
        "continuous_replicates": continuous,
        "selected_replicates": selected,
        "objective_by_replicates": objective,
    }


def gt_psf1_interval(
        labels,
        predictions_a,
        predictions_b,
        target_ids,
        source,
        destination,
        timestamps,
        adjacency,
        kernel,
        alpha=0.05,
        target_cost=1.0,
        neighbor_cost=1.0,
        replicate_grid=(2, 4, 8, 16),
        require_allocation=True,
):
    summary = paired_f1_target_summary(
        labels, predictions_a, predictions_b, target_ids)
    target_count = int(summary["target_ids"].numel())
    if not (
        source.numel() == destination.numel()
        == timestamps.numel() == target_count
    ):
        raise ValueError("one source, destination and timestamp per target")
    if kernel.shape != (target_count, target_count):
        raise ValueError("kernel must match unique targets")

    graph = dependency_diagnostics(adjacency, kernel)
    denominator_a = float(
        target_count
        * (2.0 * summary["moments"][0]
           + summary["moments"][1] + summary["moments"][2])
    )
    denominator_b = float(
        target_count
        * (2.0 * summary["moments"][3]
           + summary["moments"][4] + summary["moments"][5])
    )
    minimum_repeats = int(summary["replicate_counts"].min())
    reasons = []
    if graph["max_degree"] > math.sqrt(target_count):
        reasons.append("max_degree")
    if graph["max_component_fraction"] > 0.5:
        reasons.append("component_fraction")
    if graph["effective_target_count"] < 30.0:
        reasons.append("effective_target_count")
    if min(denominator_a, denominator_b) < 10.0:
        reasons.append("positive_contributions")
    if require_allocation and minimum_repeats < 2:
        reasons.append("sampler_replicates")

    total_variance = _variance_from_kernel(
        summary["target_influence"], kernel)
    valid_within = summary["within_variance"][
        torch.isfinite(summary["within_variance"])
    ]
    within_per_draw = (
        float(valid_within.mean()) if valid_within.numel() else math.nan
    )
    within_current = 0.0
    if valid_within.numel() == target_count:
        within_current = float(
            (
                summary["within_variance"]
                / summary["replicate_counts"]
            ).sum()
            / (target_count * target_count)
        )
    between_per_target = max(
        0.0, target_count * (total_variance - within_current))
    allocation = None
    if math.isfinite(within_per_draw):
        allocation = allocation_plan(
            between_per_target,
            within_per_draw,
            target_cost=target_cost,
            neighbor_cost=neighbor_cost,
            replicate_grid=replicate_grid,
        )
    result = _interval(summary["point"], total_variance, alpha=alpha)
    result.update({
        "status": STATUS_OUT_OF_SCOPE if reasons else STATUS_OK,
        "diagnostic_reasons": reasons,
        "graph_diagnostics": graph,
        "target_count": target_count,
        "draw_count": int(labels.numel()),
        "minimum_replicates": minimum_repeats,
        "total_variance": total_variance,
        "between_per_target": between_per_target,
        "within_per_draw": within_per_draw,
        "within_current_variance": within_current,
        "allocation": allocation,
        "moments": summary["moments"],
        "gradient": summary["gradient"],
    })
    return result


def comparator_intervals(
        labels,
        predictions_a,
        predictions_b,
        target_ids,
        source,
        destination,
        kernel,
        alpha=0.05,
):
    summary = paired_f1_target_summary(
        labels, predictions_a, predictions_b, target_ids)
    draw_contributions = paired_contributions(
        labels, predictions_a, predictions_b)
    pooled_moments = draw_contributions.mean(dim=0)
    row_point, row_gradient = paired_f1_value_gradient(pooled_moments)
    row_influence = (
        draw_contributions - pooled_moments
    ) @ row_gradient
    row_variance = (
        float(row_influence.var(unbiased=True)) / row_influence.numel()
        if row_influence.numel() > 1 else math.inf
    )
    target_influence = summary["target_influence"]
    target_variance = (
        float(target_influence.var(unbiased=True))
        / target_influence.numel()
        if target_influence.numel() > 1 else math.inf
    )
    dyadic_variance = endpoint_dyadic_variance(
        target_influence, source, destination)
    graph_variance = _variance_from_kernel(target_influence, kernel)
    return {
        "row_iid": _interval(float(row_point), row_variance, alpha),
        "target_iid": _interval(
            summary["point"], target_variance, alpha),
        "endpoint_dyadic": _interval(
            summary["point"], dyadic_variance, alpha),
        "graph_hac": _interval(
            summary["point"], graph_variance, alpha),
    }
