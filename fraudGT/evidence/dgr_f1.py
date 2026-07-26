"""Dynamic graph replicability certification for paired F1 streams."""

import math
from statistics import NormalDist

import numpy as np
from scipy.stats import beta, t

from fraudGT.evidence.gtf1c import f1_from_counts
from fraudGT.evidence.trefic import exact_f1_sign_utility


DECISION_IMPROVEMENT = "REPLICABLE_IMPROVEMENT"
DECISION_HARM = "REPLICABLE_HARM"
DECISION_INSUFFICIENT = "INSUFFICIENT_INFORMATION"


def f1_intervention_delta(
        tp,
        fp,
        fn,
        add_corrected,
        add_broken,
        remove_corrected,
        remove_broken,
):
    """Return paired F1 delta and its exact directional sign utility."""
    base_f1 = f1_from_counts(tp, fp, fn)
    routed_tp = int(tp) + int(add_corrected) - int(remove_broken)
    routed_fp = int(fp) + int(add_broken) - int(remove_corrected)
    routed_fn = int(fn) - int(add_corrected) + int(remove_broken)
    if min(routed_tp, routed_fp, routed_fn) < 0:
        raise ValueError("intervention counts are infeasible")
    routed_f1 = f1_from_counts(routed_tp, routed_fp, routed_fn)
    denominator = int(tp) + int(fp) + int(fn)
    rho = int(tp) / denominator if denominator > 0 else 0.0
    utility = exact_f1_sign_utility(
        add_corrected,
        add_broken,
        remove_corrected,
        remove_broken,
        rho,
    )
    return {
        "base_f1": base_f1,
        "routed_f1": routed_f1,
        "delta": routed_f1 - base_f1,
        "rho": rho,
        "utility": utility,
    }


def studentized_mean_interval(values, delta):
    """Two-sided Studentized interval for an independent stream mean."""
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    mean = float(values.mean()) if count else 0.0
    if count < 2:
        return {
            "mean": mean,
            "standard_error": math.inf,
            "lower": -math.inf,
            "upper": math.inf,
            "count": count,
        }
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    critical = float(t.ppf(1.0 - float(delta) / 2.0, count - 1))
    radius = critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - radius,
        "upper": mean + radius,
        "count": count,
    }


def clopper_pearson_interval(successes, trials, delta):
    """Two-sided exact interval for an independent Bernoulli rate."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return {
            "rate": 0.0,
            "lower": 0.0,
            "upper": 1.0,
            "successes": successes,
            "trials": trials,
        }
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    tail = float(delta) / 2.0
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(
            1.0 - tail, successes + 1, trials - successes))
    )
    return {
        "rate": successes / trials,
        "lower": lower,
        "upper": upper,
        "successes": successes,
        "trials": trials,
    }


def _checkpoint_decision(mean_interval, positive, harmful, pi0):
    if (
        mean_interval["lower"] > 0.0
        and positive["lower"] >= float(pi0)
    ):
        return DECISION_IMPROVEMENT
    if (
        mean_interval["upper"] < 0.0
        and harmful["lower"] >= float(pi0)
    ):
        return DECISION_HARM
    return DECISION_INSUFFICIENT


def dgr_f1_trajectory(
        stream_deltas,
        checkpoints=(8, 16, 32, 64),
        epsilon=0.005,
        pi0=0.75,
        delta=0.05,
):
    """Evaluate the preregistered joint decision at fixed checkpoints."""
    stream_deltas = np.asarray(stream_deltas, dtype=np.float64)
    checkpoints = tuple(int(value) for value in checkpoints)
    if not checkpoints or any(value <= 1 for value in checkpoints):
        raise ValueError("checkpoints must contain stream counts above one")
    if sorted(set(checkpoints)) != list(checkpoints):
        raise ValueError("checkpoints must be unique and increasing")
    if stream_deltas.size < checkpoints[-1]:
        raise ValueError("not enough streams for the final checkpoint")

    endpoint_delta = float(delta) / (len(checkpoints) * 3)
    trajectory = []
    stopped = False
    stopping_checkpoint = None
    final_decision = DECISION_INSUFFICIENT
    for checkpoint in checkpoints:
        values = stream_deltas[:checkpoint]
        mean_interval = studentized_mean_interval(
            values, endpoint_delta)
        positive = clopper_pearson_interval(
            int((values >= float(epsilon)).sum()),
            checkpoint,
            endpoint_delta,
        )
        harmful = clopper_pearson_interval(
            int((values <= -float(epsilon)).sum()),
            checkpoint,
            endpoint_delta,
        )
        decision = _checkpoint_decision(
            mean_interval, positive, harmful, pi0)
        if not stopped and decision != DECISION_INSUFFICIENT:
            stopped = True
            stopping_checkpoint = checkpoint
            final_decision = decision
        trajectory.append({
            "checkpoint": checkpoint,
            "mean": mean_interval,
            "practical_improvement": positive,
            "practical_harm": harmful,
            "decision": decision,
        })
    return {
        "decision": final_decision,
        "stopping_checkpoint": stopping_checkpoint,
        "endpoint_delta": endpoint_delta,
        "trajectory": trajectory,
    }


def mean_only_trajectory(
        stream_deltas,
        checkpoints=(8, 16, 32, 64),
        delta=0.05,
):
    """Group-sequential decision using only the paired stream mean."""
    values = np.asarray(stream_deltas, dtype=np.float64)
    checkpoints = tuple(int(value) for value in checkpoints)
    endpoint_delta = float(delta) / len(checkpoints)
    for checkpoint in checkpoints:
        interval = studentized_mean_interval(
            values[:checkpoint], endpoint_delta)
        if interval["lower"] > 0.0:
            return {
                "decision": DECISION_IMPROVEMENT,
                "stopping_checkpoint": checkpoint,
            }
        if interval["upper"] < 0.0:
            return {
                "decision": DECISION_HARM,
                "stopping_checkpoint": checkpoint,
            }
    return {
        "decision": DECISION_INSUFFICIENT,
        "stopping_checkpoint": None,
    }


def replication_only_trajectory(
        stream_deltas,
        checkpoints=(8, 16, 32, 64),
        epsilon=0.005,
        pi0=0.75,
        delta=0.05,
):
    """Group-sequential decision using only practical replication."""
    values = np.asarray(stream_deltas, dtype=np.float64)
    checkpoints = tuple(int(value) for value in checkpoints)
    endpoint_delta = float(delta) / (len(checkpoints) * 2)
    for checkpoint in checkpoints:
        current = values[:checkpoint]
        positive = clopper_pearson_interval(
            int((current >= float(epsilon)).sum()),
            checkpoint,
            endpoint_delta,
        )
        harmful = clopper_pearson_interval(
            int((current <= -float(epsilon)).sum()),
            checkpoint,
            endpoint_delta,
        )
        if positive["lower"] >= float(pi0):
            return {
                "decision": DECISION_IMPROVEMENT,
                "stopping_checkpoint": checkpoint,
            }
        if harmful["lower"] >= float(pi0):
            return {
                "decision": DECISION_HARM,
                "stopping_checkpoint": checkpoint,
            }
    return {
        "decision": DECISION_INSUFFICIENT,
        "stopping_checkpoint": None,
    }


def one_stream_decision(delta, epsilon=0.005):
    if float(delta) >= float(epsilon):
        return DECISION_IMPROVEMENT
    if float(delta) <= -float(epsilon):
        return DECISION_HARM
    return DECISION_INSUFFICIENT


def iid_paired_f1_interval(
        tp,
        fp,
        fn,
        tn,
        add_corrected,
        add_broken,
        remove_corrected,
        remove_broken,
        delta=0.05,
):
    """Row-IID delta-method interval from the eight paired categories."""
    counts = np.asarray([
        int(tp) - int(remove_broken),
        int(remove_broken),
        int(fp) - int(remove_corrected),
        int(remove_corrected),
        int(fn) - int(add_corrected),
        int(add_corrected),
        int(tn) - int(add_broken),
        int(add_broken),
    ], dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("intervention counts are infeasible")
    total = float(counts.sum())
    if total <= 1:
        return {
            "point": 0.0,
            "standard_error": math.inf,
            "lower": -math.inf,
            "upper": math.inf,
        }

    base_rows = np.asarray([
        [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1],
        [0, 0, 0], [0, 0, 0],
    ], dtype=np.float64)
    routed_rows = np.asarray([
        [1, 0, 0], [0, 0, 1],
        [0, 1, 0], [0, 0, 0],
        [0, 0, 1], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
    ], dtype=np.float64)
    probabilities = counts / total
    base_mean = probabilities @ base_rows
    routed_mean = probabilities @ routed_rows

    def value_gradient(vector):
        tp_value, fp_value, fn_value = vector
        denominator = 2.0 * tp_value + fp_value + fn_value
        if denominator <= 0.0:
            return 0.0, np.zeros(3, dtype=np.float64)
        value = 2.0 * tp_value / denominator
        gradient = np.asarray([
            2.0 * (fp_value + fn_value) / denominator ** 2,
            -2.0 * tp_value / denominator ** 2,
            -2.0 * tp_value / denominator ** 2,
        ])
        return value, gradient

    base_f1, base_gradient = value_gradient(base_mean)
    routed_f1, routed_gradient = value_gradient(routed_mean)
    influence = (
        (routed_rows - routed_mean) @ routed_gradient
        - (base_rows - base_mean) @ base_gradient
    )
    variance = float(probabilities @ influence ** 2)
    standard_error = math.sqrt(max(variance, 0.0) / total)
    critical = NormalDist().inv_cdf(1.0 - float(delta) / 2.0)
    point = routed_f1 - base_f1
    radius = critical * standard_error
    return {
        "point": point,
        "standard_error": standard_error,
        "lower": point - radius,
        "upper": point + radius,
    }
