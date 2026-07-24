# TIER Phase 1c High-Precision Intervention Separability

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: preregistered mechanism diagnosis
- Branch: `feature/tier-independent-evidence-routing`
- Parent evidence commit: `c89dc52`
- Sampling protocol: `dynamic_random`
- Primary endpoint: test corrected-to-broken ratio under a validation-selected policy
- Fixed-panel A2: excluded

## Research Question

Phase 1 and Phase 1b showed that independent raw-graph evidence is informative,
but replacing A2 predictions globally breaks too many correct A2 predictions.
Phase 1c asks a narrower question:

> Can observable confidence, disagreement, and support signals identify a
> high-precision subset on which the frozen evidence model should override the
> frozen A2 model?

This phase does not train CrossFusion or ErrorRouter. It tests whether the
routing problem is separable before adding another learned module.

## Frozen Inputs

Each task reuses:

- the final support-masked Phase 1b evidence checkpoint;
- the original 500-epoch A2 checkpoint;
- the same evidence family and context-selection policy;
- the original dynamic-random validation and test loaders.

No backbone, evidence encoder, threshold, or router parameter is updated.

## Candidate Intervention Policies

An intervention is possible only when A2 and the evidence classifier disagree.
Candidate policies combine four observable conditions:

1. disagreement direction: either direction, evidence-positive only, or
   evidence-negative only;
2. maximum A2 confidence margin from its validation-calibrated threshold;
3. minimum evidence confidence margin from its validation-calibrated threshold;
4. minimum context support count.

All numerical cutoffs are quantiles computed from validation disagreements.
The registered grid is:

- A2 maximum-margin quantiles: `0.1, 0.2, ..., 1.0`;
- evidence minimum-margin quantiles: `0.0, 0.1, ..., 0.9`;
- support-count minimum quantiles: `0.0, 0.25, 0.5, 0.75`;
- three disagreement directions.

The resulting 1,200 policies are evaluated on validation data only. A policy is
validation-eligible when it satisfies all original safety requirements:

- correction rate on A2 errors at least `10%`;
- corrected-to-broken ratio at least `1.5`;
- corrected minus broken greater than zero;
- changed predictions at least `max(50, 10% of A2 errors)`.

Among eligible policies, select the one with the largest corrected-minus-broken
count, breaking ties by corrected-to-broken ratio, correction count, and changed
count. If no policy is eligible, retain the highest-net-utility policy for
diagnosis but mark the task as validation failure.

## Locked Test Evaluation

The selected validation cutoffs are applied once to the independently sampled
test stream. Phase 1c passes only if the locked test policy also satisfies every
safety requirement and improves paired same-batch F1 over frozen A2.

The paired-subset F1 is a mechanism diagnostic. It is reported with the initial
A2 reference and delta for experiment bookkeeping, but it must not replace the
full-model six-dataset headline table.

## Matrix and Compute

The matrix contains 12 tasks:

- datasets: AML Small-LI and AML Large-LI;
- families: `structure`, `temporal`, and `flow_role`;
- context selection: `recent` and `role_motif`;
- one registered seed per dataset: 42 for Small-LI and 44 for Large-LI.

Tasks are independent and may occupy all available GPUs. The launcher waits for
the corresponding Phase 1b manifest before reading a checkpoint, so an
unfinished parent task cannot be evaluated accidentally.

## Decision Rule

| Outcome | Interpretation | Next action |
|---|---|---|
| Same family passes Small-LI and Large-LI | Routing signal is cross-scale separable | Implement train-only OOF ErrorRouter, then CrossFusion |
| Passes one scale only | Scale-dependent routing | Analyze scale-conditioned support; no universal claim |
| Validation passes but test fails | Policy overfit or sampling instability | Do not train a router; simplify the policy family |
| No validation policy passes | Observable routing signal is inadequate | Revise evidence representation or stop TIER |

No threshold may be relaxed after viewing test results.
