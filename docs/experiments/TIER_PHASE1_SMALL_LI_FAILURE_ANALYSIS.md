# TIER Phase 1 Small-LI Failure Analysis

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: analysis
- Verification Status: EXECUTED on remote commit `204b3075`
- Dataset / Seed: AML Small-LI / 42
- Sampling Protocol: `dynamic_random`
- Historical headline baseline: initial A2 only
- Fixed-panel A2: excluded

## Results

| Evidence selection | Val-selected Test F1 | Delta vs initial A2 | Raw-best Test F1 | Delta vs initial A2 | Decision |
|---|---:|---:|---:|---:|---|
| recent | 0.30664 | -0.15583 | 0.35363 | -0.15304 | fail |
| role_motif | 0.31499 | -0.14748 | 0.35294 | -0.15373 | fail |

The comparable initial A2 values are `0.46247` for Val-selected Test F1 and
`0.50667` for Raw-best Test F1. The two TIER rows are independent
evidence-only classifiers, not A2-plus-evidence fusion models.

## What Worked

Both selections establish that the raw evidence representation is nontrivial:

| Evidence selection | Normal F1 | Normal - shuffled | Normal - off | AUPRC | Overall coverage | Positive coverage |
|---|---:|---:|---:|---:|---:|---:|
| recent | 0.30664 | 0.30316 | 0.30664 | 0.23071 | 0.99999 | 1.00000 |
| role_motif | 0.31499 | 0.31108 | 0.31499 | 0.23185 | 0.99999 | 1.00000 |

The nearly complete collapse under shuffled evidence rules out a target-only
shortcut in this evidence-only model. The off condition predicts no positives
at the selected validation threshold. Role/motif prioritization improves
Val-selected F1 by `0.00835` over recent selection, so it is the stronger of
the two tested context selectors on Small-LI.

## Why It Failed

The paired diagnostic uses a frozen terminal A2 checkpoint on fresh dynamic
batches only to compare prediction changes. It is not a substitute for the
historical initial-A2 headline baseline.

| Evidence selection | Paired A2 errors | Corrected | Broken | Corrected / broken | Correction rate | Changed | Paired retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| recent | 271 | 40 | 121 | 0.331 | 0.148 | 161 | 0.9912 |
| role_motif | 276 | 36 | 81 | 0.444 | 0.130 | 117 | 0.9911 |

Both selectors satisfy the following evidence conditions:

- normal minus shuffled is above `0.010`;
- normal minus off is above `0.005`;
- more than 10% of sampled A2 errors are corrected;
- changed predictions exceed 50;
- coverage is not confined to a tiny positive subset.

Both fail the safety condition `corrected / broken >= 1.5`. They act on real
evidence, but an independent thresholded evidence classifier replaces too many
correct A2 decisions with mistakes. This is a routing/decision problem, not a
failure of evidence availability.

## Consequence For The Paper Line

Do not claim that evidence-only TIER improves FraudGT. It does not.

The defensible provisional statement is narrower:

> Raw transaction-centered higher-order evidence is predictive and
> counterfactually necessary for its own decisions, but its standalone
> decision boundary is not sufficiently safe to replace FraudGT.

This is exactly the failure mode a utility-supervised error router must solve:
it should learn to expose evidence only where the evidence is likely to correct
an A2 error, rather than applying the evidence classifier's global threshold.
However, the preregistered Phase 1 gate requires a qualified selection on both
Small-LI and Large-LI before implementing CrossFusion and the router. The
Small-LI result alone is insufficient to advance.

## Decision Rule

1. Wait for both Large-LI Phase 1a manifests on the same commit and protocol.
2. Run `run/tier_phase1_result_audit.py` on all four manifests.
3. If no selection qualifies on both scales, stop this EvidenceEncoder
   configuration. Analyze the error subsets before changing architecture.
4. If role_motif qualifies on Large-LI but not Small-LI, report the scale
   asymmetry and run evidence-family ablations before any fusion model.
5. Only if the preregistered cross-scale gate passes, implement representation
   CrossFusion and an out-of-fold, train-only utility target for ErrorRouter.

The next model must not reintroduce a generic logit gate, residual coefficient,
prototype correction, or a validation/test-label-derived router target.
