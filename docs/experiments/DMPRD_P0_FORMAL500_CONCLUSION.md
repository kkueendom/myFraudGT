# DMPRD P0 Six-Dataset Formal Conclusion

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Decision Date: 2026-07-18
- Verification Status: SIX-DATASET 500-EPOCH VERIFIED
- Branch: `feature/support-calibrated-prototype-residual`
- Formal Experiment Commit: `d6b3c0c`
- Queue Result: 6 complete / 0 failed
- Decision: `FAIL`

## Formal Decision

Reject the current P0 support- and uncertainty-calibrated prototype residual as
a replacement for A2. Retain A2 as the mainline decoder.

All six P0 tasks and their matched A2 references reached epoch 499. The
pre-registered formal screen fails:

- mean validation-selected test-F1 delta: `-0.01430`;
- validation-selected wins: `1/6`;
- mean raw-best delta: `-0.00795`;
- raw-best wins: `3/6`;
- raw regressions below `-0.010`: `2`.

The required thresholds were mean val delta above `+0.005`, at least `4/6` val
wins, non-negative mean raw delta, and at most one large raw regression. P0
misses every threshold.

## Complete Results

| Dataset | A2 val | P0 val | Delta | A2 raw | P0 raw | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.46247 | 0.45896 | -0.00351 | 0.50667 | 0.50881 | +0.00214 |
| Small-HI | 0.77984 | 0.77582 | -0.00402 | 0.79497 | 0.79216 | -0.00281 |
| Medium-LI | 0.51163 | 0.50242 | -0.00921 | 0.59031 | 0.55708 | -0.03323 |
| Medium-HI | 0.77574 | 0.77762 | +0.00188 | 0.78940 | 0.79424 | +0.00484 |
| Large-LI | 0.30108 | 0.27692 | -0.02416 | 0.44720 | 0.42708 | -0.02012 |
| Large-HI | 0.72897 | 0.68220 | -0.04677 | 0.76223 | 0.76370 | +0.00147 |
| **Mean delta** | | | **-0.01430** | | | **-0.00795** |

The failure is not only a validation-checkpoint problem. Raw-best also declines
on average, with material losses on Medium-LI and Large-LI.

## Why the 120-Epoch Result Did Not Generalize

The earlier quick screen used a scheduler whose full budget ended at epoch 120.
When the 500-epoch runs are truncated at matched points, the result is unstable:

| 500-schedule cutoff | Mean val delta | Mean raw delta | Val wins |
|---:|---:|---:|---:|
| 119 | +0.00019 | -0.00002 | 3/6 |
| 239 | -0.01323 | +0.00399 | 4/6 |
| 359 | +0.00058 | -0.00506 | 3/6 |
| 479 | -0.01430 | -0.00844 | 1/6 |
| 499 | -0.01430 | -0.00795 | 1/6 |

The independent 120-epoch schedule had reported mean val gain `+0.02180`.
Therefore, that gain was training-budget and scheduler dependent rather than a
stable architectural advantage.

## Mechanism Failure

### 1. Base-entropy gate collapse

At the end of formal training, mean base uncertainty is only `0.011-0.028`.
Because P0 multiplies local reliability, base uncertainty, and prototype
confidence, every dataset's mean gate collapses close to the `0.25` floor:

- Small-LI: `0.2529`;
- Small-HI: `0.2573`;
- Medium-LI: `0.2566`;
- Medium-HI: `0.2556`;
- Large-LI: `0.2593`;
- Large-HI: `0.2594`.

Long-trained classifiers can be confidently wrong. Entropy therefore is not a
reliable signal for whether prototype evidence should be allowed to correct a
prediction. The current gate closes precisely on many samples that may need
correction.

### 2. Global beta compensates for the closed gate

P0 learns a larger beta, but `beta * gate` remains only about `31%-35%` of A2's
effective prototype-residual scale, averaging approximately one third. Beta and
gate perform overlapping scaling roles, so one grows while the other shrinks.
This makes the claimed sample-level routing poorly identifiable.

### 3. Prototype residual saturation

Final P0 `delta_abs` is approximately `0.96-1.00` on every dataset. The tanh
residual is nearly saturated and no longer represents fine-grained differences
between samples. High local reliability (`0.87-0.96`) also becomes almost
constant after long training because cumulative support counts saturate.

## Supported Conclusion

The experiments support a narrower finding:

> Controlling the prototype residual can help under short training budgets, but
> multiplying cumulative prototype reliability by base predictive entropy does
> not produce a stable long-horizon decoder improvement.

Do not launch additional P0 seeds or present P0 as the paper mainline. A future
version must remove direct base-entropy multiplication, eliminate beta/gate
scale redundancy, and prevent prototype-residual saturation before another
formal screen.

## Reproducible Evidence

- Formal plan: `docs/experiments/DMPRD_P0_FORMAL500_PLAN.md`
- Machine-readable results: `docs/experiments/DMPRD_P0_FORMAL500_RESULTS.tsv`
- Audit: `run/dmprd_p0_formal_audit.py`
- P0 outputs: `results/dmprd_p0_formal500/`
- A2 references: `results/dmprd_formal500/`
