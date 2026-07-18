# CAMPR Method and Representative Experiment Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: method design + pre-registered experiment plan
- Plan Date: 2026-07-18
- Verification Status: V2 500-EPOCH SMALL-LI + LARGE-LI HARD GATE RUNNING
- Branch: `feature/campr-counterfactual-prototype-router`
- Parent: P0 conclusion commit `55f5972`

## 1. Why P0 Was Rejected

The six matched 500-epoch P0 runs completed without runtime failures, but P0
lost to A2 on the primary validation-selected test F1:

- mean delta: `-0.01430`;
- wins: `1/6`;
- mean raw-best delta: `-0.00795`.

The failure was mechanistic. Base entropy fell to `0.011-0.028`, forcing the
multiplicative P0 gate to its `0.25` floor. Cumulative support reliability and
the bounded prototype residual also saturated. P0 therefore stopped providing
meaningful sample-level routing and reduced the effective A2 residual scale.

## 2. Method Hypothesis

The useful question is not whether the base classifier is uncertain or whether
a prototype has been observed many times. It is whether applying the prototype
residual would improve the current sample. CAMPR learns this decision from a
training-only counterfactual target while keeping inference label-free.

CAMPR retains A2's four prototypes per class, bounded prototype residual, and
global residual scale. It removes P0's entropy and support-count multiplication.

## 3. Counterfactual Advantage Supervision

For training sample `i`, let `d_i` be the A2 bounded prototype residual and
`beta` its learned global scale. The benefit of the un-routed A2 correction is:

\[
a_i = \ell(z_{base,i}, y_i)
      - \ell(z_{base,i}+\beta d_i, y_i).
\]

Positive `a_i` means that the A2 residual lowers classification loss. Because
the magnitude changes with training stage, beta, and dataset, CAMPR centers and
normalizes it using the class-weighted batch mean and mean absolute deviation:

\[
\hat a_i =
\frac{a_i-\mathbb{E}_w[a]}{
\mathbb{E}_w[|a-\mathbb{E}_w[a]|]+\epsilon},
\qquad
t_i = \sigma(\hat a_i/\tau), \qquad
\mathcal{L}_{route}=\operatorname{BCE}(q_i,t_i).
\]

This relative target matches mean-preserving inference: the router learns which
samples deserve more of a fixed residual budget, while beta learns whether the
residual should be globally strong or weak. It is also invariant to the raw
loss-difference scale that caused the first CAMPR pilot target to collapse.

Labels are used only to construct this detached training target. The inference
router receives ten label-free signals derived from base confidence, positive
and negative prototype similarities, prototype margin, residual direction and
magnitude, readiness, and base-residual alignment.

## 4. Mean-Preserving Routing

Let `q_i` be the router probability. CAMPR first divides it by the batch mean,
centers the relative scores, and applies one shared scale so every multiplier
stays in `[0.5, 1.5]` while preserving a mean of exactly one:

\[
\frac{1}{B}\sum_i \tilde g_i=1,
\qquad 0.5\leq\tilde g_i\leq1.5.
\]

The final prediction is:

\[
z_i=z_{base,i}+\beta\,r_i\,\tilde g_i\,d_i,
\]

where `r_i` is A2 prototype readiness. The router reallocates a fixed residual
budget across samples; `beta` alone controls the global residual strength.

If the router is constant, every `g_i=1` and CAMPR becomes A2 exactly. This
fallback invariant prevents the P0 failure mode in which a collapsed gate
silently shrinks the whole prototype branch.

## 5. Novelty Relative to A2 and P0

1. Counterfactual benefit supervision replaces heuristic uncertainty gating.
2. Mean-preserving routing separates global residual magnitude from per-sample
   allocation.
3. Exact constant-router fallback preserves A2 rather than attenuating it.
4. Training labels supervise routing without entering inference logits.
5. The implementation adds only a ten-input MLP (121 parameters in the smoke
   configuration), keeping the method attributable and inexpensive.

## 6. Leakage and Reproducibility Controls

- Router inputs are detached, label-free statistics available at inference.
- Counterfactual targets and losses are computed only during training.
- Prototype banks are queried before being updated with the current batch.
- The matched A2 references use the same seeds, datasets, parent architecture,
  500-epoch budget, and scheduler.
- Validation-selected test F1 is primary; raw-best is diagnostic only.

## 7. Pilot Failure and V2 Repair

The first five-task CAMPR pilot was stopped at epochs `8-17`. All processes
were healthy, but the fixed absolute temperature `0.10` produced target
standard deviations of only `0.0004-0.0008` on the three full runs. The
counterfactual auxiliary objective was therefore effectively constant.

V2 uses the centered, scale-normalized target above and changes the
dimensionless temperature to `1.0`. Pilot outputs remain preserved under
`results/campr_formal500/`; V2 writes to `results/campr_formal500_v2/` so the
failed pilot cannot be mistaken for formal evidence.

## 8. Smoke Verification

The V2 one-epoch Small-LI smoke test completed on GPU2:

- output: `results/campr_smoke_v4/AML-Small-LI-CAMPRSmokeV4-gpu0/42`;
- parameters: `281598` (121 more than A2/P0);
- test route mean: `1.0000`;
- observed test route range: approximately `[0.9437, 1.0236]`;
- batch 1 had `ready=0` and the expected zero counterfactual advantage;
- batch 2 had `ready=0.25`, advantage standard deviation `0.00056`, and
  normalized target standard deviation `0.2375` (versus approximately
  `0.0014` before V2), confirming that the repaired auxiliary target is active.

An independent invariant test also verified route mean `1.00000000`, strict
configured bounds, and zero numerical error for constant-router A2 fallback.

## 9. Representative 500-Epoch Screen

Full CAMPR runs:

| Dataset | Seed | Reason |
|---|---:|---|
| Large-LI | 44 | P0's largest LI-scale regression |
| Medium-HI | 42 | The only P0 validation-selected win |
| Small-LI | 42 | Fast small-scale representative |

Counterfactual auxiliary ablations (`campr_aux_weight=0`):

| Dataset | Seed |
|---|---:|
| Large-LI | 44 |
| Small-LI | 42 |

All five jobs use `optim.max_epoch=500` and the original 500-epoch cosine
scheduler. Epoch 119/239/359 audits are diagnostic only and do not terminate a
healthy run.

## 10. Pre-Registered Decision Rule

CAMPR advances to all six datasets only if the three full runs satisfy all:

1. mean validation-selected test-F1 delta over A2 is greater than `+0.005`;
2. validation-selected wins are at least `2/3`;
3. mean raw-best delta is non-negative;
4. at most one raw-best regression is below `-0.010`.

The counterfactual objective is supported if the two full-versus-no-aux pairs
have positive mean validation-selected delta and full CAMPR wins at least one.
This ablation is diagnostic and does not override the main A2 comparison.

If the main screen fails, do not launch six-dataset or multi-seed expansion.
Inspect advantage-target variance, route variance, saturation, and per-dataset
deltas before changing one mechanism at a time.

The V2 epoch-119 interim audit showed a material Small-LI regression despite a
Large-LI win. This is a high-risk signal, not a formal rejection: the revised
funnel requires complete 500-epoch results on both datasets. The two full tasks
were resumed from scheduler-preserving checkpoints. See
`CAMPR_V2_EPOCH119_INTERIM_AUDIT.md` for details.

## 11. Commands

```bash
nohup python3 run/campr_formal_queue.py \
  > .campr_formal500_v2_queue.nohup.log 2>&1 &

python3 run/campr_formal_audit.py
python3 run/campr_formal_audit.py --epoch-limit 119
```
