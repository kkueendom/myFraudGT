# ACDR-Help Method and Matched Pair-500 Plan

## Material Passport

- Mode: method design + pre-registered experiment plan
- Branch: `feature/acdr-marginal-help-critic`
- Parent revision: `3424e1e`
- Primary metric: Test F1 at the checkpoint selected by validation F1
- Status: **INTERIM FAILURE; paired screen stopped and not advancing**
- Outcome record: `docs/experiments/ACDR_HELP_INTERIM_FAILURE.md`
- Preserved output: `results/acdr_help_pair500_faf972865181/`

## 1. Motivation

P0 failed because its fixed multiplicative reliability rule collapsed near its
floor and reduced the useful A2 prototype residual. CAMPR V2 removed that
attenuation, but its target asked whether the residual helped when moving from
the base decoder to A2. The actual router decision is different: once the model
is already at A2, should the residual be reduced or increased?

CAMPR V2 also normalized both targets and routes with batch statistics. Its
inference result could therefore depend on unrelated samples in the same batch.
The generic router inherited GNN dropout, and its counterfactual auxiliary loss
was much larger than the classification loss early in training.

ACDR-Help addresses these mechanism failures while preserving the proven A2
prototype bank, bounded residual, readiness mask, and global beta.

## 2. A2 Reference

Let `z_base,i` be the original FraudGT decoder logits, `d_i` the bounded A2
prototype residual, `beta` its learned global scale, and `r_i` prototype
readiness. A2 predicts:

\[
z_i^{A2}=z_{base,i}+\beta r_i d_i.
\]

ACDR-Help does not replace this branch. It learns only a sample-level residual
dose around the A2 operating point.

## 3. Label-Free Marginal Help Critic

The inference critic is a dedicated network with no dropout:

```text
[detached h_i ; 10 label-free scalars]
  -> Linear(dim_in + 10, 32)
  -> GELU
  -> Linear(32, 1)
  -> sigmoid
```

The scalar inputs are raw base margin, absolute base margin, residual margin,
absolute residual margin, positive prototype similarity, negative prototype
similarity, prototype margin, absolute prototype margin, base-residual
alignment, and prototype readiness. No label, batch mean, batch extrema,
dataset identity, or dataset size enters inference.

The final linear layer is zero-initialized. Therefore the initial critic output
is exactly `q_i=0.5`. The residual dose is

\[
a_i=1+\rho(2q_i-1), \qquad \rho=1,
\]

and the final logits are

\[
z_i=z_{base,i}+\beta r_i a_i d_i.
\]

At `q_i=0.5`, `a_i=1` and the method is numerically identical to A2. Unlike
P0, a collapsed neutral critic cannot silently shrink the complete prototype
branch. Unlike CAMPR, one sample's dose is invariant to batch composition.
With the matched hidden width `dim_in=64`, the critic adds 2,433 parameters.

## 4. A2-Centered Counterfactual Supervision

Training compares two detached residual doses placed symmetrically around A2:

\[
z_i^- = z_{base,i}+0.75\beta r_i d_i,
\qquad
z_i^+ = z_{base,i}+1.25\beta r_i d_i.
\]

The marginal advantage and hard target are

\[
\Delta_i=\ell(z_i^-,y_i)-\ell(z_i^+,y_i),
\qquad
t_i=\mathbb{1}[\Delta_i>0].
\]

Thus `t_i=1` means a stronger-than-A2 residual lowers current-label loss, and
`t_i=0` means the residual should be reduced. Samples with
`|Delta_i| < 1e-4`, or without ready prototypes, are ignored because their
ordering is not reliable. Help and harm examples receive inverse-frequency
weights so each observed target group contributes equally.

The auxiliary objective is

\[
\mathcal L_{help}=0.05\,
\operatorname{BalancedBCE}(s_i,t_i),
\]

and is active only during epochs 10 through 150. The classification loss still
trains the dose end to end throughout training. Labels are used only in the
detached auxiliary target; they are absent from the inference critic. Prototype
banks are queried before, and updated after, current-batch prediction.

## 5. Claimed Innovation Relative to A2, P0, and CAMPR

1. A2-centered marginal supervision learns the local dose direction at the
   actual deployment point instead of asking whether moving from base to A2 was
   useful.
2. A sample-local dose function removes CAMPR's batch-composition dependence.
3. Neutral zero initialization gives an exact A2 fallback invariant.
4. Dead-zone filtering and help/harm balancing prevent tiny loss differences
   or a dominant target group from forcing a constant critic.
5. The global beta controls overall prototype strength, while the critic learns
   interpretable sample-level dose allocation.

These points form a coherent method contribution, but novelty alone is not an
empirical claim. The method advances only if the matched pair screen succeeds.

## 6. Mechanism Diagnostics

Low-frequency training logs report:

- help ratio among effective samples;
- effective target coverage;
- dose mean, standard deviation, minimum, maximum, and endpoint fractions;
- correlation between dose and the hard help target;
- marginal advantage mean and standard deviation;
- prototype readiness.

Reject the mechanism early for a persistent effective coverage below `5%`,
dose standard deviation below `0.01`, correlation near zero after the auxiliary
window has started, more than `30%` of samples at an endpoint, non-finite
values, or any batch-independence/leakage failure.

## 7. Matched 500-Epoch Pair Screen

| Dataset | Seed | Matched A2 Test F1 at val-selected checkpoint |
|---|---:|---:|
| Small-LI | 42 | 0.46247 |
| Large-LI | 44 | 0.30108 |

Both jobs use the same dataset split, seed, model backbone, A2 prototype
settings, optimizer, class weights, batch size, and 500-epoch cosine scheduler.
Early stopping and auto-resume are disabled. Interim epoch limits inspect the
same 500-epoch trajectory; they are not separate short-scheduler experiments.

The code revision is embedded in the output directory and run name. The queue
refuses to launch from a dirty worktree or a different branch. It excludes GPU0
and considers OCR processes non-blocking only when the configured free-memory
and utilization thresholds still pass.

Final advancement requires both datasets to improve validation-selected Test
F1 by at least `+0.005`:

- Small-LI: at least `0.46747`;
- Large-LI: at least `0.30608`.

Raw-best Test F1 is reported only as a stability diagnostic. It is not used to
select checkpoints or tune the method. Failure on either dataset rejects this
version and prevents expansion to Medium-HI/Large-HI.

## 8. Historical Commands (Do Not Resume)

These commands are retained for reproducibility only. The paired screen has
been rejected; do not restart the queue or resume either incomplete task.

CPU invariants:

```bash
/d/miniconda3/envs/fraudGT/bin/python3.9 run/acdr_help_invariant_test.py
```

Historical pair-queue invocation:

```bash
nohup /d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/acdr_help_pair_queue.py \
  > .acdr_help_pair_queue.nohup.log 2>&1 &
```

Matched audit:

```bash
/d/miniconda3/envs/fraudGT/bin/python3.9 run/acdr_help_pair_audit.py
/d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/acdr_help_pair_audit.py --epoch-limit 119
```

The audit defaults to the established A2 results at
`/e/yky/FraudGT_dmprd_quickcheck/results/dmprd_formal500` and accepts `--a2`
or `A2_RESULTS_ROOT` for relocation.

## 9. Interim Outcome and Next Direction

The paired screen was rejected before 500 epochs. At the epoch-165 matched
audit, Small-LI regressed by `-0.01660` on validation-selected Test F1 and
`-0.01594` on raw-best, while Large-LI improved by `+0.03971` on both metrics.
The latest validation diagnostics showed mean doses of `1.9984` on Small-LI
and `1.9931` on Large-LI, with mean help probabilities above `0.996`.

This means the A2-centered target recovered an absolute preference for stronger
prototype residuals, but the binary critic became an almost constant global
amplifier. Its function is therefore redundant with A2's global `beta`, and the
claimed sample-level routing contribution is unsupported. Large-LI was stopped
around epoch 171 and Small-LI around epoch 268; the incomplete trajectories are
preserved under `results/acdr_help_pair500_faf972865181/` and must not be
resumed as a formal 500-epoch experiment.

The next COSTAR direction is motivated by this exact failure. It should keep
global residual scale under `beta` and constrain the learned sample correction
to be orthogonal to the global-scale direction. A collapsed sample router must
then contribute zero and recover A2, rather than expressing another uniform
increase in residual magnitude. See `ACDR_HELP_INTERIM_FAILURE.md` for the full
evidence and interpretation.
