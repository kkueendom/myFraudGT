# CPTR: Conservative Prototype Threshold-Transfer Residual

## Contract

- Decoder: `cptr`
- Branch: `feature/cptr-threshold-transfer`
- Base commit: COSTAR dynamic-random `4abe58c`
- Screen: Small-LI seed 42 and Large-LI seed 44, 500 epochs
- Primary metric: validation-selected Test F1
- Baseline: the initial A2 same-metric table in
  `run/dynamic_random_a2_baseline.json`

CPTR does not change FraudGT's sampler. Train, validation, and test retain the
original dynamically sampled `LinkNeighborLoader(shuffle=True)` path.
`val.fixed_target_panel=False`; no fixed target panel, independent evaluation
generator, or sampler RNG restore is introduced.

## A2 anchor

Let the independently trained A2 margin be

\[
m_i^{A2}=m_i^{base}+e_i.
\]

The original weighted-cross-entropy loss is computed only from `z_A2`. All
features entering the CPTR adapter are detached, and adapter parameters are
gradient-clipped separately. Thus CPTR cannot change the encoder, base decoder,
prototype head, or A2 residual coefficient through its auxiliary objective.

## Boundary-localized correction

The adapter predicts a bounded candidate margin correction

\[
d_i=0.10\tanh f_\phi(\operatorname{stopgrad}(\xi_i)).
\]

For a threshold `t`, define

\[
u_i=\sigma((m_i^{A2}-t)/T_b),\qquad b_i=4u_i(1-u_i).
\]

`b_i` equals one at the threshold and approaches zero away from it. The
candidate therefore changes only decisions near the A2 operating boundary:

\[
\widetilde m_i=m_i^{A2}+b_i d_i.
\]

## Class-stratified bidirectional threshold transfer

Each training batch is split within each class after sorting by global target
edge id. Alternating items form halves A and B, making the split independent of
sampler output order. Both classes must have at least two examples; otherwise
the adapter loss is exactly zero and no deployment buffer is updated.

1. Select the hard-F1 threshold `t_A` from detached A2 margins and labels in A;
   optimize soft-F1 of the boundary-localized candidate on B at `t_A`.
2. Select `t_B` from B; optimize the same objective on A at `t_B`.
3. Compare candidate and A2 soft-F1 at the transferred threshold and penalize
   negative uplift.

Only current **training** batch labels enter this objective. Validation/test
labels never enter threshold selection, adapter gradients, or deployment
buffers. The two calibration thresholds update a checkpointed training EMA
only after the current batch prediction has been formed.

## Conservative temporal deployment gate

Chronological AML edge ids are divided by checkpointed training-index terciles
into three temporal environments. For each valid transferred evaluation half,
CPTR computes detached candidate-minus-A2 soft-F1 uplift separately by
environment and maintains checkpointed training-only EMA mean, EMA variance,
and count.

For environment `e`,

\[
LCB_e=\mu_e-z\sqrt{v_e/n_e},\qquad
g_e=\min(1,[LCB_e]_+/s_g).
\]

The deployed prediction is

\[
m_i^{CPTR}=m_i^{A2}+g_{e(i)}b_i d_i.
\]

If the training threshold count or environment count is below its minimum, or
if `LCB_e<=0`, `g_e` is exactly zero and the output is bitwise the detached A2
logit tensor after margin reconstruction. The default correction bound is
`0.10`.

Inference uses only per-sample detached features, global edge id, adapter
parameters, and checkpointed train buffers. It reads no current-batch labels,
means, quantiles, thresholds, ordering, or partition information. Reordering
or partitioning the same samples therefore preserves logits.

## Pair-500 decision rule

| Dataset | Initial A2 selected | Initial A2 raw-best | Stable-win target |
|---|---:|---:|---:|
| Small-LI | 0.46247 | 0.50667 | selected >= 0.46747 |
| Large-LI | 0.30108 | 0.44720 | selected >= 0.30608 |

Metrics are compared only to the matching initial A2 column. Improvements
below `0.005` are marked as possible dynamic-sampling variation. No server job
is launched by this implementation commit.
