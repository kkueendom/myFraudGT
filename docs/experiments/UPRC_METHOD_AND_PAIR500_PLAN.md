# UPRC: Uncertainty-Localized Pairwise Rank Corrector

## Material Passport

- Branch: `feature/uprc-pairwise-rank-corrector`
- Parent protocol: `protocol/fixed-eval-panel`
- Primary metric: validation-selected Test F1
- First gate: Small-LI seed 42 and Large-LI seed 44
- Budget after smoke: 500 epochs from epoch 0

## Failure-Driven Motivation

CADE learns a saturated, dataset-level direction: positive on Small-LI and
negative on Large-LI. Its pointwise weighted CE does not force the decoder to
raise difficult positives and lower difficult negatives within the same model.
TD-SCAR avoids that collapse but its discrete router oscillates on Large-LI.
UPRC removes the learned quality gate and route selection. It directly trains a
single signed correction with an objective aligned to rare-event ranking.

## Method

A2 remains the independent anchor:

\[
z_i^A=z_i^{base}+r_i^A.
\]

UPRC reads detached edge, positive/negative prototype, similarity, readiness,
A2/base margin, uncertainty, and A2/base agreement features. A deterministic
no-dropout expert proposes an L2-bounded direction `d_i`. The correction is
localized by A2 uncertainty and prototype readiness:

\[
u_i=4\sigma(m_i^A)(1-\sigma(m_i^A)),\qquad
\ell_i=0.1+0.9u_i,
\]

\[
\boxed{z_i^{final}=z_i^A+\ell_i q_i d_i,\quad \lVert d_i\rVert_2\le0.25.}
\]

Here `q_i` is prototype readiness, not a learned gate. The direction output is
zero initialized, making the initial model bit-exact A2.

## Rare-Event Ranking Objective

Within each training batch, UPRC deterministically selects the lowest-scored
positives and highest-scored negatives under detached A2. For up to 128 examples
per class, it minimizes a pairwise logistic ranking loss:

\[
L_{rank}=\frac{1}{|P||N|}\sum_{p\in P,n\in N}
\operatorname{softplus}\left(-\frac{s_p-s_n}{0.25}\right).
\]

A class-balanced point loss supplies a stable signal even when only a few hard
pairs are available. Small center and norm penalties discourage dataset-level
constant shifts and oversized corrections. Unlike ordinary weighted CE, the
pairwise gradient simultaneously pushes hard positives upward and hard negatives
downward, so a one-sign collapse cannot optimize the main branch objective.

## Isolation And Gate

The original weighted CE is computed only on `z_A`. UPRC inputs and anchor logits
are detached in all auxiliary paths, its parameters are clipped separately, and
module construction preserves the A2 RNG stream. Training and evaluation panels,
seeds, scheduler, threshold tuning, and checkpoint selection remain matched.

UPRC advances only if completed 500-epoch validation-selected Test F1 exceeds
fixed-panel A2 by at least `0.005` on both Small-LI and Large-LI. Raw-best remains
diagnostic. Before a full pair, a short real-data screen must show finite loss,
both positive and negative corrections, a nonzero branch gradient, and unchanged
A2 anchor behavior.
