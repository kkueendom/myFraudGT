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
no-dropout expert proposes an L2-bounded amplitude `a_i`. A signed prototype
direction prevents the dataset-level one-sign collapse observed in CADE:

\[
e_i=\tanh\left(\frac{sim_i^{pos}-sim_i^{neg}}{0.1}\right).
\]

The correction is localized by A2 uncertainty and prototype readiness:

\[
u_i=4\sigma(m_i^A)(1-\sigma(m_i^A)),\qquad
\ell_i=0.1+0.9u_i,
\]

\[
\boxed{z_i^{final}=z_i^A+\ell_i q_i e_i a_i,
\quad \lVert a_i\rVert_2<0.25.}
\]

Here `q_i` is prototype readiness, not a learned gate. The amplitude output is
zero initialized, making the initial model bit-exact A2. When positive and
negative prototype evidence are both present, one constant amplitude cannot
collapse all corrections to the same sign.

## Rare-Event Ranking Objective

Within each training batch, UPRC deterministically selects the lowest-scored
positives and highest-scored negatives under detached A2. For up to 128 examples
per class, it maximizes the candidate's pairwise margin gain over A2:

\[
L_{rank}=\frac{1}{|P||N|}\sum_{p\in P,n\in N}
\operatorname{softplus}\left(
-\frac{(s_p-s_n)-(s_p^A-s_n^A)}{0.25}\right).
\]

A class-balanced point loss applies the same counterfactual-gain principle per
example and supplies a stable signal even when only a few hard pairs are
available. Small center and norm penalties discourage dataset-level constant
shifts and oversized corrections. Unlike ordinary weighted CE, the pairwise
gradient simultaneously pushes hard positives upward and hard negatives
downward, so a one-sign collapse cannot optimize the main branch objective. A
smooth radial squash bounds the correction while retaining gradient beyond the
nominal bound.

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
