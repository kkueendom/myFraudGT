# CADE: Counterfactual Advantage-Guided Directional Evidence Editor

## Material Passport

- Branch: `feature/cade-directional-editor`
- Parent protocol: `protocol/fixed-eval-panel`
- Primary metric: validation-selected Test F1
- First gate: Small-LI seed 42 and Large-LI seed 44
- Budget: 500 epochs from epoch 0

## Motivation

P0, CPAR, ACDR, and DABR all controlled the magnitude of A2's existing
prototype residual. That residual saturates near its `tanh` bounds, and the
counterfactual dose targets almost never request suppression. CADE instead
learns a new signed correction that may agree with or oppose A2.

## Method

A2 remains an independently optimized anchor:

\[
z_i^A=z_i^{base}+r_i^A.
\]

From detached edge and positive/negative prototype representations, CADE's
direction expert produces `u_i`. Common softmax shift is removed for multi-logit
classification, and an L2 projection bounds the correction without elementwise
`tanh` saturation:

\[
d_i=\frac{\bar u_i}{\max(1,\lVert\bar u_i\rVert_2/B)},\qquad B=0.5.
\]

A separate quality expert predicts whether that proposed direction is useful:

\[
q_i=[2\sigma(g_i)-1]_+,
\qquad
\boxed{z_i^{final}=z_i^A+q_i d_i}.
\]

Both output layers are zero initialized. Initially `d_i=0`, `q_i=0`, and the
forward output is bit-exact A2.

## Training Isolation

The original weighted cross-entropy is computed only on `z_A`. CADE receives:

\[
L_{CADE}=0.25L_{direction}+0.10L_{quality}+10^{-4}L_{norm}.
\]

`L_direction` trains `stopgrad(z_A)+d`; `L_quality` predicts the detached
relative CE advantage of that direction. These losses remain active for all
500 epochs. All editor inputs are detached, A2/editor gradients are clipped
separately, and the no-dropout editor is constructed inside `fork_rng`.

## Prospective Pair Gate

Both methods use the same fixed target panels, seeds, scheduler, threshold
tuning, and max-validation-F1 checkpoint rule. CADE advances only if completed
500-epoch Test F1 exceeds rerun fixed-panel A2 by at least `0.005` on both
Small-LI and Large-LI. Raw-best is diagnostic and cannot select a method or
checkpoint.

Mechanism rejection also occurs if the editor remains inactive, produces only
one signed direction, violates the `0.5` bound, or changes the A2 anchor
trajectory.
