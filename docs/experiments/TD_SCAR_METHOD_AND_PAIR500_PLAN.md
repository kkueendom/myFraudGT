# TD-SCAR: Temporal-Drift Signed Conflict-Aware Residual

## Material Passport

- Branch: `feature/td-scar-drift-prototypes`
- Parent protocol: `protocol/fixed-eval-panel`
- Primary metric: validation-selected Test F1
- First gate: Small-LI seed 42 and Large-LI seed 44
- Budget: 500 epochs from epoch 0

## Motivation

A2 supplies a useful four-slot class-prototype residual, but its elementwise
`tanh` output is often saturated. Earlier methods only rescaled that same
residual and therefore could not learn a genuinely new correction direction.
TD-SCAR preserves A2 as an independently trained anchor and adds two signed,
non-saturating evidence routes: stable multi-prototype evidence and recent
time-weighted evidence.

## Method

The exact A2 prediction is

\[
z_i^A=z_i^{base}+r_i^A.
\]

The stable branch computes a signed log-mean-exp contrast between the positive
and negative four-slot prototype banks. The recent branch uses class memories
accumulated with exponential time weights relative to the largest training
timestamp observed in that epoch. A recent memory becomes visible only in the
next epoch, so the current batch cannot affect its own prediction. Both signed
contrasts use `asinh`, which retains sign and does not impose a hard saturation
bound.

Two no-dropout experts propose L2-bounded signed corrections
`d_i^stable` and `d_i^recent`. A three-way hard router selects exactly one
candidate:

\[
\boxed{
z_i^{final}\in\{z_i^A,\ z_i^A+d_i^{stable},\
z_i^A+d_i^{recent}\}.
}
\]

The router is initialized with an A2 prior. Both direction output layers start
at zero, so the initial prediction is bit-exact A2. It cannot average conflicting
routes; when neither branch has learned a reliable advantage, it selects A2.

## Training Isolation

The original weighted cross-entropy is computed only on `z_A`. Detached
stable/recent candidates receive direction losses, while the router learns the
per-example candidate with the lowest detached weighted CE. Improvements below
`1e-4` relative loss default to the A2 target. New modules are constructed in
`fork_rng`, contain no dropout, use detached A2 inputs, and are gradient-clipped
separately from A2.

## Prospective Pair Gate

TD-SCAR and fixed-panel A2 use identical seeds, target panels, scheduler,
threshold tuning, and maximum-validation-F1 checkpoint selection. TD-SCAR
advances only if completed 500-epoch Test F1 exceeds the rerun fixed-panel A2
by at least `0.005` on both Small-LI and Large-LI. Raw-best is diagnostic only.

Mechanism rejection also occurs if both evidence routes remain unused, either
direction violates the `0.5` norm bound, the recent bank changes within its
own accumulation epoch, or the A2 anchor trajectory changes.
