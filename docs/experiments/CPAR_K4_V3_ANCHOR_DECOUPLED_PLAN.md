# CPAR-K4 V3: Anchor-Decoupled Counterfactual Action Routing

## Material Passport

- Origin: academic-research-suite / experiment-agent
- Mode: plan + run
- Branch: `feature/cpar-k4-crn-safe`
- Primary metric: validation-selected Test F1
- Screen: Small-LI seed 42 and Large-LI seed 44, 500 epochs
- Decision gate: both datasets must exceed matched A2 by at least `0.005`

## Why V2 is not a clean test

V2 eventually routes almost every sample to dose `1.5`. Its task loss flows
through the routed logits, so a globally useful scale can train the router and
also change the base, prototype head, beta, and encoder updates. Constructing
the extra router also advances global RNG, and joint gradient clipping lets
router gradients change the A2 gradient norm. Therefore V2's gain or loss
cannot be attributed only to sample routing.

## V3 method

For sample `i`, A2 predicts

\[
z_i^{A2}=z_i^{base}+r_i,
\qquad
r_i=\beta\,ready_i\,\delta_i.
\]

CPAR evaluates four detached counterfactual actions during training:

\[
d_k\in\{0,0.5,1,1.5\},
\qquad
\ell_{ik}=CE(z_i^{base}+d_k r_i,y_i).
\]

The lowest-loss action is used only when its normalized advantage over the
second-best action exceeds a fixed gap. Otherwise the target is the neutral
action `d=1`. A class-weighted, action-group-balanced auxiliary CE trains a
label-free router from detached edge representation, base margin, prototype
similarities, prototype margin, readiness, residual margin, and base-residual
alignment.

At inference, the entropy-calibrated expected action gives

\[
z_i^{V3}=z_i^{base}+d_i r_i.
\]

Uniform router probabilities and an explicit neutral decision return A2
exactly.

## Anchor decoupling

V3 optimizes original parameters with the untouched A2 objective:

\[
L=L_{A2}(z^{A2},y)+\lambda L_{route}.
\]

`L_route` receives only detached A2 features. The returned routed logits are
used for train diagnostics and validation/test prediction, but never replace
`z_A2` in the main training loss. Router and A2 parameters are gradient-clipped
separately.

Router construction runs inside a forked CPU RNG context. Consequently, the
A2 parameter initialization and subsequent data-loader RNG state match the A2
control despite the additional module. This common-random-number design turns
the experiment into a controlled test of the decoder correction.

## Formal protocol

| Dataset | Seed | A2 selected Test F1 | Required V3 F1 |
|---|---:|---:|---:|
| Small-LI | 42 | 0.46247 | at least 0.46747 |
| Large-LI | 44 | 0.30108 | at least 0.30608 |

- Both runs start with `optim.max_epoch=500` and no early stopping.
- The existing A2 trajectory is the matched reference.
- Raw-best Test F1 is diagnostic only.
- No validation/test labels enter gradients or routing.
- Intermediate reads do not alter the scheduler.
- Failure on either completed dataset rejects expansion.

## Mechanism checks

The route must retain meaningful action diversity after auxiliary supervision
ends. A pass requires more than metric gain: diagnostics must rule out a
near-uniform non-neutral action, confirm an exact-A2 fallback subset, and show
that the gain is not merely another global beta multiplier.
