# DABR: Decoupled Advantage-Balanced Residual Routing

## Material Passport

- Branch: `feature/dabr-decoupled-ternary-router`
- Primary metric: validation-selected Test F1
- First-stage datasets: Small-LI seed 42 and Large-LI seed 44
- Training budget: 500 epochs from the start, no early stopping
- Strength variants: `s25` and `s50`
- Expansion gate: one fixed strength must beat A2 by at least `0.005` on both
  datasets after the complete trajectory

## Motivation

CPAR-K4 treats four residual doses as separate actions. For binary
cross-entropy, however, loss along one fixed residual direction is monotonic
for an individual labelled sample. The best counterfactual is therefore almost
always an endpoint; the intermediate K4 actions do not define four genuinely
different decisions.

DABR uses the identifiable question instead:

> Should A2's prototype residual be suppressed, retained, or amplified for
> this sample?

## Method

A2 supplies the independently trained anchor

\[
z_i^{A2}=z_i^{base}+r_i,
\qquad r_i=\beta\,ready_i\,\delta_i.
\]

For fixed strength `rho`, DABR evaluates detached actions

\[
d\in\{1-\rho,1,1+\rho\},
\qquad
\ell_{i,d}=CE(z_i^{base}+d r_i,y_i).
\]

The best action becomes a target only when its normalized advantage over the
second-best action exceeds `0.005`; otherwise the target is neutral. Training
weights retain the original class weighting and give each observed action
group equal total supervision mass.

The label-free router consumes detached edge representation, base and residual
margins, prototype similarities, prototype margin, readiness, and
base-residual alignment. Entropy controls how far its expected action may move
from one. Uniform probabilities and a neutral argmax return A2 exactly.

## Isolation guarantees

- A2 is optimized only with its original weighted cross-entropy.
- Router loss cannot update A2 features, encoder, prototype head, or beta.
- A2 and router gradients are clipped separately.
- Router construction restores global RNG state.
- Inference uses no labels or batch-level statistics.
- The same sample is invariant to inference batch partition and ordering.

## Parallel screen

| Variant | Doses | GPUs | Purpose |
|---|---|---|---|
| `s25` | `{0.75, 1.00, 1.25}` | 3-4 | conservative correction |
| `s50` | `{0.50, 1.00, 1.50}` | 5-6 | stronger correction |

Both variants use identical code and a commit-locked queue. Strength is a
pre-registered config parameter embedded in every output name. No test metric
is used to switch strength during a run.

## Decision rule

Matched A2 thresholds are:

| Dataset | A2 selected Test F1 | Required DABR F1 |
|---|---:|---:|
| Small-LI | 0.46247 | at least 0.46747 |
| Large-LI | 0.30108 | at least 0.30608 |

Only one fixed strength passing both rows may advance to Medium-HI and
Large-HI. Raw-best is diagnostic only. A metric pass is rejected as a method
claim if the router collapses to one non-neutral action on almost all samples.
