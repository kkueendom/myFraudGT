# EPRA: End-to-End Prototype Rank Alignment

## Material Passport

- Branch: `feature/epra-dynamic-random`
- Parent protocol: `protocol/dynamic-random-a2` at `f6965a1`
- Primary metric: validation-selected Test F1
- First gate: Small-LI seed 42 and Large-LI seed 44
- Budget: 500 epochs from epoch 0, no early stopping
- Dynamic-random A2 reference: Small-LI `0.46247`, Large-LI `0.30108`

## Failure-Driven Motivation

CADE, TD-SCAR, and UPRC all add a detached post-hoc correction to an A2
anchor. Their correction can fit current training labels but cannot improve the
underlying representation. On Large-LI, UPRC also sees no positive example in
roughly half of random microbatches. Its point objective then supplies only a
negative-class gradient, while its unbounded gain target keeps increasing the
correction until the configured bound becomes the effective stopping rule.

EPRA removes the inference-time corrector. It trains the original A2 model to
separate difficult positives and negatives directly, while using the existing
past-batch multi-prototype bank to shape a class-separable edge representation.
The deployed decoder and parameter count are exactly A2.

## Method

Let `m_i` be the binary margin produced by A2. In each mixed-class training
microbatch, EPRA selects at most 128 lowest-scored positives and 128
highest-scored negatives using detached scores. The pair gap is divided by the
detached within-batch margin standard deviation `sigma_m` (lower bounded by
one), making the target invariant to logit scale. EPRA applies a finite-margin
ranking loss:

\[
L_{rank}=\frac{1}{|P||N|}\sum_{p,n}
\tau_r\operatorname{softplus}
\left(\frac{\gamma-(m_p-m_n)/\operatorname{sg}(\sigma_m)}{\tau_r}\right).
\]

Unlike UPRC's counterfactual gain loss, this objective acts on A2 itself and
its gradient vanishes after the target gap `gamma=0.5` is reached.

For edge representation `h_i`, EPRA queries A2's positive and negative
prototype banks before the current batch updates them. The class-balanced
prototype alignment loss is:

\[
L_{proto}=\frac12\sum_{c\in\{0,1\}}
\mathbb E_{i:y_i=c}
CE\left(
\left[\cos(h_i,p_i^-),\cos(h_i,p_i^+)\right]/\tau_p,
y_i\right).
\]

The complete training objective is:

\[
\boxed{L=L_{A2}+s(e)
\left(\lambda_r L_{rank}+\lambda_p L_{proto}\right),}
\]

where `s(e)` is zero for five warm-up epochs and ramps linearly to one over 25
epochs. Both auxiliary losses return exactly zero unless the current
microbatch contains both classes, preventing the negative-only drift observed
in UPRC. Validation and test labels never enter either loss or the prototype
bank.

## Novelty Relative to A2 and UPRC

1. The pointwise weighted-CE A2 objective is aligned with rare-event ranking
   through finite-margin hard-pair optimization.
2. A2's online multi-prototype evidence becomes an end-to-end representation
   constraint rather than a detached post-hoc routing signal.
3. Mixed-class activation makes the auxiliary objective robust to rare-event
   microbatches without introducing class-dependent inference rules.
4. EPRA has zero inference-time parameters and exact A2 deployment cost.

## Parallel Pair Screen

Two pre-registered strengths share one implementation commit:

| Variant | `lambda_r` | `lambda_p` | Purpose |
|---|---:|---:|---|
| EPRA-Low | 0.05 | 0.02 | Conservative representation update |
| EPRA-Mid | 0.15 | 0.05 | Test whether Low underuses ranking signal |

One variant advances only if its completed 500-epoch Val-selected Test F1
beats the corresponding initial A2 result by at least `0.005` on both datasets.
Raw-best is reported separately against the initial A2 Raw-best result. A delta
below `0.005` is marked as possible sampling fluctuation. The winning strength,
not a dataset-wise mixture, is used for later datasets.

## Required Ablation If EPRA Passes

1. A2
2. A2 + finite-margin rank loss
3. A2 + prototype alignment
4. Full EPRA

The primary attribution is full EPRA versus A2. Component variants need not be
strictly monotone on every dataset, but the full method must have the best mean
and worst-dataset result across the representative pair.
