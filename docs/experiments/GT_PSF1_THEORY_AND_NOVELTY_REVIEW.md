# Graph-Time Paired F1 Inference: Theory and Novelty Review

## Material Passport

- Origin skill: academic-research-suite / deep-research literature and
  methodology review
- Review date: 2026-07-26
- Parent audit:
  `METHODS_JOURNAL_MAINLINE_COMPLETION_AUDIT.md`
- Candidate path: independent graph-dependent paired-F1 inference
- Current status: conditional go for derivation and untouched simulation only
- GPU authorization: none until the derivation and preregistration gates pass

## Research Question

For two fixed fraud classifiers evaluated on the same transaction edges, how
can the difference in population F1 be estimated when:

1. transaction edges share source and destination entities;
2. dependence can propagate beyond directly shared entities;
3. temporal overlap induces additional dependence; and
4. stochastic neighborhood sampling can produce different predictions for the
   same target edge?

The target is inference for a fixed pair of classifiers. It is not a router,
decoder, intervention policy, checkpoint-selection rule or another temporal
evidence encoder.

## Executive Verdict

An ordinary delta-method interval with dyadic or network-robust standard errors
is not sufficiently novel. The required ingredients already exist:

- paired F1/F-beta inference under independent instances;
- dyadic cluster-robust variance estimation, including directed and
  longitudinal dyads;
- network HAC and dependent wild bootstrap inference for smooth functions of
  network-dependent sample means.

A development screen is defensible only for the narrower two-stage estimand:

> paired F1 under stochastic graph evaluation, with target-edge dependence and
> repeated neighborhood-sampling uncertainty separated explicitly.

The potentially new contribution is not the F1 gradient or network bootstrap.
It is the combination of:

1. a paired F1 estimand defined over both target-edge and graph-sampler
   randomness;
2. a two-level influence decomposition separating between-target graph-time
   dependence from within-target neighborhood Monte Carlo variation;
3. a design-effect and power allocation rule for the number of unique target
   edges versus repeated neighborhood samples;
4. diagnostics that distinguish classifier disagreement from evaluation
   sampler instability.

This is a **conditional go**, not a novelty claim. The route must stop if the
two-level estimator reduces algebraically to a standard network smooth-function
bootstrap without a new allocation result, or if untouched simulations do not
show a material validity advantage.

## Verified Literature Boundary

### Paired F1 Inference

Hsu, Liu and Shyr's 2026 psF1pair preprint models the two classifiers'
predictions jointly within each positive and negative instance. It uses
four-cell multinomial distributions, permits classifier correlation on the
same instance, derives exact and asymptotic inference, and provides power and
sample-size calculations. Its explicit sampling assumption is independence
across different instances.

Source:
[Hsu, Liu, and Shyr, psF1pair](https://www.medrxiv.org/content/10.64898/2026.07.15.26358166v1.full)

The preceding psF1 framework already covers interval estimation, testing,
power and sample-size planning for single and comparative F1/F-beta scores.

Source:
[Hsu et al., Statistics in Medicine, 2026](https://onlinelibrary.wiley.com/doi/10.1002/sim.70557)

Lam, Gopal and Qian compare four confidence-interval methods for F1, further
ruling out a claim based only on a new ordinary F1 interval.

Source:
[Lam, Gopal, and Qian, F1 interval comparison](https://arxiv.org/abs/2309.14621)

### Dyadic Dependence

Aronow, Samii and Assenova derive a nonparametric sandwich estimator for data
whose dyads share members. Their scope includes repeated, weighted, directed
and longitudinal dyads and extensions to generalized linear models.

Source:
[Aronow, Samii, and Assenova, 2015](https://doi.org/10.1093/pan/mpv018)

Cameron, Gelbach and Miller provide cluster-robust inference for non-nested
two-way and multiway clustering, including nonlinear estimators and GMM.

Source:
[Cameron, Gelbach, and Miller, 2011](https://doi.org/10.1198/jbes.2010.07136)

These results mean that clustering by source, destination or time, by itself,
is an application rather than a strong methods contribution.

### Network Dependence Beyond Shared Endpoints

Kojevnikov provides block and modified dependent wild bootstrap methods for
network-dependent processes. The paper establishes first-order consistency for
sample means and smooth functions of means and supplies positive-semidefinite
alternatives to network HAC variance estimates.

Source:
[Kojevnikov, network-dependent bootstrap](https://arxiv.org/abs/2101.12312)

Canen, Sugiura and coauthors show that a dyadic-robust estimator can be
inconsistent when non-negligible dependence propagates beyond direct dyadic
neighbors, and develop a network-spillover robust variance estimator.

Source:
[Inference in Linear Dyadic Data Models with Network Spillovers](https://www.cambridge.org/core/journals/political-analysis/article/inference-in-linear-dyadic-data-models-with-network-spillovers/60C668E3E974A9E9493A96816B88B063)

Therefore, direct-endpoint clustering cannot be assumed sufficient for
transaction graphs with neighborhood aggregation.

## Candidate Estimand

Let `e` denote a target transaction and `u` a random neighborhood sample drawn
by the evaluation loader. For fixed classifiers A and B, define:

`Y(e)`:
the fraud label;

`A(e,u), B(e,u)`:
the two binary predictions under neighborhood sample `u`;

`X(e,u)`:
the vector of six confusion contributions:

```text
X = (
  1[A=1,Y=1], 1[A=1,Y=0], 1[A=0,Y=1],
  1[B=1,Y=1], 1[B=1,Y=0], 1[B=0,Y=1]
).
```

Let `mu = E_e E_{u|e}[X(e,u)]`. The paired population quantity is:

```text
Delta(mu) =
  2 mu_B,tp / (2 mu_B,tp + mu_B,fp + mu_B,fn)
  -
  2 mu_A,tp / (2 mu_A,tp + mu_A,fp + mu_A,fn).
```

This is the F1 difference of population confusion moments. It is not the mean
of event-level F1 differences. That distinction must remain explicit because
F1 is nonlinear.

The estimator pools the registered target/sampler draws and evaluates
`Delta(mu_hat)`.

## Two-Level Influence Decomposition

Let `g(mu)` be the gradient of `Delta`. The first-order influence of draw
`(e,u)` is:

```text
psi(e,u) = g(mu)^T (X(e,u) - mu).
```

Write:

```text
m(e) = E[psi(e,u) | e]
eta(e,u) = psi(e,u) - m(e).
```

Then:

```text
psi(e,u) = m(e) + eta(e,u).
```

The proposed variance target separates:

```text
V_total =
  V_graph_time({m(e)})
  +
  E_e[V_sampler(eta(e,u) | e)] / R_e,
```

where `R_e` is the number of neighborhood replicates for target `e`.

Interpretation:

- `V_graph_time` is uncertainty from which related transaction edges are
  observed and how their labels and predictions co-vary;
- `V_sampler` is uncertainty caused by stochastic neighborhood construction
  for the same target;
- increasing unique targets primarily reduces the first component;
- repeating neighborhoods for the same target only reduces the second.

The finite-sample estimator may use:

1. per-target means of the plug-in influence;
2. a graph-time HAC or dependent wild bootstrap over target-edge means;
3. a within-target residual variance correction;
4. positive-semidefinite projection when a graph kernel covariance matrix is
   required.

## Candidate Design-Allocation Result

For approximately constant `R` and cost:

```text
cost = M * c_target + M * R * c_neighbor,
```

where `M` is the number of unique targets, a first-order planning variance is:

```text
Var(Delta_hat) approximately
  D_graph * sigma_between^2 / M
  +
  sigma_within^2 / (M R).
```

`D_graph` is a graph-time design effect estimated from pilot data.

The method must derive and validate a prospective allocation rule for `M` and
`R`. A useful result would identify when another unique target is more valuable
than another neighborhood replicate and provide a minimum budget for a
one-sided practical-margin test.

Without this allocation result, the candidate is mostly an application of
existing smooth-function network inference.

## What Is Inherited

- the F1/F-beta mapping from confusion moments;
- the multivariate delta method;
- within-instance paired classifier covariance;
- dyadic and multiway cluster covariance constructions;
- network HAC kernels;
- dependent wild bootstrap construction;
- positive-semidefinite covariance projection;
- ordinary power analysis.

These components must be cited and cannot be claimed as original.

## What Could Be New

Only the following combined relation is potentially original:

> two-stage, graph-time dependent inference and prospective evaluation-budget
> allocation for a non-decomposable paired F1 estimand when the same target
> transaction can produce different predictions under stochastic neighborhood
> sampling.

The paper would need to contribute all of:

1. a precise estimand under dynamic graph sampling;
2. a consistency or asymptotic-normality result under stated graph and sampler
   conditions;
3. a valid variance/bootstrap construction;
4. a unique-target versus neighborhood-replicate allocation result;
5. simulation evidence against psF1pair, row-IID delta, endpoint dyadic
   clustering, graph HAC without within-target correction, and naive event
   bootstrap;
6. at least one external graph application beyond AML.

## Claims That Are Not Allowed

- first paired F1 inference method;
- first F1 confidence interval or power calculation;
- first graph-robust, cluster-robust or bootstrap inference method;
- first dependent wild bootstrap on networks;
- first treatment of directed or longitudinal dyads;
- guaranteed finite-sample coverage unless it is actually proved;
- general validity for arbitrary graph dependence;
- superiority based only on the already observed FraudGT scenarios.

## Strongest Counter-Argument

F1 is a smooth function of six sample means away from a zero denominator.
Kojevnikov already proves bootstrap consistency for smooth functions of means
under network dependence. Repeated stochastic neighborhoods can be represented
as repeated observations or an augmented dependence graph. Therefore a
reviewer may argue that the candidate is a straightforward application with no
new theorem.

This objection is fatal unless the two-stage variance decomposition and
budget-allocation theorem produce a result that is not already implied by
standard network bootstrap practice and matters empirically.

## Fresh Development Gate

The next screen must use untouched scenarios. It must not replace the failed
DGR-F1 interval on the DGR scenarios.

### Mathematical Gate

All must pass before formal simulations:

1. exact plug-in F1 identity passes for arbitrary paired confusion tables;
2. the influence gradient agrees with numerical differentiation;
3. the variance decomposition is nonnegative and collapses to:
   - psF1pair/row-IID behavior with independent unique targets and `R=1`;
   - graph-only behavior when within-target sampler variance is zero;
   - repeated-measures behavior when graph dependence is absent;
4. duplicate target draws are never treated as independent unique targets;
5. the allocation optimum satisfies boundary and monotonicity checks.

### Untouched Simulation Gate

Pre-register fresh generators covering:

- independent targets, deterministic neighborhoods;
- endpoint-shared dyadic dependence;
- longer-range graph spillovers;
- temporal dependence;
- stochastic neighborhoods with low and high within-target variance;
- rare-positive discrete regimes;
- dense or hub-dominated graphs that violate the working assumptions.

At minimum, require:

- null 95% interval coverage between 0.94 and 0.97 in every in-scope regime;
- one-sided type-I error at most 0.05;
- practical-effect power at least 0.80 at the planned budget;
- interval width materially below a conservative cluster bootstrap in at
  least one preregistered in-scope regime;
- explicit abstention or failed diagnostic in out-of-scope dense/hub regimes;
- allocation rule within 10% of the empirical minimum cost for 80% power in
  both low- and high-sampler-variance regimes.

### External-Validity Gate

Passing simulation only authorizes:

1. a frozen-checkpoint repeated dynamic evaluation on AML;
2. an independent public graph dataset with a stochastic neighborhood
   classifier;
3. prospective comparison of estimated and empirical repeated-evaluation
   uncertainty.

It does not authorize a predictive improvement claim.

## Decision

`CONDITIONAL_GO_GT_PSF1_DERIVATION`.

Proceed to a separate branch and write a prospective derivation plus simulation
specification. Do not start GPU work yet. Stop this path if:

- no allocation theorem can be stated independently of a selected simulator;
- the estimator is only a renamed generic network bootstrap;
- graph/sampler variance components cannot be identified from registered
  repeated target draws;
- no external graph benchmark can be secured.
