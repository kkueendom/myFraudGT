# GTF1C Literature and Novelty Boundary

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Mode: focused method-boundary review
- Review date: 2026-07-26
- Research question: can an auxiliary evidence intervention be certified to
  improve paired F1 under graph and temporal dependence?
- Evidence status: scoped primary-source review, not a systematic review
- Parent negative result: `CPSE_PHASE0B_OOF_RESULTS.md`

## Why the Method Object Changed

CPSE exposed a failure that row-level harm control cannot resolve. On
Large-LI, its normal policy changed 62 predictions, corrected 59 false
positives, and broke only 3 true positives. Row-count utility was strongly
positive, but the summed held-out F1 delta was `-0.10`. In an extremely
imbalanced problem, a small number of broken true positives can dominate many
corrected false positives.

The next method must therefore certify the metric used by the paper rather
than a decomposable proxy:

> certify a locked evidence intervention by a lower confidence bound on its
> paired F1 change, with dependence units derived from transaction entities
> and time.

## Literature Matrix

| Area | Representative primary source | What already exists | Boundary for GTF1C |
|---|---|---|---|
| Non-decomposable metrics | [Narasimhan et al., ICML 2015](https://proceedings.mlr.press/v37/narasimhanb15.html) | F-measure is a pseudo-linear function of confusion-matrix rates; direct and plug-in optimization are established | GTF1C cannot claim novelty for optimizing or expressing F1 |
| Consistent F-measure learning | [Koyejo et al., NeurIPS 2014](https://proceedings.neurips.cc/paper/2014/hash/3644e33a5161ec5f3997a6acb98d4447-Abstract.html) | Plug-in classifiers can be consistent for non-decomposable measures | Thresholding a probability for F1 is not the contribution |
| Complex confusion-matrix metrics | [Narasimhan et al., ICML 2015](https://proceedings.mlr.press/v37/narasimhanb15.html) | F-beta and other ratio-of-linear measures are smooth functions of confusion matrices | The delta-method representation is inherited statistical machinery |
| High-probability policy testing | [Learn Then Test, AOAS 2025](https://doi.org/10.1214/24-AOAS1998) | A candidate family can be calibrated and tested with finite-sample risk control | Data splitting and testing a locked policy are not individually novel |
| Conformal risk control | [Angelopoulos et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Expected bounded losses can be controlled over calibrated prediction sets | GTF1C must not be presented as the first risk-controlled classifier |
| Network-dependent inference | [Kojevnikov, 2021](https://arxiv.org/abs/2101.12312) | Block and dependent-wild bootstraps support inference for smooth functions of network-dependent means | A generic network bootstrap is not the contribution |
| Clustered metric inference | [Beyond Point Estimates, 2026 preprint](https://arxiv.org/abs/2606.03656) | Smooth confusion-matrix functionals permit cluster-robust intervals and paired classifier tests | A clustered F1 confidence interval alone is directly anticipated |
| Dynamic graph evaluation | [Poursafaei et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html) | Evaluation sampling can materially change temporal graph conclusions | Repeating FraudGT dynamic loaders is necessary evidence, not novelty |
| Selective prediction | [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | Coverage-risk selection and abstention are established | An intervention gate or abstention rule is not novel |

## Defensible Combination

No individual component is new. The potentially defensible contribution is
the following relationship:

1. construct paired base and evidence-routed predictions on the exact same
   target transactions;
2. select an add/remove intervention only from train-only target-edge OOF
   predictions;
3. evaluate the intervention as a non-decomposable change in F1, rather than
   as row accuracy or break rate;
4. estimate uncertainty from preregistered graph-time dependency units;
5. require a locked normal-evidence policy to beat shuffled and off controls;
6. validate the resulting certification decision under the original
   dynamic-random FraudGT protocol.

The method is not “a new F1 interval.” It is a qualification procedure for
whether an auxiliary temporal evidence view may alter a frozen fraud-graph
predictor when the target metric is non-decomposable and observations are
graph/time dependent.

## Statistical Object

For a fixed intervention policy \(\lambda\), each target transaction
contributes to the routed and base confusion vectors:

\[
v_i^\lambda =
(\mathrm{TP}_{r,i},\mathrm{FP}_{r,i},\mathrm{FN}_{r,i},
 \mathrm{TP}_{b,i},\mathrm{FP}_{b,i},\mathrm{FN}_{b,i}).
\]

Let

\[
\psi(v) =
\frac{2\mathrm{TP}_r}
     {2\mathrm{TP}_r+\mathrm{FP}_r+\mathrm{FN}_r}
-
\frac{2\mathrm{TP}_b}
     {2\mathrm{TP}_b+\mathrm{FP}_b+\mathrm{FN}_b}.
\]

The point estimate is the paired F1 change
\(\widehat{\Delta F1}=\psi(\bar v)\). Its first-order influence value is

\[
\phi_i =
\nabla\psi(\bar v)^\top(v_i-\bar v).
\]

For preregistered graph-time groups \(g=1,\ldots,G\), define
\(S_g=\sum_{i\in g}\phi_i\). The initial empirical certification object is

\[
\mathrm{LCB}_{1-\delta}
=
\widehat{\Delta F1}
-
z_{1-\delta}
\sqrt{
\frac{G}{G-1}
\frac{\sum_g S_g^2}{n^2}
}.
\]

A policy is F1-qualified only when it was selected on a disjoint OOF fold and
its certification-fold lower bound is positive. This is an asymptotic
cluster-robust object. It is not yet a finite-sample guarantee for arbitrary
transaction graphs.

## Strongest Counter-Argument

A reviewer can reasonably argue:

> This is clustered paired inference applied to F1, followed by ordinary
> sample splitting. The graph-time groups are heuristic and may not capture
> cross-window entity dependence.

This objection remains valid unless experiments show:

- row-IID and row-harm methods falsely qualify interventions under realistic
  rare-positive graph dependence;
- the graph-time paired F1 procedure controls false qualification;
- it retains nonzero power and useful coverage in positive regimes;
- results are robust to alternative group constructions;
- real normal evidence is distinguished from shuffled and off evidence;
- later dynamic validation uses locked policies and same-batch comparisons.

## Claim Boundary

Allowed only after successful validation:

- graph-time paired F1 certification for auxiliary evidence intervention;
- empirical false-qualification control under registered dependence regimes;
- evidence qualification that is aligned with a non-decomposable fraud
  metric;
- a documented distinction between row correction utility and F1 utility.

Forbidden:

- first F1 confidence interval;
- first cluster-robust classifier comparison;
- first network bootstrap;
- finite-sample or distribution-free graph guarantee without a proof;
- causal effects from normal/shuffled/off comparisons;
- improved FraudGT prediction when the method only abstains.

## Review Verdict

`PROCEED_TO_REAL_GRAPH_SEMI_SYNTHETIC_PHASE0_ONLY`.

The method relationship is sufficiently distinct for a controlled
falsification experiment. It is not yet a journal mainline. Failure to control
false F1 qualification or failure to retain power stops GTF1C before any
validation/test loader is opened.
