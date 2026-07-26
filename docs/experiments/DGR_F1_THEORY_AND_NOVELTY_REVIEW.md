# DGR-F1 Theory and Novelty Boundary

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Review date: 2026-07-26
- Parent failures:
  `GTF1C_PHASE0_V2_FORMAL_RESULTS.md` and
  `TREFIC_PHASE0_DEVELOPMENT_RESULTS.md`
- Research question: when can an F1 improvement claim be expected to
  replicate across independent dynamic fraud-graph sampling streams?
- Evidence status: focused primary-source review, not a systematic review

## Why This Is a New Research Object

GTF1C and TREFIC decide whether an evidence policy may alter individual frozen
predictions. Both are intervention-policy certification methods. DGR-F1
instead evaluates a completed model comparison across independent dynamic
sampling streams. It does not choose thresholds, routes, or predictions.

The intended output is one of:

- `REPLICABLE_IMPROVEMENT`;
- `REPLICABLE_HARM`;
- `INSUFFICIENT_INFORMATION`.

An abstention is therefore a first-class evaluation result, not a zero-action
prediction policy.

## Prior Work Boundary

| Area | Primary source | Existing result | Consequence |
|---|---|---|---|
| F1 pseudo-linearity | [Parambath et al., NeurIPS 2014](https://papers.neurips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html) | F-measure optimization can be reduced to cost-sensitive classification | The directional F1 identity is not sufficient novelty |
| Dynamic graph evaluation | [Poursafaei et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html) | Evaluation sampling changes temporal-link conclusions | Sampling sensitivity alone is not novel |
| GNN link-evaluation pitfalls | [Li et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html) | Unified splits, metrics and realistic samples are required | A protocol audit alone is a benchmark contribution |
| Classifier comparison | [Demsar, JMLR 2006](https://www.jmlr.org/papers/v7/demsar06a.html) | Nonparametric comparison across datasets is established | Win counts and rank tests are not new |
| Replication probability | [Berrar, JMLR 2024](https://jmlr.org/beta/papers/v25/24-0158.html) | Replication probability can be estimated for benchmark and CV comparisons | Reporting reproducibility probability is not new |
| Time-uniform inference | [Howard et al., Annals of Statistics 2021](https://doi.org/10.1214/20-AOS1991) | Nonparametric confidence sequences support optional stopping | A confidence sequence is inherited machinery |
| Bounded-mean confidence sequences | [Kuchibhotla and Zheng, ICML 2021](https://proceedings.mlr.press/v139/kuchibhotla21a.html) | Near-optimal confidence sequences exist for bounded observations | Generic sequential mean inference is not novel |
| Network-dependent inference | [Kojevnikov, 2021](https://arxiv.org/abs/2101.12312) | Network bootstrap methods support inference under dependence | Calling rows graph-dependent is not novel |
| Multi-dataset practical equivalence | [Wainer, JMLR 2023](https://www.jmlr.org/papers/v24/22-0907.html) | Bayesian comparison and practical-equivalence regions are established | A practical Delta F1 threshold is not novel |

## Proposed Method Relationship

Working name:

> DGR-F1: Dynamic Graph Replicability Certification for F1.

For independent dynamic stream \(s\), let

\[
D_s = F1_{\mathrm{new},s} - F1_{\mathrm{base},s}.
\]

For a preregistered practical margin \(\epsilon\), define

\[
S_s^+ = \mathbb{1}[D_s \ge \epsilon],
\qquad
S_s^- = \mathbb{1}[D_s \le -\epsilon].
\]

DGR-F1 evaluates two endpoints at fixed sequential checkpoints:

1. the mean paired stream delta;
2. the probability of achieving a practical improvement or harm.

A replicable-improvement claim requires both:

\[
\mathrm{LCB}\{E[D_s]\} > 0,
\qquad
\mathrm{LCB}\{P(S_s^+=1)\} \ge \pi_0.
\]

Replicable harm uses the symmetric upper/lower conditions. Every other case is
reported as insufficient information.

For evidence interventions, each stream also records:

\[
J_s(\rho_s)
=
(A_{c,s}-R_{b,s})
+\rho_s(R_{c,s}-A_{b,s}),
\]

where \(\rho_s=TP_s/(TP_s+FP_s+FN_s)\). The sign of \(J_s\) must agree with the
sign of \(D_s\). This separates failure caused by evidence direction from
failure caused by changing base F1 sensitivity.

## Potential Contribution

The only potentially defensible contribution is the full combination:

1. same-batch paired model evaluation inside each untouched dynamic loader
   stream;
2. independent process streams as replication units, avoiding row-IID claims;
3. joint mean-effect and practical-replication certification;
4. exact F1 directional mechanism audit within each stream;
5. group-sequential three-way reporting with an explicit information limit;
6. prospective validation under rare positives, graph correlation and
   temporal ratio drift.

## Strongest Counter-Argument

A reviewer can describe DGR-F1 as a binomial confidence interval plus a paired
mean interval applied to repeated GNN evaluation. That criticism remains valid
unless the complete method:

- controls false improvement claims when one-event and row-IID methods fail;
- distinguishes positive mean but non-replicable effects;
- detects ratio-driven sign reversal;
- retains power for stable improvements on both Small-LI- and Large-LI-like
  rare-positive templates;
- prospectively predicts the outcome of frozen real checkpoints.

## Claim Boundary

Forbidden:

- first replication-probability estimator;
- first confidence sequence for model evaluation;
- distribution-free validity for correlated streams;
- a predictive FraudGT improvement;
- treating events within one loader stream as independent replicates.

Allowed only after validation:

- a joint replicability criterion tailored to dynamic, non-decomposable F1
  comparison;
- an F1 mechanism audit that separates intervention direction from base-ratio
  variation;
- empirical identification of underpowered or non-replicable fraud-graph
  improvement claims.

## Review Verdict

`PROCEED_TO_ONE_CONTROLLED_DEVELOPMENT_SCREEN`.

Failure stops DGR-F1. No search over the practical margin, replication target,
checkpoint schedule, confidence level, or scenario parameters is allowed.
