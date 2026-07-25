# Dynamic Evidence Reliability: Literature and Novelty Boundary

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Mode: focused literature and innovation-boundary review
- Review date: 2026-07-25
- Research question: how can temporal evidence interventions be measured and
  controlled under dynamic fraud-graph sampling when evidence sensitivity does
  not imply corrective utility?
- Empirical input:
  `docs/experiments/CET_DYNAMIC_RELIABILITY_RESULTS.md`
- Evidence status: scoped primary-source review, not a systematic-review claim

## Why the Research Question Changed

CET-FraudGT v1 failed its registered two-scale gate. Repeated dynamic
evaluation made the failure more specific:

| Dataset | Fusion F1 vs initial A2 | Fusion F1 vs same-batch A2 | Normal-shuffled | Corrected-broken |
|---|---:|---:|---:|---:|
| Small-LI | -0.18101 +/- 0.02015 | -0.17119 +/- 0.01859 | +0.00142 +/- 0.00485 | -179.25 +/- 28.68 |
| Large-LI | +0.02000 +/- 0.04757 | -0.01042 +/- 0.02743 | +0.29120 +/- 0.05111 | -2.42 +/- 17.80 |

Small-LI uses the evidence branch but not its alignment. Large-LI reacts to
aligned history, but that reaction does not reliably correct the frozen A2.
Therefore the next question is not how to add another fusion coefficient. It
is whether an evidence intervention can be qualified with measurable,
out-of-sample harm control and nontrivial corrective coverage.

## Scoped Search

The review used primary conference proceedings, PMLR pages and official paper
pages. Searches covered:

- dynamic and temporal graph evaluation;
- selective classification and abstention;
- conformal prediction and conformal risk control;
- non-exchangeable and covariate-shift risk control;
- graph-specific conformal prediction and uncertainty calibration;
- fraud-oriented risk control.

The search is sufficient to establish a conservative novelty boundary. It does
not establish absolute priority.

## Literature Matrix

| Area | Representative primary source | What already exists | Consequence for this project |
|---|---|---|---|
| Dynamic graph evaluation | [Towards Better Evaluation for Dynamic Link Prediction, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html) | Evaluation sampling can make temporal graph tasks artificially easy; stronger sampling can change conclusions | Dynamic-sampling sensitivity is important but is not itself a novel method |
| Temporal graph benchmarks | [Temporal Graph Benchmark, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html) | Standardized temporal evaluation, dataset diversity and repeatable protocols | A repeated FraudGT audit is useful evidence, not sufficient novelty |
| Graph OOD evaluation | [GOOD, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html) | Explicit covariate and concept-shift benchmarks with repeated runs | Distribution-shift reporting and multiple runs are established practice |
| Selective prediction | [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | Joint prediction and rejection under a coverage constraint | An abstention or intervention gate is not novel by itself |
| Class-conditional selection | [Selective Classification via One-Sided Prediction, AISTATS 2021](https://proceedings.mlr.press/v130/gangrade21a.html) | Class-wise risk and coverage trade-offs | Separate add/remove risks need stronger justification than ordinary one-sided selection |
| High-probability risk control | [Learn Then Test, Annals of Applied Statistics 2025](https://doi.org/10.1214/24-AOAS1998) | Calibrate a family of decisions and test risk constraints with finite-sample confidence | Searching thresholds and controlling a bounded risk are established |
| Expected-risk control | [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Expected value of arbitrary monotone bounded losses can be controlled | Calling broken predictions a loss is not a new contribution |
| Non-exchangeable risk control | [Non-Exchangeable Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf) | Relevance-weighted risk bounds under non-exchangeability, including temporal drift | Dynamic or temporally weighted risk control already has a direct predecessor |
| Covariate-shift LTT | [High Probability Risk Control Under Covariate Shift, PMLR 2025](https://proceedings.mlr.press/v266/almeida25a.html) | Importance-weighted LTT, including a synthetic fraud FPR experiment | Fraud application and shift-aware risk control are not novel alone |
| Graph conformal prediction | [Conformal Prediction Sets for GNNs, ICML 2023](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) | Graph-aware conformity scores with marginal coverage | Applying conformal prediction to a GNN is established |
| Topology-aware graph conformalization | [Conformalized GNNs, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) | Topology-aware correction and graph-specific coverage analysis | Any graph dependence claim needs an explicit assumption and validation |
| Graph calibration under shift | [G-DeltaUQ, ICLR 2024](https://openreview.net/forum?id=ZL6yd6N1S2) | Intrinsic graph uncertainty under distribution shift | Better confidence estimates are adjacent, not the intended contribution |

## What Is Not Novel

The following elements are already known or are routine combinations:

1. repeating evaluation with different sampling streams;
2. comparing normal, shuffled and ablated evidence;
3. selecting a threshold from a calibration split;
4. abstaining when confidence is low;
5. using conformal or LTT-style risk bounds;
6. stratifying results by support, degree or confidence;
7. applying risk control to fraud detection;
8. applying conformal methods to graph predictions.

These elements can support the method, but none can be the headline
contribution.

## Potentially Defensible Method Relationship

A strong method contribution would require all parts below to operate as one
testable relationship:

1. **Paired evidence intervention.** For each sampled target transaction,
   evaluate the frozen base and aligned, shuffled and off evidence states on
   the same sampled graph batch.
2. **Train-only counterfactual utility.** Construct add and remove intervention
   losses exclusively from target-edge OOF predictions. Validation and test
   labels cannot train or choose the intervention policy.
3. **Graph- and time-aware risk units.** Calibrate over preregistered temporal
   or entity-disjoint blocks rather than treating correlated edge exposures as
   independent rows.
4. **Harm-constrained coverage.** Maximize corrected coverage subject to an
   upper confidence bound on the probability of breaking a base-correct
   prediction.
5. **Evidence-specific qualification.** Require aligned evidence to beat both
   shuffled and off controls in sensitivity and corrective utility.
6. **Dynamic-stream validation.** Report risk-coverage and correction-harm
   curves across independent dynamic loader streams, with no sampler reset.

The potentially novel relation is therefore not “conformal FraudGT.” It is a
paired, graph-dependent risk-control formulation for deciding whether an
auxiliary temporal evidence view may intervene on a frozen graph fraud
predictor.

## Required Formal Object

For a base prediction \(b_i\), evidence proposal \(e_i\), label \(y_i\), and
intervention decision \(a_i(\lambda)\), define:

\[
L_{\mathrm{break},i}(\lambda)
=
\mathbb{1}[a_i(\lambda)=1]
\mathbb{1}[b_i=y_i]
\mathbb{1}[e_i\neq y_i],
\]

\[
U_{\mathrm{correct},i}(\lambda)
=
\mathbb{1}[a_i(\lambda)=1]
\mathbb{1}[b_i\neq y_i]
\mathbb{1}[e_i=y_i].
\]

The method must choose a policy family and calibration rule that maximizes
corrective coverage while controlling a preregistered upper bound on
\(E[L_{\mathrm{break}}]\). Add and remove interventions must be calibrated
separately because their class imbalance and costs differ.

This is only a target specification. A paper cannot claim a finite-sample
guarantee until the dependence assumptions and proof are complete.

## Strongest Counter-Argument

A reviewer can say:

> The proposed framework is non-exchangeable conformal risk control applied to
> hand-designed normal/shuffled/off features. Repeated dynamic loaders are
> repeated evaluation, and the current evidence models have no useful
> correction region on Small-LI and negligible coverage on Large-LI.

This criticism is currently valid. It becomes answerable only if the project
adds a graph-specific dependence treatment and demonstrates nonzero,
out-of-sample correction coverage across scales and evidence models.

## Feasibility Risk

The current data are unfavorable:

- Small-LI fusion broke more predictions than it corrected in all eight
  dynamic events.
- Large-LI fusion had positive net correction in only 6 of 12 events and
  changed fewer than 50 predictions in 11 events.
- Low- and medium-support Large-LI regions produced only five interventions
  over 80,986 sampled exposures.
- The earlier TIER OOF policy also failed on both scales.

A statistically valid controller may therefore abstain everywhere. That is a
valid safety result but not a useful predictive method. Nontrivial coverage
must be demonstrated before more encoder training.

## Claim Boundary

Allowed only after successful experiments:

- a graph- and time-blocked paired risk-control formulation for temporal
  evidence interventions;
- empirical control of base-harm risk under the registered dynamic protocol;
- separation of evidence sensitivity from corrective utility;
- cross-scale risk-coverage evidence.

Forbidden:

- first conformal method for graphs;
- first risk-controlled fraud detector;
- first non-exchangeable conformal method;
- distribution-free or finite-sample guarantee without a proved assumption;
- stable F1 improvement from a single dynamic evaluation;
- causal effect identification from normal/shuffled/off ablations.

## Review Verdict

`PROCEED_TO_TRAIN_ONLY_FEASIBILITY_ONLY`.

The reliability question is scientifically defensible, but the current
method is not journal-ready. Do not train a new encoder yet. First test whether
the existing train-only OOF interventions contain a nonzero harm-controlled
coverage region. In parallel, quantify the dynamic sampling distribution of
the frozen A2 across all six datasets so later gains can be interpreted
against measured sampling variability rather than the fixed 0.005 heuristic
alone.
