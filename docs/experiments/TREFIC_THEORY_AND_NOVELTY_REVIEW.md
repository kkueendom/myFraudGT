# TREFIC Theory and Novelty Boundary

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Review date: 2026-07-26
- Research question: when is the sign of evidence-induced F1 change
  identifiable under temporal variation in a frozen fraud predictor?
- Parent result: `GTF1C_PHASE0_V2_FORMAL_RESULTS.md`
- Evidence status: scoped primary-source review, not a systematic review

## Exact Intervention Identity

Let the frozen base confusion counts be \(T=\mathrm{TP}\),
\(P=\mathrm{FP}\), and \(N=\mathrm{FN}\). Let an evidence policy make four
types of changes:

- \(A_c\): add a missed fraud, changing FN to TP;
- \(A_b\): add a false alarm, changing TN to FP;
- \(R_c\): remove a false alarm, changing FP to TN;
- \(R_b\): remove a true fraud prediction, changing TP to FN.

After intervention:

\[
T' = T + A_c - R_b,\quad
P' = P + A_b - R_c,\quad
N' = N - A_c + R_b.
\]

Define

\[
W = T + P + N,\qquad
\rho = \frac{T}{W}.
\]

Direct cross multiplication of base and routed F1 gives:

\[
\operatorname{sign}(F1'-F1)
=
\operatorname{sign}
\left[
(A_c-R_b)+\rho(R_c-A_b)
\right].
\]

For remove-only intervention, F1 improves only when
\(\rho R_c > R_b\). If one true positive is broken, the required corrected
false positives exceed \(1/\rho\). For add-only intervention, F1 improves
when \(A_c > \rho A_b\).

This identity explains the CPSE result. Large-LI has \(T=5\), \(P=163\),
\(N=67\), so \(\rho=0.02128\). Its 59 corrected false positives and 3 broken
true positives give:

\[
-3 + 0.02128 \times 59 = -1.744 < 0.
\]

Positive row-count utility therefore correctly predicts neither the sign nor
the size of F1 change.

## What Prior Work Already Covers

| Area | Primary source | Established result | Consequence |
|---|---|---|---|
| F-measure pseudo-linearity | [Parambath et al., NeurIPS 2014](https://papers.neurips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html) | F-measure optimization reduces to cost-sensitive classification with unknown costs | The weighted sign identity is not a standalone novelty |
| F-measure bounds | [Bascol et al., AISTATS 2019](https://proceedings.mlr.press/v89/bascol19a.html) | Tight links between cost-sensitive error profiles and attainable F-measure | TREFIC cannot claim the first F1 cost interpretation |
| Non-decomposable optimization | [Narasimhan et al., ICML 2015](https://proceedings.mlr.press/v37/narasimhana15.html) | Pseudo-linear confusion metrics can be optimized through adaptive linearization | Additive directional utility is inherited machinery |
| Clustered metric inference | [Beyond Point Estimates, 2026 preprint](https://arxiv.org/abs/2606.03656) | Smooth confusion metrics support cluster-robust paired inference | A clustered F1 interval is not novel |
| Network-dependent inference | [Kojevnikov, 2021](https://arxiv.org/abs/2101.12312) | Network block and wild bootstraps support smooth-function inference | Graph dependence alone is not novel |
| Robust validation under shift | [Cauchois et al., JASA 2024](https://doi.org/10.1080/01621459.2023.2298037) | Performance can be certified over specified distributional uncertainty sets | A generic uncertainty envelope is not novel |

## New Identifiability Problem

The intervention weights depend on the frozen base ratio \(\rho\), and
\(\rho\) changes across dynamic fraud-graph streams. Large-LI OOF folds have:

| Fold | TP | FP | FN | \(\rho\) |
|---:|---:|---:|---:|---:|
| 0 | 2 | 100 | 27 | 0.01550 |
| 1 | 3 | 42 | 12 | 0.05263 |
| 2 | 0 | 21 | 28 | 0.00000 |

A remove policy can look beneficial and be certified on folds 0 and 1, then
be incapable of strict F1 improvement on fold 2. This is the failure observed
in GTF1C.

The proposed research object is therefore not another confidence interval. It
is:

> directionally weighted evidence utility that must remain positive over a
> preregistered temporal envelope of the base F1 sensitivity ratio.

The working name is TREFIC: Temporal Ratio-Envelope F1 Intervention
Certification.

## Proposed Conditional Guarantee

For a policy \(\lambda\), define

\[
J_\lambda(\rho)
=
(A_c-R_b)+\rho(R_c-A_b).
\]

Because \(J_\lambda\) is affine in \(\rho\),

\[
\min_{\rho\in[\rho_L,\rho_U]} J_\lambda(\rho)
=
\min\{J_\lambda(\rho_L),J_\lambda(\rho_U)\}.
\]

TREFIC uses the conservative envelope \([0,\rho_U]\). The upper endpoint is a
one-sided confidence bound estimated only from selection and certification
OOF base confusion counts. A candidate qualifies only if graph-component and
time-block lower bounds for its additive utility are positive at both
endpoints.

The guarantee is conditional:

- future base \(\rho\) lies inside the registered envelope;
- the graph/time dependence approximation is adequate;
- directional utility is stable from certification to evaluation.

No distribution-free temporal guarantee is claimed.

## Potentially Defensible Combination

The contribution can only be the complete relationship:

1. exact F1-aligned decomposition of add/remove evidence intervention;
2. explicit separation of base-ratio uncertainty from evidence-utility
   uncertainty;
3. temporal ratio-envelope certification on train-only OOF predictions;
4. graph/time lower bounds at both envelope endpoints;
5. normal/shuffled/harmful and dynamic-fold falsification.

## Strongest Counter-Argument

A reviewer can describe TREFIC as cost-sensitive F1 optimization with a
hand-chosen uncertainty interval and cluster-robust inference. That criticism
is fatal unless TREFIC:

- rejects the formal GTF1C temporal failures without changing their score
  generators;
- retains useful add-direction power on both Small-LI and Large-LI;
- controls false qualification in all registered regimes;
- distinguishes normal evidence from shuffled and harmful controls;
- states and empirically audits envelope coverage.

## Claim Boundary

Forbidden:

- first F1 cost-sensitive method;
- first robust F1 interval;
- first graph-dependent classifier comparison;
- a guarantee outside the registered \(\rho\) envelope;
- improved FraudGT prediction without real-evidence validation.

Allowed only after successful validation:

- a temporal ratio-envelope criterion for F1-aligned evidence intervention;
- empirical identification of when remove interventions are not supportable
  under rare-positive temporal variation;
- separation of base-metric sensitivity from evidence utility.

## Review Verdict

`PROCEED_TO_ONE_DEVELOPMENT_SCREEN`.

Failure stops TREFIC. No search over the lower envelope endpoint, confidence
level, time-block count, or policy grid is allowed.
