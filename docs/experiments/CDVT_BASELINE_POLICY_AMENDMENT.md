# CDVT Baseline and Reporting Policy Amendment

Effective date: 2026-07-29

## Purpose

This document supersedes the baseline gate in
`CDVT_FINAL_MODEL_FREEZE.md` and the A2 advancement language in
`CDVT_PHASE0_PHASE1_PREREGISTRATION.md`. Those files remain historical records
of decisions made before the paper objective was clarified. This amendment
does not change the frozen CDVT architecture, training data, sampling protocol,
or any observed result.

The paper question is whether a causal transaction-event view improves the
original FraudGT account-graph model. A2 is an unpublished decoder extension,
so it cannot replace FraudGT as the primary scientific baseline.

## Baseline hierarchy

1. **Primary published baseline: PE-FraudGT.** CDVT retains the ports and ego-ID
   account view and does not use reverse message passing, so PE-FraudGT is the
   architecture-matched parent in FraudGT Table 2.
2. **Strong published reference: Multi-FraudGT.** This is the strongest
   multi-view FraudGT configuration in the original paper and is reported to
   show how CDVT compares with the paper's stronger variant.
3. **Internal strong comparator: initial A2.** A2 records a successful
   unpublished decoder-level extension. It is reported separately to show
   whether CDVT is competitive with the strongest prior internal result, but it
   is not the headline baseline.
4. **Matched implementation control: account-only.** For representative-scale
   multi-seed experiments, FraudGT account-only and CDVT are trained with the
   same seeds and protocol. Same-seed paired deltas are the strongest causal
   evidence for the CDVT contribution.

## Published reference values

All values below are test F1 from the checkpoint selected by validation F1.
FraudGT reports means over five random seeds.

| Dataset | PE-FraudGT | Multi-FraudGT | Initial A2 |
|---|---:|---:|---:|
| AML Small-LI | 0.4581 | 0.4701 | 0.46247 |
| AML Small-HI | 0.7641 | 0.7613 | 0.77984 |
| AML Medium-LI | 0.4353 | 0.4406 | 0.51163 |
| AML Medium-HI | 0.7422 | 0.7593 | 0.77574 |
| AML Large-LI | 0.3044 | 0.3743 | 0.30108 |
| AML Large-HI | 0.6864 | 0.7334 | 0.72897 |

The A2 column is a historical single-run internal reference, not a published
five-seed mean.

## Formal decision rules

Phase 2 advances when frozen seed-42 CDVT:

- exceeds PE-FraudGT on at least four of six datasets;
- has a positive six-dataset mean delta against PE-FraudGT;
- has no unexplained severe collapse; and
- has normal event context outperform shuffled and off interventions on at
  least two representative scales.

Phase 3 reports Small-LI, Medium-LI, and Large-LI with real seeds 42, 43, and
44. The primary Phase 3 estimate is the same-seed paired CDVT minus account-only
delta, with each seed, mean, and standard deviation retained. Published
PE-FraudGT and Multi-FraudGT means remain descriptive references because their
five individual seed results are unavailable for paired inference.

## Metric discipline

- The headline metric is Val-selected Test F1.
- Raw-best Test F1 is supplementary and may only be compared with the A2
  Raw-best column.
- Val-selected and Raw-best values must never be compared across columns.
- Absolute deltas below 0.005 are marked as potentially within
  dynamic-sampling variation.
- Every new manifest records `sampling_protocol=dynamic_random`, dataset,
  variant, seed, Git commit, config, checkpoint, selected epoch, Val-selected
  Test F1, Raw-best Test F1, and the appropriate baseline deltas.
- Architecture and hyperparameters remain frozen while Phase 2 is evaluated;
  this amendment cannot be used to select a model from test results.

## Interpretation boundary

Passing the Phase 2 gate supports continuing the application-oriented paper;
it does not establish universal superiority. The paper may claim improvement
over the architecture-matched PE-FraudGT baseline under the original dynamic
sampling protocol. It must separately disclose performance against
Multi-FraudGT and A2, including losses.
