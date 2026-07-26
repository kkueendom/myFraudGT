# Multi-Model Reliability Method Readiness Review

## Material Passport

- Origin Skill: academic-research-suite / academic-paper-reviewer
- Review mode: methodology focus plus adversarial challenge
- Review date: 2026-07-26
- Target: strong methods-oriented journal
- Sampling protocol: `dynamic_random`
- Primary evidence:
  `docs/experiments/MULTI_MODEL_DYNAMIC_RELIABILITY_RESULTS.md`
- Verification status: ANALYZED

## Editorial Summary

**Current decision: reject as a predictive-method paper; potentially
publishable later after a new method and prospective validation.**

The completed work is a rigorous negative benchmark. It contains 31 manifests,
220 dynamic validation/test events, three independently trained evidence
families, and same-batch normal/shuffled/off/base diagnostics. It establishes
that evidence sensitivity and corrective utility are different quantities.
It does not yet establish a method that improves FraudGT or safely controls
evidence intervention.

The decisive results are:

- aggregate sensitivity/utility mismatch in `9/10` model/dataset units;
- event-consistent mismatch in CET and TIER, but not COSTAR;
- useful-aligned evidence in `0/10` units;
- the preregistered three-family generality gate fails;
- the previous train-only OOF risk-control feasibility gate fails on both
  Small-LI and Large-LI.

Additional decoder gates, residual weights, prototypes, ordinary routers, or
threshold searches cannot repair these missing claims.

## Methodology Reviewer

### Strengths

1. All formal comparisons preserve the original dynamic-random loader
   behavior and the registered initial A2 baseline.
2. Same-batch base diagnostics separate model effects from sampled-batch
   variation.
3. Normal, shuffled, and off conditions distinguish evidence presence from
   evidence alignment.
4. Changed, corrected, and broken predictions expose cases where headline F1
   hides harmful interventions.
5. The result aggregation distinguishes repeated loader events from
   independent model seeds and avoids unsupported IID significance tests.
6. Stop rules were enforced after CET, TIER, and COSTAR failures.

### Blocking Methodology Issues

1. **No successful methodological object.** The current contribution is an
   evaluation protocol and taxonomy. Repetition, counterfactual ablation,
   selective prediction, and risk bounds all have direct predecessors.
2. **No useful out-of-sample region.** Zero of ten evidence units are
   useful-aligned. A safety controller would currently abstain or intervene
   at negligible coverage.
3. **Dependence is not modeled.** Dynamic loader events and repeated target
   exposures are correlated. Row-level Wilson bounds cannot support a
   graph-dependent guarantee.
4. **Cross-model generality is incomplete.** Only two families satisfy the
   event-consistency requirement. COSTAR's Large-LI mean label is not
   repeatable.
5. **No prospective validation.** Existing analyses diagnose fixed failed
   checkpoints. They do not validate a preregistered controller on unseen
   graph/time blocks.
6. **No positive primary endpoint.** Current data cannot support improved F1,
   nonzero safe correction coverage, or controlled intervention harm.

### Required Methodological Upgrade

The paper needs a graph- and time-dependence-aware paired risk controller.
Its policy must be selected using train-only OOF predictions and evaluated on
unseen dependency blocks. Add and remove interventions must be calibrated
separately. The method must maximize corrective coverage subject to an upper
bound on harm to base-correct predictions.

The dependence adjustment must be validated in a controlled benchmark where
the true harm risk is known. Only after that validation may the method be
applied to real AML data.

## Devil's Advocate Review

### Strongest Counter-Narrative

The results can be explained without a new reliability phenomenon:

> Weak auxiliary classifiers perturb a stronger base model under highly
> variable sampled evaluation. Normal/shuffled/off gaps merely show that the
> branch is active; negative correction counts show that it is not useful.

This explanation fits all current results and is more parsimonious than a
claim that a new reliability method has been discovered.

### Critical Challenges

1. A paper cannot infer generality from three families when only two satisfy
   its own repeatability criterion.
2. Adding a fourth arbitrary model solely to pass the family count would be
   post hoc benchmark construction.
3. A risk controller that always abstains can be statistically valid but does
   not support a useful fraud-detection method.
4. A claimed graph-dependent guarantee would be invalid if entity sharing,
   temporal overlap, duplicate edge exposure, or adaptive policy selection is
   omitted from the assumptions.
5. A new temporal encoder trained on the same failed evidence proposals would
   repeat CET rather than answer the reliability question.

### What Would Falsify the Counter-Narrative

The project must show all of the following:

- a preregistered dependence-aware controller controls harm in simulated
  graph/time dependence regimes where row-IID methods fail;
- a genuinely independent causal evidence source has nonzero train-only OOF
  corrective coverage on both Small-LI and Large-LI;
- aligned evidence beats shuffled and off controls;
- the locked controller retains nonzero coverage and positive paired utility
  on unseen graph/time blocks;
- the result persists across dynamic streams and later across model seeds.

## Editorial Decision

`MAJOR_METHOD_REDESIGN_REQUIRED`

The current artifacts can support a benchmark or negative-results paper at a
less method-focused venue. They are not sufficient for the stated target of a
strong methods-oriented journal.

The recommended route is not to collect more failed decoder variants. It is
to develop and validate one integrated methodological relationship:

> graph-time dependency-aware paired risk control for deciding whether a
> causal auxiliary evidence view may alter a frozen fraud-graph prediction.

A new evidence source is required only to test whether this method can achieve
nonzero corrective coverage in real data. It must be qualified before any
full FraudGT fusion model is trained.

## Revision Roadmap

### Priority 1: Method Validity

- Define dependency units from unique transaction IDs, shared entities, and
  temporal proximity.
- State the dependence assumption and derive a valid harm upper bound.
- Validate empirical coverage and power under controlled dependence.
- Separate add and remove interventions and prohibit validation/test policy
  fitting.

### Priority 2: Real-Data Feasibility

- Construct a causal, self-supervised predictive-surprise evidence view that
  is not a renamed TIER/CET branch.
- Run six train-only OOF folds across Small-LI and Large-LI.
- Require nonzero correction coverage, positive paired F1, and aligned-over-
  shuffled utility on both scales.
- Stop the predictive path if either scale fails.

### Priority 3: Formal FraudGT Evaluation

- Lock the method before opening validation/test.
- Use Val-selected Test F1 as the primary endpoint and Raw-best as secondary.
- Run repeated dynamic streams during screening and three model seeds only
  after the two-scale gate passes.
- Report harm, coverage, correction utility, calibration, complexity, and
  failure cases in addition to F1.

## Claim Boundary

Allowed now:

- a rigorous multi-model negative reliability audit;
- evidence sensitivity does not imply corrective utility in the evaluated
  fixed checkpoints;
- dynamic sampling can reverse single-run conclusions.

Not allowed now:

- a new predictive FraudGT model;
- stable improvement over A2;
- three-family generality;
- a finite-sample graph-dependent guarantee;
- useful harm-controlled evidence intervention.

