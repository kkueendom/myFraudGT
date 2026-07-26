# GTF1C Phase 0 v2 Directional Development Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent failure: `GTF1C_PHASE0_V1_DEV_RESULTS.md`
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations allowed: 0
- Statistical confidence or F1-threshold tuning allowed: no

## Authorized Method Change

v1 chose one candidate from a mixed 48-policy family. v2 restores the required
directional structure.

On the selection fold:

1. choose one add threshold from the 16 add-only candidates by empirical
   paired F1;
2. choose one remove threshold from the 16 remove-only candidates by empirical
   paired F1;
3. form three locked candidates: add-only, remove-only, and joint.

On the certification fold:

1. evaluate all three locked candidates;
2. use one-sided `delta/3` for each candidate;
3. GTF1C requires both the time-block and graph-component paired F1 lower
   bounds to be positive;
4. among qualified candidates, deploy the one with the largest conservative
   lower bound, breaking ties by changed count;
5. abstain when no candidate qualifies.

The evaluation fold is used only after candidate selection and certification.
No threshold is re-estimated there.

Row-net and GTPRC row-harm receive the same three preselected candidates. Each
chooses its largest positive certification-fold row utility among candidates
that satisfy its registered break-risk rule.

## Authorized Construct Change

The negative regimes keep their v1 outcome alignment but add a role-specific
score offset so that locked selection-fold thresholds retain comparable
coverage on certification and evaluation folds:

| Regime | Selection offset | Certification offset | Evaluation offset |
|---|---:|---:|---:|
| CPSE remove fragility | 0.0 | 2.0 | 2.0 |
| Alignment drift | 0.0 | 4.0 | 4.0 |
| Rare duplicate | 0.0 | 2.0 | 2.0 |

Offsets do not encode labels and do not change score ordering within a fold.
They repair the failed pressure construct by preserving marginal threshold
scale while the registered alignment changes.

Positive-regime generators are unchanged.

## Fixed Parameters

- real OOF input: unchanged;
- datasets: Small-LI and Large-LI;
- three fold rotations;
- 64 fresh replicates per regime;
- seven distinct regimes and seven GPUs;
- 16 quantiles per direction;
- `block_count=32`;
- family-level `delta=0.05`;
- minimum certification changes: 20;
- practical paired F1 delta: 0.005;
- normal/shuffled/harmful controls;
- fresh seeds `73001` through `73007`.

## Gate

The v1 development gate is reused without modification:

- positive oracle opportunities in all eight positive units;
- at least two negative units expose row-net or GTPRC row-harm false
  qualification above 0.10;
- GTF1C false qualification at most 0.07 in every negative unit;
- shuffled and harmful GTF1C qualification at most 0.05 in every unit;
- GTF1C qualification at least 0.25 in three Small-LI positive regimes and
  two Large-LI positive regimes;
- median oracle coverage at least 0.30 in at least three positive units;
- IID GTF1C qualification at least 80% of IID paired-F1 qualification on both
  datasets.

## Stop Rule

This is the only authorized redesign. If v2 fails any gate:

- stop GTF1C;
- do not change block count, confidence level, practical F1 threshold or
  policy quantiles;
- do not run a formal stress benchmark;
- retain the v1/v2 results as evidence that rare-positive dynamic fraud
  streams may be statistically uncertifiable with the available OOF sample;
- return to the broader dynamic-evidence reliability paper question rather
  than another predictive decoder or router.
