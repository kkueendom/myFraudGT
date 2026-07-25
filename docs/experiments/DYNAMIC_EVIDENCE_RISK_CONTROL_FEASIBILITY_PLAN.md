# Dynamic Evidence Risk Control: Feasibility Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: completed; F1 failed and F2 completed
- Branch: `feature/cet-fraudgt-complementary-encoder`
- Sampling protocol: `dynamic_random`
- Formal baseline: initial A2 only
- Model training allowed in this phase: no
- Results:
  - `docs/experiments/DYNAMIC_EVIDENCE_RISK_CONTROL_FEASIBILITY_RESULTS.md`
  - `docs/experiments/A2_DYNAMIC_SAMPLING_STABILITY_RESULTS.md`

## Objective

Determine whether temporal evidence contains a nontrivial, train-only OOF
intervention region whose correction utility generalizes across held-out
folds while controlling harm to base-correct predictions.

This phase is a feasibility gate. It is not another router search and does not
open validation or test labels for policy development.

## Experiment F1: Leave-One-Fold-Out OOF Risk-Coverage Audit

### Inputs

Use only the six existing TIER Phase 2b OOF tables:

- Small-LI seed 42, folds 0-2;
- Large-LI seed 44, folds 0-2.

Allowed fields:

- detached A2 score;
- normal, shuffled and off evidence scores;
- evidence margin and base-evidence disagreement;
- support;
- target timestamp and source/destination identifiers if already recorded;
- train label and fold ID.

Validation and test loaders must remain unopened.

### Candidate Policies

Audit add and remove directions independently. Preregister only monotone policy
families:

1. evidence-utility score threshold;
2. utility score plus minimum evidence support;
3. utility score plus maximum A2 confidence margin;
4. intersection of aligned-over-shuffled sensitivity and utility score.

No arbitrary Boolean rule search, unrestricted tree search or post hoc dataset
specific feature engineering is allowed.

### Cross-Fold Procedure

For each held-out fold:

1. fit or calibrate the utility score on the other two folds;
2. select the least restrictive policy satisfying the registered harm bound on
   the two calibration folds;
3. freeze the policy;
4. evaluate changed, corrected, broken and paired F1 delta on the held-out
   fold;
5. repeat for normal, shuffled and off evidence.

Rows sharing a target entity or temporal block must be grouped for uncertainty
estimation when identifiers permit it. If the stored OOF artifact lacks the
required identifiers, report row-level estimates as descriptive only and do
not claim graph-dependent guarantees.

### Risk and Utility

Primary harm:

\[
R_{\mathrm{break}}
=
\frac{\sum_i L_{\mathrm{break},i}}
{\max(\sum_i \mathbb{1}[a_i=1],1)}.
\]

Primary utility:

\[
C_{\mathrm{net}}
=
\sum_i U_{\mathrm{correct},i}
-
\sum_i L_{\mathrm{break},i}.
\]

Also report intervention coverage, corrected/broken ratio, paired F1 delta,
and Wilson or Clopper-Pearson upper confidence bounds. Statistical assumptions
must be stated next to every bound.

### Feasibility Gate

Both datasets must satisfy all conditions on held-out folds:

- at least 50 total changed predictions;
- corrected greater than broken;
- corrected/broken at least 1.5;
- paired F1 delta greater than 0;
- normal net correction exceeds shuffled by at least 10 predictions or 0.01
  paired F1;
- the registered harm upper bound is at most 0.40;
- at least two of three held-out folds have positive net correction.

If either dataset fails, stop predictive evidence intervention work. Do not
tune thresholds, train a new router or reinterpret the same OOF labels.

## Experiment F2: Six-Dataset A2 Dynamic-Sampling Stability Audit

### Purpose

Measure how much Val-selected Test F1 varies solely because the original
dynamic val/test loaders continue sampling without RNG restoration. This
quantifies the uncertainty against which future improvements must be judged.

### Fixed Models

Use the existing initial-A2 checkpoints. Do not retrain A2 and do not replace
the registered historical scores.

### Task Layout

Run seven scientifically distinct streams concurrently:

| GPU | Task |
|---:|---|
| 0 | Small-LI A2 stream 1 |
| 1 | Small-HI A2 stream 1 |
| 2 | Medium-LI A2 stream 1 |
| 3 | Medium-HI A2 stream 1 |
| 4 | Large-LI A2 stream 1 |
| 5 | Large-HI A2 stream 1 |
| 6 | rotating second stream, one dataset at a time |

Each stream performs at least eight consecutive full val/test evaluation
events. The val-derived threshold is recomputed for each event and then applied
to the immediately following test event.

### Protocol

- train, val and test configuration remains the original FraudGT configuration;
- `LinkNeighborLoader(..., shuffle=True)`;
- `val.fixed_target_panel=False`;
- no fixed target edge panel;
- no independent evaluation generator;
- no sampler RNG restoration;
- no model update;
- process RNG seeded once at stream start;
- every result records `sampling_protocol=dynamic_random`.

### Outputs

Per event:

- val-selected threshold;
- validation F1;
- test F1, precision, recall and confusion counts;
- sampled rows, unique edges and unique-edge rate;
- positive count and prevalence;
- elapsed time.

Per dataset:

- mean, standard deviation, min, max and 95% bootstrap interval;
- threshold distribution;
- sign and magnitude relative to the registered initial A2 score;
- empirical probability that an apparent gain exceeds 0.005;
- recommended dataset-specific “sampling variation” band.

The registered initial A2 table remains the formal baseline. Repeated
checkpoint evaluation is a sampling audit, not a replacement result.

## Decision After F1 and F2

### If F1 Passes

Implement a graph/time-blocked paired risk controller and test it on frozen
evidence proposals before any new encoder training. A journal claim remains
conditional on a formal dependence argument and cross-scale held-out results.

### If F1 Fails

Stop predictive intervention development. Use F2 and the completed CET/TIER
counterfactual audits to frame an evaluation-method paper on temporal evidence
reliability under dynamic fraud-graph sampling. Expand the benchmark across
multiple existing FraudGT variants only after preregistering the model set.

**Observed:** F1 failed on both Small-LI and Large-LI. Predictive intervention
development is stopped.

### If F2 Shows Large Sampling Variation

Future model comparisons require repeated dynamic streams or multiple seeds
whose confidence interval excludes zero. The fixed 0.005 heuristic is retained
for protocol continuity but cannot support a stability claim by itself.

**Observed:** all six dataset-specific event bands exceed 0.005; Medium-LI and
Large-LI are especially unstable. Proceed to a preregistered multi-model
reliability benchmark.

## Engineering Discipline

- Commit every code change before execution.
- Never overwrite previous outputs.
- Smoke test every registered task.
- Record dataset, model, seed, commit, config, checkpoint, protocol, metrics,
  loader audit and runtime in each manifest.
- Keep polling infrequent and logs concise.
- Do not create duplicate GPU jobs merely to fill a card.
