# GTF1C Phase 0 Real-Graph Stress Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Branch: `feature/gtf1c-graph-time-f1-certification`
- Parent result: `CPSE_PHASE0B_OOF_RESULTS.md`
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations allowed: 0
- Full FraudGT or evidence-model training allowed: no

## Research Question

Can graph-time paired F1 certification reject interventions that improve
row-count utility but reduce fraud F1, while retaining useful power for
genuinely positive interventions on realistic AML graph topology?

## Immutable Real-Graph Input

Reuse only the six train-only OOF payloads from:

`/e/yky/FraudGT_cet_results/cpse_phase0b_formal_a78d3396`

CPSE features are not used. The stress benchmark uses:

- frozen OOF A2 score and threshold;
- fraud label loaded after all original OOF training;
- source and destination entity IDs;
- transaction timestamp and unique edge ID;
- existing OOF fold.

Observed base difficulty:

| Dataset | Rows | Positives | TP | FP | FN | Base OOF F1 |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 523,859 | 239 | 44 | 44 | 195 | 0.26911 |
| Large-LI | 104,360 | 72 | 5 | 163 | 67 | 0.04167 |

These are frozen train-only OOF predictions, not the registered initial-A2
validation/test baseline.

## Fold Protocol

For every replicate, all three OOF folds rotate through:

1. **selection fold:** construct score quantiles and choose one candidate
   policy by empirical paired F1;
2. **certification fold:** evaluate the locked policy and decide whether its
   paired F1 lower bound is positive;
3. **evaluation fold:** report the frozen policy's paired F1 and intervention
   behavior.

No policy is selected and certified on the same fold. No validation or test
loader is instantiated.

## Policy Family

The family contains:

- 16 add-only thresholds;
- 16 remove-only thresholds;
- 16 joint add/remove quantile pairs;
- abstention.

Add and remove scores are generated and calibrated separately. A candidate
must change at least 20 selection-fold predictions. The selection fold chooses
the candidate with the largest empirical paired F1 delta. The certification
fold tests only that locked candidate.

Normal scores contain the registered regime signal. Shuffled scores permute
the normal score within direction and fold. Harmful scores reverse its
alignment. The normal policy thresholds remain frozen when evaluating
shuffled and harmful controls.

## Compared Qualification Methods

1. `row_net`: positive corrected-minus-broken with row-Wilson break control;
2. `gtprc_row_harm`: the existing graph-time grouped break-risk control;
3. `iid_paired_f1`: paired F1 lower bound treating rows as independent;
4. `time_block_paired_f1`: paired F1 lower bound using chronological blocks;
5. `gtf1c_graph_time`: proposed paired F1 lower bound using chronological
   blocks with within-block shared-entity components;
6. `oracle_f1`: evaluation-fold best positive-F1 candidate, used only as a
   power and coverage reference.

All non-oracle methods receive the same selected policy. Their only difference
is the certification rule.

## Seven Distinct GPU Regimes

Each task evaluates both Small-LI and Large-LI and all three fold rotations.

| GPU | Regime | Purpose |
|---:|---|---|
| 0 | `iid_positive` | positive aligned reference without shared effects |
| 1 | `entity_positive` | positive utility with shared-entity dependence |
| 2 | `temporal_positive` | positive utility with burst-level dependence |
| 3 | `graph_time_positive` | positive utility with combined entity/time dependence |
| 4 | `cpse_remove_fragility` | many corrected FP but a few broken TP make F1 negative |
| 5 | `alignment_drift` | selection utility degrades before certification/evaluation |
| 6 | `rare_duplicate` | duplicated rare positives and high-degree entity stress |

These are seven distinct dependence/utility mechanisms, not duplicate seeds.

## Development Screen

- 64 replicates per regime;
- 8 replicates per GPU batch;
- one fixed seed per regime;
- 16 candidate quantiles;
- one-sided `delta=0.05`;
- minimum evaluation changes: 20;
- F1 improvement smaller than 0.005 is classified as practically unresolved.

Development may change only simulation construct parameters when:

- the intended positive regime has no positive oracle policy; or
- an intended negative regime does not expose a row-harm/F1 disagreement.

It may not tune GTF1C thresholds or confidence levels to observed method
performance. Any construct change requires a new plan/result commit before
execution.

## Development Gate

The screen passes only if:

- every regime contains both datasets and all three fold rotations;
- the four positive regimes have positive oracle F1 opportunities on both
  datasets;
- at least two negative regimes make `row_net` or `gtprc_row_harm` falsely
  qualify at a rate above 0.10;
- `gtf1c_graph_time` false qualification is at most 0.07 in every negative
  regime;
- shuffled and harmful false qualification is at most 0.05 in every regime;
- GTF1C qualification rate is at least 0.25 in at least three positive
  regimes on Small-LI and at least two on Large-LI;
- GTF1C median oracle-coverage fraction is at least 0.30 in at least three
  positive regime/dataset units;
- in `iid_positive`, GTF1C retains at least 80% of the qualification rate of
  `iid_paired_f1`.

Failure means redesign the statistical object or stop this route. It does not
authorize changing the CPSE encoder, decoder, evidence gate, or original A2.

## Formal Gate

Formal execution is prohibited until a development result commit passes the
construct gate. The formal run will use 512 fresh replicates per regime and
new fixed seeds. Its gate will be frozen in a separate preregistration commit.

## Required Manifest

Every regime task must record:

- regime, dataset, replicate count, seed and Git commit;
- source OOF files and their commits;
- actual fold sizes, labels and base confusion matrices;
- graph-time grouping definition and group-count diagnostics;
- selected direction and threshold;
- normal/shuffled/harmful qualification;
- changed, corrected, broken, paired F1 delta and lower bound;
- qualification, false-qualification and oracle-coverage rates;
- validation/test loader iterations, both fixed to 0;
- `sampling_protocol=dynamic_random`;
- runtime, device and failure reason.

## Stop Rule

If GTF1C cannot distinguish the CPSE-like row-utility/F1 mismatch while
retaining nonzero power in positive regimes, stop the method. Do not replace
the metric with accuracy, tune a decoder, or search thresholds on validation
or test labels.
