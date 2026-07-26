# GTF1C Phase 0 v1 Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: development screen completed; gate failed
- Local implementation commit: `3b2b2fe`
- Remote experiment commit: `1134ca2e`
- Remote output:
  `/e/yky/FraudGT_cet_results/gtf1c_phase0_dev_1134ca2e`
- Local output:
  `/Users/kun/FraudGT_experiment_workspace/gtf1c_phase0_dev_1134ca2e`
- Machine aggregate:
  `docs/experiments/GTF1C_PHASE0_V1_DEV_RESULTS.json`
- Manifests: 7/7
- Replicates: 64 per regime
- Dataset/fold trials: 2,688
- Runtime, CUDA, manifest, or protocol errors: none
- Validation/test loader iterations: 0

## Development Gate

| Requirement | Result |
|---|---|
| Positive oracle opportunities in all eight units | pass |
| Row-harm exposes at least two negative units | fail: 0 |
| GTF1C controls all negative units | pass |
| Shuffled/harmful controls rejected | pass |
| Small-LI power in at least three positive regimes | pass: 4 |
| Large-LI power in at least two positive regimes | fail: 0 |
| Median oracle coverage at least 0.30 in three units | pass: 4 |
| IID power retention at least 0.80 on both datasets | fail: Small 1.00, Large 0.45 |

Decision: `REDESIGN_OR_STOP_GTF1C`.

## Positive-Regime Results

| Regime | Dataset | Oracle opportunity | IID-F1 qualification | GTF1C qualification | GTF1C false qualification | Median oracle coverage |
|---|---|---:|---:|---:|---:|---:|
| IID | Small-LI | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| Entity | Small-LI | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| Temporal | Small-LI | 0.948 | 0.854 | 0.786 | 0.021 | 1.000 |
| Graph-time | Small-LI | 0.984 | 0.995 | 0.948 | 0.016 | 1.000 |
| IID | Large-LI | 0.891 | 0.219 | 0.099 | 0.099 | 0.000 |
| Entity | Large-LI | 0.891 | 0.229 | 0.078 | 0.078 | 0.000 |
| Temporal | Large-LI | 0.865 | 0.276 | 0.068 | 0.057 | 0.000 |
| Graph-time | Large-LI | 0.854 | 0.260 | 0.125 | 0.120 | 0.000 |

Small-LI demonstrates that the statistical implementation can retain power
when the base confusion matrix contains enough positive decisions. Large-LI
does not. It has only five OOF true positives across all three folds.

## Structural Failure Analysis

The v1 policy family combined 16 add-only, 16 remove-only, and 16 joint
candidates, then selected one global candidate on the selection fold.

Among GTF1C-qualified Large-LI trials:

- IID: 19/19 selected remove-only;
- entity: 15/15 selected remove-only;
- temporal: 6 remove-only and 7 joint;
- graph-time: 18 remove-only and 6 joint.

Most qualified trials evaluated on fold 2. That fold has:

- 28 fraud positives;
- 21 A2 false positives;
- 0 A2 true positives;
- base F1 equal to 0.

A remove-only policy cannot improve F1 on this fold. It may remove false
positives, but without a base true positive its routed F1 remains zero. The
previous two folds favor remove because they contain many false positives, so
selection and certification can both pass before the direction becomes
useless on the third fold.

This is not fixed by changing a confidence coefficient. It shows that v1
violates the project's required independent add/remove calibration and allows
one direction to dominate the policy family.

## Negative-Construct Failure

The three intended negative regimes did not expose row-harm/F1 disagreement:

| Regime | Selected trials | Mean certification changes | Row-net qualifications | GTPRC row-harm qualifications |
|---|---:|---:|---:|---:|
| CPSE remove fragility | 128 | 0.22 | 0 | 0 |
| Alignment drift | 384 | 21,537.21 | 0 | 0 |
| Rare duplicate | 128 | 1.78 | 0 | 0 |

In CPSE-fragility and rare-duplicate, the selection-fold score scale was much
larger than the certification/evaluation scale. The locked threshold
therefore changed almost no later-fold rows. Alignment drift changed many
rows, but its certification utility was already too poor for either row-harm
method to qualify. The intended disagreement was not activated.

The preregistration explicitly permits construct-only changes when a negative
regime fails to expose row-harm/F1 disagreement.

## Decision

Do not run a formal v1 experiment.

Authorize exactly one structural development revision:

1. select add and remove thresholds independently;
2. certify the preselected add, remove, and joint candidates with multiplicity
   correction;
3. preserve score scale across negative-regime folds while changing
   alignment;
4. keep all confidence levels, F1 thresholds, graph-time groups, datasets and
   OOF inputs unchanged.

Failure of this v2 screen stops GTF1C. It does not authorize a v3 threshold
search.
