# GTF1C Phase 0 v2 Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: development gate passed; formal result not yet established
- Local implementation commit: `63ee89f`
- Remote experiment commit: `cc5ec29c`
- Remote output:
  `/e/yky/FraudGT_cet_results/gtf1c_phase0_v2_dev_cc5ec29c`
- Local output:
  `/Users/kun/FraudGT_experiment_workspace/gtf1c_phase0_v2_dev_cc5ec29c`
- Machine aggregate:
  `docs/experiments/GTF1C_PHASE0_V2_DEV_RESULTS.json`
- Manifests: 7/7
- Replicates: 64 per regime
- Dataset/fold trials: 2,688
- Runtime, CUDA, manifest, or protocol errors: none
- Validation/test loader iterations: 0

## Registered Development Gate

| Requirement | Result |
|---|---|
| Positive oracle opportunities in all eight units | pass |
| Row-harm exposes at least two negative units | pass: 3 |
| GTF1C controls all negative units | pass |
| Shuffled/harmful controls rejected | pass |
| Small-LI power in at least three positive regimes | pass: 4 |
| Large-LI power in at least two positive regimes | pass: 3 |
| Median oracle coverage at least 0.30 in three units | pass: 4 |
| IID power retention at least 0.80 on both datasets | pass: 1.00 / 1.00 |

Decision: `PROCEED_TO_FORMAL_PREREGISTRATION`.

## Negative-Regime Separation

| Regime | Dataset | Row-net qualification | Row-net false qualification | GTPRC false qualification | GTF1C qualification | GTF1C false qualification |
|---|---|---:|---:|---:|---:|---:|
| CPSE remove fragility | Large-LI | 0.318 | 0.312 | 0.292 | 0.000 | 0.000 |
| Alignment drift | Large-LI | 0.276 | 0.276 | 0.266 | 0.000 | 0.000 |
| Rare duplicate | Large-LI | 0.297 | 0.297 | 0.255 | 0.000 | 0.000 |

Small-LI negative regimes produce no qualifications for any method. Their
base confusion matrix has enough true positives that the registered
remove-fragility construction does not create the same row-net/F1 reversal.
The claim is therefore mechanism-specific, not cross-scale generality.

## Positive-Regime Power

| Regime | Dataset | Oracle opportunity | IID-F1 qualification | GTF1C qualification | GTF1C false qualification |
|---|---|---:|---:|---:|---:|
| IID | Small-LI | 1.000 | 1.000 | 1.000 | 0.000 |
| Entity | Small-LI | 1.000 | 1.000 | 1.000 | 0.000 |
| Temporal | Small-LI | 0.875 | 0.854 | 0.708 | 0.078 |
| Graph-time | Small-LI | 0.990 | 0.974 | 0.849 | 0.005 |
| IID | Large-LI | 0.667 | 0.333 | 0.333 | 0.000 |
| Entity | Large-LI | 0.667 | 0.333 | 0.333 | 0.000 |
| Temporal | Large-LI | 0.672 | 0.406 | 0.203 | 0.089 |
| Graph-time | Large-LI | 0.667 | 0.365 | 0.349 | 0.042 |

Direction-specific threshold selection fixes the v1 remove-only collapse.
GTF1C retains nonzero power on both scales and completely rejects the three
registered negative Large-LI mechanisms.

## Conditional Utility

The machine aggregate stores zeros for abstained trials, so its unconditional
median oracle-coverage fraction is zero whenever qualification is below 0.5.
Qualified-trial diagnostics give a different and necessary view:

| Regime | Dataset | Qualified trials | Conditional median paired F1 delta | Conditional median clipped oracle coverage |
|---|---|---:|---:|---:|
| IID | Small-LI | 192 | 0.54793 | 1.00 |
| Entity | Small-LI | 192 | 0.44731 | 1.00 |
| Temporal | Small-LI | 136 | 0.52098 | 1.00 |
| Graph-time | Small-LI | 163 | 0.46514 | 1.00 |
| IID | Large-LI | 64 | 0.01503 | 1.00 |
| Entity | Large-LI | 64 | 0.01540 | 1.00 |
| Temporal | Large-LI | 39 | 0.01274 | 1.00 |
| Graph-time | Large-LI | 67 | 0.01664 | 1.00 |

These conditional values do not replace qualification rate. They show that
abstention, rather than zero intervention size after qualification, explains
the unconditional Large-LI coverage statistic.

## Residual Risks

The development pass is not sufficient for a method claim:

1. temporal-positive false qualification is 0.078 on Small-LI and 0.089 on
   Large-LI, above the intended 0.07 reliability target;
2. the negative disagreement is exposed only on Large-LI;
3. the graph-time groups remain an empirical dependence approximation;
4. fold rotations within a replicate are correlated and are not independent
   model seeds;
5. real CPSE evidence still fails and GTF1C has not improved FraudGT F1.

## Decision

Proceed to a fresh 512-replicate formal stress benchmark only after freezing a
stricter formal gate. The formal gate must control false qualification in all
normal units, including positive regimes, and report conditional utility
separately from abstention.
