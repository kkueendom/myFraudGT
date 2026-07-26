# GTF1C Phase 0 v2 Formal Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: formal stress benchmark completed; gate failed
- Local protocol commit: `ae3e05b`
- Remote experiment commit: `b140bdcd`
- Remote output:
  `/e/yky/FraudGT_cet_results/gtf1c_phase0_v2_formal_b140bdcd`
- Local output:
  `/Users/kun/FraudGT_experiment_workspace/gtf1c_phase0_v2_formal_b140bdcd`
- Machine aggregate:
  `docs/experiments/GTF1C_PHASE0_V2_FORMAL_RESULTS.json`
- Manifests: 7/7
- Replicates: 512 per regime
- Dataset/fold trials: 21,504
- Runtime, CUDA, manifest, or protocol errors: none
- Validation/test loader iterations: 0

## Formal Gate

| Requirement | Result |
|---|---|
| Row-harm exposes at least two negative units | pass: 3 |
| GTF1C false qualification at most 0.07 in all 14 normal units | **fail** |
| Shuffled/harmful qualification at most 0.05 in all units | pass |
| Practical failure at most 0.10 in all positive units | **fail** |
| Small-LI power in at least three positive regimes | pass: 4 |
| Large-LI power in at least two positive regimes | pass: 3 |
| Conditional coverage in three Small-LI regimes | pass: 4 |
| Conditional coverage in two Large-LI regimes | pass: 4 |
| IID power retention at least 0.80 on both datasets | pass: 1.00 / 0.998 |

Decision: `STOP_GTF1C`.

## Negative-Regime Separation

| Regime | Dataset | Row-net false qualification | GTPRC false qualification | GTF1C qualification | GTF1C false qualification |
|---|---|---:|---:|---:|---:|
| CPSE remove fragility | Large-LI | 0.3112 | 0.2975 | 0.0000 | 0.0000 |
| Alignment drift | Large-LI | 0.3034 | 0.2826 | 0.0000 | 0.0000 |
| Rare duplicate | Large-LI | 0.2799 | 0.2533 | 0.0000 | 0.0000 |

All Small-LI negative-regime methods abstain. All shuffled and harmful GTF1C
qualification rates are at most 0.0007.

The formal result confirms the motivating failure: a row-count controller can
qualify a policy in roughly 28-31% of rare-positive Large-LI trials even when
the held-out paired F1 is non-positive. Direct paired F1 certification rejects
all three registered negative mechanisms.

## Positive-Regime Results

| Regime | Dataset | Oracle opportunity | GTF1C qualification | False qualification | Practical failure | Conditional median F1 delta | Replicate any-violation |
|---|---|---:|---:|---:|---:|---:|---:|
| IID | Small-LI | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.55028 | 0.0000 |
| Entity | Small-LI | 1.0000 | 0.9980 | 0.0000 | 0.0000 | 0.44873 | 0.0000 |
| Temporal | Small-LI | 0.9004 | 0.6771 | 0.0469 | 0.0469 | 0.50465 | 0.1406 |
| Graph-time | Small-LI | 0.9922 | 0.8646 | 0.0026 | 0.0033 | 0.46547 | 0.0078 |
| IID | Large-LI | 0.6667 | 0.3333 | 0.0000 | 0.0000 | 0.01532 | 0.0000 |
| Entity | Large-LI | 0.6667 | 0.3333 | 0.0000 | 0.0000 | 0.01521 | 0.0000 |
| Temporal | Large-LI | 0.6862 | 0.2233 | **0.0944** | **0.1068** | 0.00723 | **0.2676** |
| Graph-time | Large-LI | 0.6667 | 0.3405 | 0.0508 | 0.0885 | 0.01384 | 0.1484 |

Conditional clipped oracle coverage has median 1.0 in every positive unit.
The low unconditional Large-LI coverage is caused by abstention, not by tiny
interventions after qualification.

## What Worked

1. Directional selection removed the v1 remove-only collapse.
2. IID and entity-positive regimes satisfy both reliability and power gates.
3. Graph-time-positive trial-level false qualification remains below 0.07 on
   both datasets.
4. Negative row-utility/F1 disagreement is strongly exposed and rejected.
5. Shuffled and harmful controls are almost always rejected.

## Why the Method Still Fails

The temporal-positive Large-LI unit violates both formal reliability limits.
Its trial-level false qualification is 0.0944, and more than one quarter of
replicates contain at least one qualified fold whose evaluation F1 is
non-positive.

This is not a small `Delta F1 < 0.005` ambiguity. The method's certification
assumption does not transfer reliably across temporally heterogeneous folds
when the frozen base has only five OOF true positives. Replacing 32 time
blocks, changing `delta`, or searching a different policy grid after observing
this result would be post-hoc method tuning.

## Claim Boundary

The formal experiment supports:

- row-level correction utility can be strongly positive while paired F1 is
  harmful;
- direct paired F1 certification is safer than row-harm control in the
  registered IID, entity and negative mechanisms;
- rare-positive temporal heterogeneity can defeat a graph-time certification
  method that appears successful in development.

It does not support:

- a valid graph-time F1 guarantee;
- reliable qualification under temporal dependence;
- a real FraudGT improvement;
- formal expansion to validation/test or six datasets;
- a strong methods-paper claim for GTF1C.

## Final Decision

Stop GTF1C permanently. Do not create v3 and do not tune block count,
confidence level, practical F1 threshold or policy quantiles.

The remaining scientifically defensible paper question is a reliability-limit
question:

> Under what positive-count, temporal-stability and dependency conditions is
> evidence-based F1 improvement identifiable in dynamically sampled fraud
> graphs?

Future work must derive and test those identifiability conditions. It must not
return to decoder tuning or claim a predictive gain.
