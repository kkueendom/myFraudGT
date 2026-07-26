# TREFIC Phase 0 Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent validate mode
- Experiment: TREFIC Phase 0 preregistered development screen
- Remote commit: `6cbc6d31`
- Local implementation commit: `f99779c`
- Sampling protocol: `dynamic_random`
- Source OOF root:
  `/e/yky/FraudGT_cet_results/cpse_phase0b_formal_a78d3396`
- Remote output:
  `/e/yky/FraudGT_cet_results/trefic_phase0_dev_6cbc6d31`
- Local mirror:
  `/Users/kun/FraudGT_experiment_workspace/trefic_phase0_dev_6cbc6d31`
- Replicates: 64 per regime
- Trials: 2 datasets x 7 regimes x 3 rotations x 64 = 2,688
- Validation/test loader iterations: 0/0
- Status: reproducibly completed; preregistered gate failed

## Decision

`STOP_TREFIC`.

TREFIC controlled every registered false-qualification condition but had
insufficient Large-LI qualification power. The failure is therefore not
safety but utility under rare-positive graph/time dependence.

No TREFIC v2 is permitted. Do not tune the lower ratio endpoint, Wilson
confidence level, family delta, block count, policy quantiles, or registered
score regimes.

## Gate

| Registered condition | Result |
|---|---|
| Exact F1 sign identity | PASS, 9,990 exhaustive cases |
| Row-net/GTPRC expose at least two negative units | PASS, 3 units |
| TREFIC false qualification <= 0.07 in all 14 normal units | PASS |
| Shuffled/harmful qualification <= 0.05 in all units | PASS |
| Practical failure <= 0.10 in all positive units | PASS |
| Small-LI power in at least 3 positive regimes | PASS, 4/4 |
| Large-LI power in at least 2 positive regimes | FAIL, 0/4 |
| Conditional median Delta F1 >= 0.005 when qualified | PASS |
| Small-LI oracle coverage in at least 3 regimes | PASS, 4/4 |
| Large-LI oracle coverage in at least 2 regimes | FAIL, 1/4 |
| IID power retention >= 70% on both datasets | FAIL, Small 100%, Large 0% |
| Overall | **FAIL** |

## Positive-Regime Results

### Small-LI

| Regime | TREFIC qualification | False qualification | Practical failure | Conditional median Delta F1 | Median oracle coverage |
|---|---:|---:|---:|---:|---:|
| IID | 1.0000 | 0.0000 | 0.0000 | 0.55981 | 1.0000 |
| Entity | 1.0000 | 0.0000 | 0.0000 | 0.45131 | 1.0000 |
| Temporal | 0.7031 | 0.0521 | 0.0573 | 0.49220 | 1.0000 |
| Graph-time | 0.8438 | 0.0000 | 0.0000 | 0.46440 | 1.0000 |

### Large-LI

| Regime | TREFIC qualification | False qualification | Practical failure | Conditional median Delta F1 | IID paired-F1 qualification |
|---|---:|---:|---:|---:|---:|
| IID | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.3333 |
| Entity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.3333 |
| Temporal | 0.0417 | 0.0156 | 0.0156 | 0.05632 | 0.4115 |
| Graph-time | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.3490 |

The Large-LI score generators still contained practical signal. For example,
the add candidate achieved certification-fold practical Delta F1 in 128/192
IID and entity trials, 106/192 temporal trials, and 110/192 graph-time trials.
TREFIC nevertheless rejected almost all of them because the graph/time lower
bound at an envelope endpoint remained negative.

Large-LI Wilson upper endpoints ranged from 0.0334 to 0.0687. The observed
worst endpoint lower bounds for add policies were:

| Regime | Median worst LCB | Maximum worst LCB |
|---|---:|---:|
| IID | -0.001822 | -0.000599 |
| Entity | -0.001814 | -0.000543 |
| Temporal | -0.003298 | +0.000279 |
| Graph-time | -0.002214 | -0.000147 |

This is a power limitation caused by sparse positive corrections and
graph/time dependence, not an absence of candidate-level F1 signal.

## Negative-Regime Safety

All three exposed negative units occurred on Large-LI:

| Regime | Row-net false qualification | GTPRC false qualification | GTF1C false qualification | TREFIC false qualification |
|---|---:|---:|---:|---:|
| CPSE remove fragility | 0.3125 | 0.2917 | 0.0000 | 0.0000 |
| Alignment drift | 0.2865 | 0.2656 | 0.0000 | 0.0000 |
| Rare duplicate | 0.2865 | 0.2656 | 0.0000 | 0.0000 |

TREFIC, shuffled TREFIC, and harmful TREFIC qualification were zero in every
negative unit. This confirms the intended safety mechanism but does not
compensate for the loss of positive power.

## Interpretation

### Supported

1. The exact directional identity correctly describes the sign of F1 change.
2. A zero lower endpoint prevents unsupported remove intervention when the
   future base sensitivity ratio may collapse.
3. Endpoint graph/time certification removes the negative failures that defeat
   row-count methods.
4. The same rule is usable on Small-LI.

### Not supported

1. TREFIC is not a usable cross-scale evidence intervention method.
2. It does not retain Large-LI IID power.
3. It does not justify a fresh formal run.
4. It cannot support a predictive FraudGT improvement claim.

## Research Consequence

The accumulated CET, CPSE, GTF1C, and TREFIC evidence now points to a broader
problem:

> Under dynamic random fraud-graph sampling, rare-positive F1 comparison may
> be non-identifiable or severely underpowered at the fold level even when an
> intervention has real local utility.

The next research object should therefore be evaluation reliability under
dynamic graph sampling, with an explicit abstention or "insufficient
information" outcome. It should not be another decoder, router, or relaxed
TREFIC threshold.

## Reproducibility

- Seven distinct GPU tasks used GPUs 0 through 6.
- Each task used a different registered regime and seed `75001` through
  `75007`; no duplicate jobs were run.
- All seven manifests used one remote Git commit.
- All source manifests reported `sampling_protocol=dynamic_random`.
- Validation and test loaders were never opened.
- Seven manifests and 2,688 trials completed without runtime error.
- Full machine-readable aggregation is stored in
  `TREFIC_PHASE0_DEVELOPMENT_RESULTS.json`.
