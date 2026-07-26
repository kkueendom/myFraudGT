# GTPRC Phase 0A v2c Formal Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: formal controlled simulation completed; gate passed
- Remote experiment commit: `d1fcaf9d`
- Formal output:
  `/e/yky/FraudGT_cet_results/gtprc_phase0a_v2c_formal_d1fcaf9d`
- Local output:
  `/Users/kun/FraudGT_experiment_workspace/gtprc_phase0a_v2c_formal_d1fcaf9d`
- Manifests: 7/7
- Replicates: 512 per regime, 3,584 total
- GPUs: seven distinct RTX 2080 Ti tasks
- Runtime, CUDA, or manifest errors: none
- Validation/test labels opened: no
- Machine aggregate:
  `docs/experiments/GTPRC_PHASE0A_V2C_FORMAL_RESULTS.json`

## Formal Gate

| Requirement | Result |
|---|---|
| Row-IID violation greater than 0.10 in at least three dependent regimes | pass |
| GTPRC violation at most 0.07 in all regimes | pass |
| GTPRC oracle fraction at least 0.30 in five regimes | pass |
| IID coverage retention at least 0.80 | pass |
| Shuffled and harmful controls rejected in all regimes | pass |
| Median coverage differs in at least three dependent regimes | pass |
| GTPRC lowers violation by at least 0.05 in three dependent regimes | pass |

Decision: `PROCEED_TO_PHASE0B`.

## Primary Results

| Regime | Row-IID violation | GTPRC violation | Row-IID median coverage | GTPRC median coverage | GTPRC oracle fraction | GTPRC mean test harm |
|---|---:|---:|---:|---:|---:|---:|
| IID | 0.0000 | 0.0000 | 0.8434 | 0.8370 | 0.9505 | 0.3724 |
| Entity cluster | 0.2148 | 0.0000 | 0.9181 | 0.4775 | 0.5153 | 0.1396 |
| Temporal autocorrelation | 0.2461 | 0.0000 | 0.9202 | 0.4760 | 0.5178 | 0.1397 |
| Entity and temporal | 0.2578 | 0.0000 | 0.9238 | 0.2582 | 0.2799 | 0.0507 |
| Prevalence drift | 0.1367 | 0.0000 | 0.9161 | 0.4707 | 0.5140 | 0.1375 |
| Alignment drift | 0.2461 | 0.0000 | 0.8386 | 0.0235 | 0.0267 | 0.0763 |
| Rare positive and duplicate | 0.0352 | 0.0000 | 0.9102 | 0.3004 | 0.3285 | 0.0435 |

Shuffled and harmful false qualification is `0/512` in every regime.

## Interpretation

The controlled result establishes three properties:

1. treating correlated rows as IID can select near-boundary policies whose
   evaluation harm exceeds `alpha=0.40`;
2. using the true graph-time dependency unit makes policy selection more
   conservative and controls harm in these registered regimes;
3. when rows are IID, dependency grouping does not create a material coverage
   penalty.

The result also exposes the expected power trade-off. In combined dependency
and alignment-drift regimes, GTPRC retains only 0.28 and 0.03 of oracle
coverage. Dependence validity does not guarantee useful intervention
coverage.

## Claim Boundary

This experiment does **not** show:

- improved FraudGT F1;
- useful evidence in AML data;
- validity for arbitrary transaction graphs;
- a finite-sample theorem under unknown dependence;
- that CPSE will pass OOF qualification.

The aligned score is a controlled outcome-correlated proxy constructed to
create a known risk-coverage frontier. It is not a learned fraud score.

The grouped Hoeffding object targets an equal-group mean under the registered
independent-group generator. A paper-level graph guarantee still requires a
formal mapping from transaction dependencies to valid groups and a proof for
unequal, overlapping real graph units.

## Next Authorized Work

Phase 0B may now begin:

- construct CPSE using fraud-label-free causal next-event prediction;
- run six train-only OOF tasks on Small-LI and Large-LI;
- store source, destination, timestamp, edge ID, and dependency groups;
- qualify normal/shuffled/off evidence with locked GTPRC;
- do not instantiate validation/test loaders.

Failure on either scale stops CPSE and prohibits full-model training.

