# GTPRC Phase 0A v1 Controlled-Dependence Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: completed; construct-validity gate failed
- Remote experiment commit: `0890cee0`
- Local implementation commit: `69fb83f`
- Formal output:
  `/e/yky/FraudGT_cet_results/gtprc_phase0a_formal_0890cee0`
- Local output:
  `/Users/kun/FraudGT_experiment_workspace/gtprc_phase0a_formal_0890cee0`
- Manifests: 7/7
- Replicates: 512 per regime, 3,584 total
- GPUs: 7 distinct RTX 2080 Ti tasks
- Runtime errors: none
- AML validation/test loaders opened: no

## Registered Gate

The machine aggregate reports `PROCEED_TO_PHASE0B`:

- violation gate: pass;
- shuffled/harmful control gate: pass;
- oracle-coverage gate: pass;
- nonuniform-dominance gate: pass.

This machine gate is preserved in
`GTPRC_PHASE0A_V1_RESULTS.json`. It is not overwritten after seeing the
results.

## Construct-Validity Audit

The registered gate is insufficient because every compared method returned
exactly the same decision in every regime:

| Regime | Qualification rate | Violation rate | Median coverage | Median oracle fraction |
|---|---:|---:|---:|---:|
| IID | 1.000 | 0.000 | 0.4998 | 1.000 |
| Entity cluster | 1.000 | 0.000 | 0.4997 | 1.000 |
| Temporal autocorrelation | 1.000 | 0.000 | 0.5014 | 1.000 |
| Entity and temporal | 1.000 | 0.000 | 0.5009 | 1.000 |
| Prevalence drift | 1.000 | 0.000 | 0.4997 | 1.000 |
| Alignment drift | 1.000 | 0.000 | 0.4995 | 1.000 |
| Rare positive and duplicate | 1.000 | 0.000 | 0.4982 | 1.000 |

For every row above, row-IID, time-block, entity-block, and GTPRC have the
same qualification, violation, coverage, oracle fraction, and net correction.

## Failure Cause

The candidate quantile grid starts at 0.50. Consequently, the largest
available intervention coverage is about 50%. That policy is comfortably
below the harm limit for all generated regimes. Dependence adjustment never
becomes decision-relevant, so the experiment cannot test whether GTPRC
protects against anti-conservative row-IID calibration.

The shuffled and harmful controls are rejected, but this only validates the
score-alignment part of the generator. It does not validate the proposed
dependence method.

## Decision

`STOP_GTPRC_V1_CONSTRUCT_TOO_EASY`

Do not start CPSE Phase 0B from this result. A method paper cannot cite a
stress test in which all methods tie as evidence that the proposed method is
needed or superior.

The adaptive v2 experiment must be preregistered before execution and must:

1. expose policies from near-zero to near-total coverage;
2. place at least part of the policy frontier near the harm boundary;
3. reduce the number of independent graph-time units while retaining many
   correlated rows;
4. require row-IID to violate the bound while GTPRC controls it in multiple
   dependent regimes;
5. retain an IID regime where dependence adjustment does not create a large
   unnecessary coverage penalty.

