# GTPRC Phase 0A v2 Boundary-Stress Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent failure: `GTPRC_PHASE0A_V1_RESULTS.md`
- AML validation/test labels allowed: no
- Full FraudGT training allowed: no
- GPUs: seven distinct dependence regimes

## Adaptive Change From v1

v1 did not fail statistically; it failed construct validity. Its candidate
grid exposed at most 50% coverage, and that policy was safe for every method.
The dependence correction never affected a decision.

v2 changes the stress-test data-generating process, not the desired result:

1. candidate policies span approximately 100% to below 1% coverage;
2. unconditional evidence intervention is harmful enough that the full-
   coverage policy lies above `alpha=0.40`;
3. aligned scores create a gradual risk-coverage frontier rather than a
   perfectly separated outcome;
4. dependent regimes contain many rows but substantially fewer independent
   graph-time units;
5. the IID regime remains a negative control for unnecessary conservatism.

The method family remains fixed across regimes and receives the same
Bonferroni correction.

## v2 Group Bound

For candidate policy \(\lambda\), partition active interventions into
preregistered groups \(g=1,\ldots,m_\lambda\). Let \(\bar L_g(\lambda)\) be
the within-group break rate. The development implementation uses:

\[
\widehat R_{\mathrm{group}}(\lambda)
=
\frac{1}{m_\lambda}
\sum_{g=1}^{m_\lambda}\bar L_g(\lambda),
\]

\[
U_{\mathrm{group}}(\lambda)
=
\widehat R_{\mathrm{group}}(\lambda)
+
\sqrt{
\frac{\log(K/\delta)}
{2m_\lambda}
}.
\]

`K` is the number of candidate policies. Row-IID uses a Bonferroni-adjusted
Wilson upper bound. Time-only, entity-only, and graph-time methods differ only
in their grouping unit.

This bound targets the equal-group mean risk under independent groups. It is
not yet claimed to apply to arbitrary real transaction graphs. The controlled
experiment tests empirical validity before theory or AML application.

## Development Screen

Seven GPUs each run one regime with 64 replicates. These results only qualify
the simulator and implementation. They are not formal method results.

The screen passes only if:

- row-IID violation rate exceeds 0.10 in at least three of the six dependent
  regimes;
- GTPRC violation rate is at most 0.07 in all seven regimes;
- GTPRC median oracle-coverage fraction is at least 0.30 in at least five
  regimes;
- in IID, GTPRC median coverage is at least 80% of row-IID coverage;
- shuffled and harmful false qualification is at most 0.05 in every regime;
- at least three dependent regimes produce different selected coverage for
  row-IID and GTPRC.

Failure means the stress test is still non-identifying. Record the failure and
redesign before any formal run.

## Formal v2 Evaluation

Only after the development screen passes:

- use new formal seeds not used in development;
- run 512 replicates per regime, 3,584 total;
- preserve `alpha=0.40`, `delta=0.05`, 16 candidate policies, 8,192
  calibration rows, and 16,384 evaluation rows;
- use all seven GPUs concurrently;
- apply the same gates as the development screen;
- report qualification, violation, coverage, oracle fraction, net utility,
  and shuffled/harmful false qualification for all four methods.

Formal advancement additionally requires GTPRC violation to be at least 0.05
lower than row-IID in three dependent regimes. If the methods tie again, stop
GTPRC even if a weaker machine gate passes.

## Decision Boundary

- Development fail: `REDESIGN_STRESS_TEST`; do not run formal v2.
- Formal fail: `STOP_OR_REDESIGN_GTPRC`; do not start CPSE.
- Formal pass: proceed to CPSE train-only OOF feasibility, while treating the
  graph-dependence theorem as unfinished work.

