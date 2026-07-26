# GTPRC Phase 0A v2b Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: development screen completed; failed
- Remote commit: `e7b04bec`
- Output:
  `/e/yky/FraudGT_cet_results/gtprc_phase0a_v2b_dev_e7b04bec`
- Manifests: 7/7
- Replicates: 64 per regime, 448 total
- Runtime or CUDA failures: none

## Gate

v2b passed control rejection, method differentiation, GTPRC harm control, and
IID retention. It failed:

- row-IID violation in at least three dependent regimes: `0/6`;
- GTPRC oracle fraction at least 0.30 in five regimes.

Decision: `REDESIGN_STRESS_TEST`.

## Failure Diagnosis

| Regime | Row-IID median coverage | Row-IID mean test harm | GTPRC median coverage | GTPRC mean test harm |
|---|---:|---:|---:|---:|
| IID | 0.531 | 0.259 | 0.531 | 0.259 |
| Entity cluster | 0.620 | 0.258 | 0.305 | 0.127 |
| Temporal autocorrelation | 0.641 | 0.264 | 0.329 | 0.132 |
| Entity and temporal | 0.626 | 0.261 | 0.113 | 0.043 |
| Prevalence drift | 0.622 | 0.254 | 0.327 | 0.133 |
| Alignment drift | 0.276 | 0.280 | 0.000 | 0.019 |
| Rare positive and duplicate | 0.550 | 0.184 | 0.117 | 0.037 |

The 64-policy grid and hidden-cluster removal make GTPRC more conservative,
but row-IID still remains far below `alpha=0.40`. Its selected policy is
limited by the positive-net-utility requirement before the harm upper bound
becomes active.

## Adaptive Conclusion

Do not remove the positive-net-utility condition merely to manufacture a
row-IID failure. That would validate a safety method on policies that are
knowingly useless.

A final v2c development generator may increase the base rate of correctable
events so that:

- full or high-coverage intervention remains slightly harmful;
- a positive-net-utility region exists near the harm boundary;
- aligned scores still identify safer and more corrective cases;
- shuffled and harmful controls remain invalid.

All method formulas, dependence groups, `alpha`, `delta`, and development
gates remain unchanged. If this targeted construct still fails, stop the
synthetic route instead of continuing generator tuning.

