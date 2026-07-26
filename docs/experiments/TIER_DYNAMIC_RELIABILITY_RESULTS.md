# TIER Fixed-Checkpoint Dynamic Reliability Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: ANALYZED
- Status: completed
- Sampling protocol: `dynamic_random`
- Training or tuning: none
- Audit commit: `32b3aab4`
- Streams: 2 per model/dataset pair
- Events: 8 per stream, 64 total
- Val/test iterations: 256 per event
- Source root: `/Users/kun/FraudGT_experiment_workspace/tier_reliability_formal_32b3aab`
- Remote root: `/e/yky/FraudGT_cet_results/tier_reliability_formal_32b3aab`
- Formal baseline: registered historical initial A2

## Conclusion

All four dataset/evidence pairs are **sensitive but harmful**. Normal evidence
beats shuffled evidence by `0.13113` to `0.31203` mean F1, so these fixed TIER
classifiers genuinely use causal transaction-history evidence. However, their
same-batch F1 is lower than frozen A2 by `0.12876` to `0.17209`.

Across all 64 events:

- positive same-batch TIER-versus-A2 F1 events: `0/64`;
- events with corrected predictions greater than broken predictions: `4/64`;
- useful-aligned model/dataset pairs: `0/4`;
- sensitive-but-harmful pairs by aggregate mean: `4/4`;
- pairs satisfying both sensitivity and harmful-utility conditions in at
  least 75% of events: `4/4`.

This confirms the central mechanism failure: evidence contains discriminative
information, but global evidence-only substitution is not aligned with A2
errors. It supports a reliability/evaluation research question; it does not
authorize another TIER router, decoder residual or CrossFusion training run.

## Protocol Audit

All eight tasks use `shuffle=True`, `val.fixed_target_panel=False`, no dedicated evaluation generator, no sampler RNG restoration, no evaluation step cap, and exactly 2,048 validation plus 2,048 test iterations per stream. Every normal/shuffled/off/frozen-A2 comparison has an identical paired sample count.

The queue completed 8/8 manifests and 64/64 events with no runtime, alignment
or CUDA errors.

## Aggregate Results

| Dataset | Evidence | Epoch | Normal F1 mean +/- sd | Frozen A2 mean +/- sd | Paired delta mean | Historical A2 | Historical delta mean | Normal-shuffled | Normal-off | Changed mean | Corrected-broken mean | Classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small-LI | recent | 39 | 0.31274 +/- 0.02035 | 0.44151 +/- 0.02288 | -0.12876 | 0.46247 | -0.14973 | 0.31041 | 0.31274 | 150.75000 | -64.50000 | `sensitive_but_harmful` |
| Small-LI | role_motif | 61 | 0.31516 +/- 0.02623 | 0.44571 +/- 0.02126 | -0.13055 | 0.46247 | -0.14731 | 0.31203 | 0.31516 | 163.06250 | -77.06250 | `sensitive_but_harmful` |
| Large-LI | recent | 79 | 0.13842 +/- 0.02305 | 0.31050 +/- 0.04821 | -0.17209 | 0.30108 | -0.16266 | 0.13113 | 0.13842 | 183.50000 | -91.87500 | `sensitive_but_harmful` |
| Large-LI | role_motif | 43 | 0.18290 +/- 0.04502 | 0.31961 +/- 0.03963 | -0.13671 | 0.30108 | -0.11818 | 0.18171 | 0.18160 | 176.31250 | -120.81250 | `sensitive_but_harmful` |

The frozen epoch-499 A2 score is a same-batch diagnostic only. The registered historical initial A2 remains the formal baseline because the available checkpoint epoch differs from the historical Val-selected epoch.

## Event Consistency

| Dataset | Evidence | Sensitivity >=0.01 | Positive utility | Corrected > broken | Sensitivity threshold crossed? | Utility sign crossed? |
|---|---|---:|---:|---:|---|---|
| Small-LI | recent | 16/16 | 0/16 | 0/16 | false | false |
| Small-LI | role_motif | 16/16 | 0/16 | 0/16 | false | false |
| Large-LI | recent | 16/16 | 0/16 | 3/16 | false | false |
| Large-LI | role_motif | 16/16 | 0/16 | 1/16 | false | false |

## Interpretation

1. A large normal-shuffled gap establishes that the fixed evidence classifier uses transaction-history evidence; it does not establish that replacing A2 is beneficial.
2. Corrective utility requires a positive same-batch F1 delta and more corrected than broken predictions. These quantities are reported separately from sensitivity.
3. The 16 events are repeated dynamic evaluations from two streams, not 16 independent model seeds. Mean and standard deviation are descriptive; no IID p-value is claimed.
4. A threshold or sign crossing means a one-event conclusion can reverse under dynamic sampling. It is not itself proof of a model effect.
5. Large-LI paired retention is reported explicitly because repeated loader requests can collapse to fewer unique A2-scored target edges. All retained targets remain strictly aligned across conditions.

## Decision Gate

- Useful-aligned pairs: 0/4
- Sensitive-but-harmful pairs: 4/4
- Used-but-unaligned pairs: 0/4
- Inactive pairs: 0/4
- Pairs with a single-event conclusion reversal: 0/4

Paper-level advancement remains conditional on the preregistered multi-model gate after combining A2, CET, TIER and COSTAR evidence.

TIER itself does not show a threshold or utility-sign reversal because its
harmful replacement effect is large and consistent. The broader claim that
dynamic sampling can reverse a single-run conclusion is supported by the
separate six-dataset A2 and CET audits, not by inventing instability in this
block.
