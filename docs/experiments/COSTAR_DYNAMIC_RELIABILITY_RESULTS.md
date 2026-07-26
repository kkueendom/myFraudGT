# COSTAR Fixed-Checkpoint Dynamic Reliability Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: ANALYZED
- Status: completed
- Sampling protocol: `dynamic_random`
- Training or tuning: none
- Audit commit: `8d95c17a`
- COSTAR model commit: `4abe58c6`
- Streams: 2 per dataset
- Events: 8 per stream, 32 total
- Val/test iterations: 256 per event
- Source root: `/Users/kun/FraudGT_experiment_workspace/costar_reliability_formal_8d95c17`
- Remote root: `/e/yky/FraudGT_cet_results/costar_reliability_formal_8d95c17`
- Formal baseline: registered historical initial A2

## Conclusion

COSTAR does not supply a third useful evidence mechanism:

- Small-LI is `inactive_evidence`;
- Large-LI is `used_but_unaligned`;
- useful-aligned datasets: `0/2`;
- mean prediction changes per event: `2.56` and `0.75`;
- mean same-batch COSTAR-versus-anchor F1 deltas: `-0.00083` and
  `-0.00035`.

The historical Large-LI delta is positive (`+0.03641` mean), but the
same-batch anchor comparison is negative and its repeated-event range crosses
zero. The apparent historical gain cannot be attributed to the COSTAR
correction.

## Protocol Audit

All four tasks use `shuffle=True`, `val.fixed_target_panel=False`, no dedicated evaluation generator, no sampler RNG restoration, no evaluation step cap, and exactly 2,048 validation plus 2,048 test iterations per stream. Every normal/shuffled/off/A2-anchor comparison has an identical paired sample count.

The queue completed 4/4 manifests and 32/32 events with no runtime, alignment
or CUDA errors.

## Aggregate Results

| Dataset | Normal F1 mean +/- sd | A2 anchor mean +/- sd | Anchor delta mean | Historical A2 | Historical delta mean | Normal-shuffled | Normal-off | Changed mean | Corrected-broken mean | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small-LI | 0.42763 +/- 0.02005 | 0.42846 +/- 0.01976 | -0.00083 | 0.46247 | -0.03484 | -0.00094 | -0.00294 | 2.56250 | 0.31250 | `inactive_evidence` |
| Large-LI | 0.33749 +/- 0.06557 | 0.33783 +/- 0.06624 | -0.00035 | 0.30108 | 0.03641 | 0.00028 | 0.01582 | 0.75000 | -0.25000 | `used_but_unaligned` |

The A2 anchor is the same-batch pre-COSTAR margin. Comparing normal to this anchor isolates the newly trained COSTAR correction. The off condition removes all evidence, including the prototype contribution, so normal-off alone cannot be attributed to the COSTAR router.

## Contribution Scale

| Dataset | Mean absolute COSTAR delta | Median COSTAR/base ratio | Applied correction rate | Positive anchor delta | Corrected > broken |
|---|---:|---:|---:|---:|---:|
| Small-LI | 0.00103 | 0.00003 | 0.98102 | 6/16 | 5/16 |
| Large-LI | 0.00004 | 0.00000 | 0.00873 | 4/16 | 4/16 |

## Interpretation

1. Normal-shuffled/off evaluates total evidence sensitivity; normal-anchor evaluates the incremental COSTAR correction.
2. A total-evidence gap with negligible normal-anchor change means the prototype/base path, not COSTAR, carries the observed effect.
3. The 16 events per dataset come from two dynamic streams and one fixed checkpoint. They are descriptive repeated evaluations, not independent model seeds.
4. The registered historical A2 remains the formal baseline; the same-batch anchor is a mechanism diagnostic.

## Decision Gate

- Useful-aligned datasets: 0/2
- Sensitive-but-harmful datasets: 0/2
- Used-but-unaligned datasets: 1/2
- Inactive datasets: 1/2
- Datasets with a single-event conclusion reversal: 2/2

Final paper-level interpretation must combine this fixed-checkpoint block with A2, CET and TIER rather than selecting one favorable dataset.

Decision: `CONFIRM_STOP_COSTAR`. Do not tune or expand its router, prototype,
residual or decoder coefficients.
