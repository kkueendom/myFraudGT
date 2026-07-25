# CET-FraudGT Phase B Results and Stop Decision

## Material Passport

- Origin: academic-research-suite / experiment-agent
- Status: completed, architecture stopped
- Primary commit: `cd72ba5b`
- Diagnostic-ablation commit: `a64eb631` (preregistered but not executed)
- Sampling protocol: `dynamic_random`
- Baseline: initial A2, never Fixed-panel A2
- Primary metric: Val-selected Test F1
- Raw result archive:
  `/Users/kun/FraudGT_experiment_workspace/cet_phaseb_cd72ba5`
- Remote result archive:
  `/e/yky/FraudGT_cet_results/phaseb_cd72ba5`

## Primary Results

| Dataset | Variant | Epochs | Val-selected Test F1 | Delta vs A2 | Raw-best Test F1 | Delta vs A2 |
|---|---|---:|---:|---:|---:|---:|
| Small-LI | encoder-only | 80 | 0.05290 | -0.40957 | 0.06695 | -0.43972 |
| Small-LI | fusion | 32 | 0.28704 | -0.17543 | 0.28704 | -0.21963 |
| Large-LI | encoder-only | 36 | 0.09742 | -0.20366 | 0.12992 | -0.31728 |
| Large-LI | fusion | 72 | 0.31868 | +0.01760 | 0.39474 | -0.05246 |

The Small-LI fusion result is far below the registered advancement threshold
of 0.46747. Large-LI exceeds its Val-selected threshold of 0.30608, but the
same model loses on Raw-best and fails the mechanism gates below. The average
Val-selected fusion delta is -0.07892.

## Counterfactual and Intervention Audit

Val-selected events:

| Dataset | Variant | Normal F1 | Shuffled F1 | Off F1 | Normal-shuffled | Changed | Corrected | Broken |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Small-LI | encoder-only | 0.05290 | 0.00630 | 0.00617 | +0.04660 | 1,352 | 69 | 1,283 |
| Small-LI | fusion | 0.28704 | 0.28483 | 0.00000 | +0.00221 | 253 | 26 | 227 |
| Large-LI | encoder-only | 0.09742 | 0.01042 | 0.00309 | +0.08700 | 274 | 31 | 243 |
| Large-LI | fusion | 0.31868 | 0.02448 | 0.30526 | +0.29420 | 17 | 5 | 12 |

Interpretation:

1. The independent history encoder contains aligned-history signal because
   both encoder-only models degrade strongly under shuffling, but it is a very
   poor global fraud classifier and breaks many A2-correct predictions.
2. Small-LI fusion does not depend on aligned history: normal and shuffled
   differ by only 0.00221 F1, below the registered 0.01 mechanism gate.
3. Large-LI fusion is sensitive to shuffling, but normal and off differ by only
   0.01342 F1. It changes only 17 predictions and breaks more than it corrects.
4. Nonzero fusion-gain norms are insufficient evidence of useful fusion.
   Small-LI mean fusion-gain norm is 1.67845 and Large-LI is 2.12028, yet the
   decision-level outcomes fail.

## OOF Complementarity Signal

The training trajectories show a severe cross-scale supervision imbalance:

- Small-LI exposes roughly 61,400-62,300 OOF-covered samples per training
  epoch, but only 14-40 are OOF A2 errors.
- Large-LI exposes roughly 114-149 OOF-covered samples per training epoch, and
  only 0-2 are OOF A2 errors.

Consequently, the registered OOF-error loss is sparse on Small-LI and nearly
inactive on Large-LI. This is not a coefficient problem: the supervision event
needed by the claimed complementarity mechanism is absent from most updates.

## Structural Diagnosis

CET v1 does not yet implement the intended temporal-subgraph hypothesis
strongly enough:

- it applies a Transformer to selected incident-event tokens without explicit
  temporal positional encoding;
- it does not preserve a connected transaction-path graph inside the evidence
  encoder;
- path, relay and cycle information enters mainly as precomputed token flags
  rather than learned path structure;
- a newly trained fusion classifier replaces the frozen A2 decision function,
  so base-retention is only an auxiliary off-history objective and cannot
  prevent widespread damage to A2-correct cases;
- the counterfactual loss distinguishes aligned from unrelated history in one
  scale, but does not establish error-complementary utility on both scales.

The observed behavior therefore repeats the earlier TIER conclusion:
history context is informative, but global replacement or unconstrained fusion
is unsafe.

## Advancement Gate

| Registered requirement | Small-LI fusion | Large-LI fusion | Pass |
|---|---:|---:|---|
| Val-selected threshold | 0.28704 >= 0.46747 | 0.31868 >= 0.30608 | No |
| Average Delta > 0 | \- | mean -0.07892 | No |
| Normal-shuffled >= 0.01 | +0.00221 | +0.29420 | No |
| Normal-off >= 0.005 | +0.28704 | +0.01342 | Yes |
| Changed >= 50 | 253 | 17 | No |
| Corrected > broken | 26 > 227 | 5 > 12 | No |
| Nonzero fusion gain | 1.67845 | 2.12028 | Yes |

The two-scale Phase B gate fails decisively.

## Stop Decision

`STOP_CET_V1`.

Do not:

- expand CET v1 to the other four datasets;
- run additional seeds;
- tune objective coefficients or thresholds;
- run the preregistered `no_oof`, `no_counterfactual` or
  `no_base_retention` loss ablations;
- revive decoder gates, residuals, prototypes, COSTAR, ordinary routers or
  CrossFusion variants.

The unused diagnostic-ablation commit remains in history for traceability, but
its queue is cancelled before execution because the full model did not pass
the primary two-scale feasibility gate.

## Research Pivot

The next research question is:

> Under dynamic fraud-graph sampling, when is causal temporal evidence
> reliable enough to improve a strong graph-transformer prediction without
> damaging its correct decisions?

The next phase must first characterize reliability rather than introduce
another fusion head. It should measure error-subset coverage, aligned-history
utility, sampling stability and support-conditioned failure modes on train-only
OOF data before any new predictive architecture is allowed.

