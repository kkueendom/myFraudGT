# COSTAR Evidence Diagnostics Results

## Material Passport

- Type: Experiment validation report
- Verification Status: ANALYZED
- Model commit: `4abe58c6`
- Diagnostic commit: `4885a8b5`
- Sampling protocol: `dynamic_random`
- Baseline: initial A2 only; Fixed-panel results excluded
- Primary metric: Val-selected Test F1
- Diagnostic checkpoint: final `499.ckpt`

## Conclusion

COSTAR does not provide evidence that its new decoder branch caused the
Large-LI improvement. The branch receives nonzero gradients, but its learned
margin correction is almost zero relative to `z_base` and changes almost no
decisions. Shuffling evidence leaves Large-LI unchanged and changes Small-LI
only at the dynamic-sampling scale.

Prototype evidence has weak complementary information: evidence-only F1 is
nonzero and it corrects some A2 errors. However, the trained prototype residual
is almost a constant logit shift (`0.3970` on Small-LI and `0.1902` on
Large-LI), while `prototype_ready` and reliability are always one. This is not
a convincing sample-specific higher-order evidence mechanism.

Decision: do not expand COSTAR and do not tune more gate/residual/prototype
coefficients. Preserve the weak evidence signal and test representation-level
fusion with an explicit A2-error router.

## Formal Results

These are complete 500-epoch dynamic-random runs compared by matching metric
to the specified initial A2 table.

| Dataset | COSTAR Val-selected | A2 Val-selected | Delta | Status | COSTAR Raw-best | A2 Raw-best | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| AML Small-LI | 0.45833 | 0.46247 | -0.00414 | possible sampling variation | 0.51509 | 0.50667 | +0.00842 | gain >= 0.005 |
| AML Large-LI | 0.34965 | 0.30108 | +0.04857 | gain >= 0.005 | 0.54438 | 0.44720 | +0.09718 | gain >= 0.005 |
| Pair mean | 0.40399 | 0.38178 | +0.02222 | 1/2 Val wins | 0.52974 | 0.47694 | +0.05280 | 2/2 Raw wins |

The first-round gate required both Val-selected deltas to be at least `+0.005`.
COSTAR fails because Small-LI needed `0.46747` and reached `0.45833`.

The original audit script can print partial rows as `completed`; that summary
was not used here. Only rows with `last_epoch=499` are formal.

## Small-LI Diagnostics

| Variant | Val F1 | Test F1 |
|---|---:|---:|
| `z_base` | 0.31175 | 0.37576 |
| A2 anchor | 0.31175 | 0.37576 |
| COSTAR full | 0.31175 | 0.37903 |
| Evidence-only total | 0.24274 | 0.31193 |
| Evidence-only prototype | 0.27792 | 0.35043 |
| Evidence-only COSTAR | 0.00000 | 0.00000 |
| Evidence off, shared threshold | n/a | 0.37121 |
| Evidence shuffled, shared threshold | n/a | 0.37600 |

| Error-subset quantity | Count |
|---|---:|
| A2 wrong / evidence right | 35 |
| A2 right / evidence wrong | 26 |
| Both wrong / both right | 274 / 519,369 |
| Net corrected minus broken | +9 |
| COSTAR decisions changed vs A2 | 4 / 519,704 |
| COSTAR changed corrected / broken | 4 / 0 |
| COSTAR median absolute delta | 0.000779 |
| `z_base` median absolute margin | 18.7831 |
| Median `|COSTAR delta|/|z_base|` | 0.0000342 |
| Normal minus shuffled / off | +0.00303 / +0.00782 |

## Large-LI Diagnostics

| Variant | Val F1 | Test F1 |
|---|---:|---:|
| `z_base` | 0.18182 | 0.33750 |
| A2 anchor | 0.18182 | 0.33750 |
| COSTAR full | 0.18182 | 0.33750 |
| Evidence-only total | 0.16000 | 0.31655 |
| Evidence-only prototype | 0.17680 | 0.29293 |
| Evidence-only COSTAR | 0.00244 | 0.00306 |
| Evidence off, shared threshold | n/a | 0.33880 |
| Evidence shuffled, shared threshold | n/a | 0.33750 |

| Error-subset quantity | Count |
|---|---:|
| A2 wrong / evidence right | 16 |
| A2 right / evidence wrong | 5 |
| Both wrong / both right | 90 / 72,361 |
| Net corrected minus broken | +11 |
| COSTAR decisions changed vs A2 | 0 / 72,472 |
| COSTAR median absolute delta | 0.0000133 |
| `z_base` median absolute margin | 16.5040 |
| Median `|COSTAR delta|/|z_base|` | 0.000000806 |
| Normal minus shuffled / off | 0.00000 / -0.00130 |

## Gradient and Contribution Interpretation

The router was not dead. Across 16 fresh train batches, gradients were
nonzero in every batch:

| Dataset | Mean COSTAR-router grad L2 | Mean z-base decoder grad L2 | Mean adapter loss |
|---|---:|---:|---:|
| Small-LI | 0.01638 | 0.25306 | 0.04600 |
| Large-LI | 0.000281 | 0.20414 | 0.04918 |

The failure is downstream of gradient flow. Large-LI has diagnostic open rate
`0.00840`, COSTAR changes zero A2 decisions, and its median correction/base
ratio is `8.06e-7`. Small-LI has open rate `0.98105`, but still changes only
four decisions. COSTAR has no hard deployment gate; its fallback statistic is
diagnostic only and does not switch predictions back to A2.

## Caveats

- Deltas below `0.005` are marked as possible dynamic-sampling variation.
- Diagnostic F1 is mechanism evidence, not the formal metric table; it uses
  one dynamic validation/test pass per dataset.
- Evidence-only and shuffled variants are exploratory comparisons; no p-value
  or confidence interval is claimed.
- Error-subset counts are sampled instances, not a fixed target-edge panel.
- Selected formal epochs were Small `211` and Large `361`. The original run had
  `train.ckpt_best=False`, so exact best-val checkpoints were not persisted.
  The artifact directory records this limitation and preserves final `499.ckpt`
  plus the config.

## Next Experiment

The weak evidence signal should be tested before the final logit:

\[
h_{evi}=\operatorname{EvidenceEncoder}(r_{raw}),\quad
h_{fused}=\operatorname{CrossFusion}(h_{base},h_{evi},support)
\]
\[
q=\operatorname{ErrorRouter}(h_{base},h_{evi},|h_{base}-h_{evi}|,support),\quad
z_{final}=(1-q)z_{base}+qz_{fused}.
\]

Use direct raw motif/time/role/local-subgraph/support inputs, evidence dropout,
and shuffled-evidence negatives. Train the router only with train labels using
detached A2 error targets and class balancing; validation/test labels must not
enter router state. First screen Small-LI seed 42 and Large-LI seed 44 under
dynamic_random, recording the same four diagnostics before any 500-epoch
confirmation. Both datasets must exceed A2 by `0.005` on Val-selected F1.
