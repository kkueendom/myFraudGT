# COSTAR Evidence Diagnostics Plan

## Material Passport

- Type: Code experiment plan and validation protocol
- Research question: Does higher-order evidence independently correct A2
  errors, or is prediction dominated by encoder-derived `z_base`?
- Model commit: `4abe58c6`
- Diagnostic branch: `analysis/costar-evidence-diagnostics`
- Sampling protocol: `dynamic_random`
- Primary metric: Val-selected Test F1
- Status: implementation pending server verification

## Hypotheses

- H1 (independent information): evidence-only F1 is nontrivial, evidence is
  right on a meaningful subset of A2 errors, and shuffling evidence reduces F1.
- H0 (base-path dominance): evidence-only is weak, evidence breaks at least as
  many A2-correct decisions as it fixes, or normal/shuffled/off predictions are
  effectively equivalent.

F1 improvement by itself does not reject H0 because dynamic sampling can move
the reported score without the COSTAR correction causing that movement.

## Paired Design

One validation loader pass and one test loader pass are collected with the
original `LinkNeighborLoader(shuffle=True)` path. Every diagnostic variant is
then computed from exactly the same sampled examples:

1. `z_base`: encoder-derived decoder margin only.
2. `a2_anchor`: `z_base` plus the A2 prototype residual.
3. `costar_full`: A2 anchor plus COSTAR correction.
4. `evidence_only_total`: full margin minus `z_base`.
5. `evidence_only_prototype`: A2 anchor minus `z_base`.
6. `evidence_only_costar`: COSTAR correction alone.
7. `evidence_off`: identical to `z_base`.
8. `evidence_shuffled`: unchanged `z_base` plus a cross-sample permutation of
   the total evidence margin.

Evidence-only variants select their own validation threshold because their
scale differs from normal logits. The direct normal/shuffled/off test also uses
one shared threshold selected by the normal full model, so threshold retuning
cannot hide damage from shuffling.

## Required Outputs

### Evidence-only

Report validation and test F1 for total evidence, prototype evidence, and the
COSTAR correction alone.

### A2 error subset

Using separately validation-calibrated A2 and total-evidence predictions,
report:

- A2 wrong / evidence right;
- A2 right / evidence wrong;
- both wrong;
- both right;
- correction rate among A2 errors;
- damage rate among A2-correct samples;
- corrected minus broken count.

### Contribution audit

Report quantiles for absolute `z_base`, prototype delta, COSTAR delta and total
evidence delta; prototype alpha/readiness/reliability; COSTAR consistency and
residual weight; contribution-to-base ratios; changed decisions; corrected and
broken decisions; and gradient norms over 16 fresh training batches.

COSTAR has no hard inference gate. Its historical `fallback_ratio` is only a
diagnostic condition: low-confidence samples are not actually switched back to
A2. The report therefore distinguishes hard-gate presence, nontrivial
correction rate, confidence-pass rate and diagnostic-open rate.

### Shuffled evidence

Report normal, shuffled and off Test F1 on the same samples and shared normal
threshold. If normal minus shuffled and normal minus off are negligible, the
current evidence is not causally supporting the prediction.

## Decision Gate

- Metrics pass only if Small-LI is at least `0.46747` and Large-LI is at least
  `0.30608` on Val-selected Test F1.
- Mechanism passes only if evidence-only is useful, net correction is positive,
  normal evidence beats shuffled/off, and the evidence branch receives
  nonzero gradients and changes a non-negligible set of decisions.
- Metrics pass + mechanism fail: treat as sampling/base-path effect and repeat
  the same seed before any claim.
- Mechanism pass + metrics fail: preserve evidence but move to
  representation-level fusion and an explicit A2-error router.
- Both fail: stop adding decoder residuals and search for independent temporal,
  role-transition, cycle, flow-return or topology-anomaly evidence.

## Reproducibility

The queue waits for epoch 499 and `499.ckpt`, uses GPU4/GPU6 only, never touches
GPU0, and does not launch another model training run. It copies the final
checkpoint and config into the diagnostic artifact directory. Existing COSTAR
training used `train.ckpt_best=False`; if the exact selected epoch was not a
periodic checkpoint, the queue records that the exact best-val state is
unavailable rather than mislabeling a nearby checkpoint.
