# CDVT Experiment Table Templates

Status date: 2026-07-31  
Frozen model: `dual_view`, `lambda_cons=0`, commit `9038f85`  
Phase 2 source commit: `df12ea6` (documentation-only descendant)  
Portable execution commit: `2fb3333` (tree-identical to `df12ea6`)  
Follow-up source commit: `34456ab`  
Phase 0 audit portable commit: `9be1737`  
Final self-contained follow-up portable commit: `f7209f2`  
Sampling protocol: `dynamic_random`

This document is the single paper-facing template for CDVT results. `TBD` means
that no authoritative manifest has been collected yet. Do not replace `TBD`
with a value copied from a log line; use the final experiment manifest or the
automatic summary output.

## Reporting Rules

1. Formal conclusions use val-selected test F1, where the checkpoint is chosen
   by validation F1.
2. PE-FraudGT is the primary published baseline because its Ports + Ego account
   view is the direct parent architecture used by CDVT.
3. Multi-FraudGT is the strongest published FraudGT reference; A2 is a strong
   internal comparator. Neither replaces PE-FraudGT as the formal gate.
4. Raw-best test F1 is supplementary and is compared only with the A2 raw-best
   column because the FraudGT paper does not report raw-best.
5. Every delta names its reference and is computed within the same metric
   column. Never compare raw-best with a published val-selected result.
6. Mark `abs(delta) < 0.005` as possible dynamic-sampling variation.
7. Phase 2 passes only when CDVT beats PE-FraudGT on at least 4/6 datasets and
   the six-dataset mean val-selected delta against PE-FraudGT is positive.
8. Multi-seed entries must come from independent seeds 42, 43, and 44. Never
   duplicate the best seed.

## Table 0. Phase 0 Engineering Qualification

This table verifies implementation behavior only. It is not a predictive
performance result.

| Check | Evidence | Result |
|---|---|---:|
| CDVT unit/integration suite | Remote `test_cdvt*.py` | 43/43 passed |
| Account path receives gradients | Gradient norm | 0.96155 |
| Event path receives gradients | Gradient norm | 0.17916 |
| Cross-view fusion receives gradients | Gradient norm | 0.21991 |
| Mixed-class dynamic batch | Fraud / normal targets | 1 / 866 |
| Fixed-batch trainability | Loss before / after 24 steps | 0.67856 / 0.04245 |
| Event coverage | Mean events per target | 9.47982 |
| Normal differs from shuffled | Mean absolute logit delta | 0.000977 |
| Normal differs from off | Mean absolute logit delta | 0.003887 |

Authoritative record:
`CDVT_Phase0_GPU_Smoke_Mixed_9be1737.json`, SHA-256
`262b235bf4c82c76f0d4859475b6b36f14419ae6c97ef6bda90d9d43c140d695`.
The batch is extremely imbalanced, so the overfit result proves only that the
end-to-end implementation can optimize a real mixed-class batch.

## Table 1. Phase 1 Model Freeze Evidence

Validation F1, not test F1, selected the frozen architecture.

| Variant | Small-LI validation F1 | Large-LI validation F1 | Two-scale mean |
|---|---:|---:|---:|
| CDVT dual-view | 0.37500 | 0.39695 | 0.38597 |
| CDVT dual-view + sampling consistency | 0.37264 | 0.34008 | 0.35636 |

Sampling consistency reduced validation F1 on both screening datasets and is
therefore excluded from the final CDVT model. The selected dual-view model had
val-selected test F1 values of 0.43798 on Small-LI and 0.33803 on Large-LI.
Their deltas against PE-FraudGT were -0.02012 and +0.03363, respectively, with
a positive two-scale mean delta of +0.00676. Against initial A2, the
corresponding deltas were -0.02449 and +0.03695, with mean +0.00623.

## Table 2. Six-Dataset Main Results, Seed 42

### Table 2a. Published FraudGT references

The FraudGT values are five-run means from Table 2 of the original paper.
CDVT seed-42 values are not a paired statistical comparison with those means;
formal stability is supplied separately by the three-seed table.

| Dataset | PE-FraudGT | CDVT val-selected | Delta vs PE | Interpretation | Multi-FraudGT | Delta vs Multi |
|---|---:|---:|---:|---|---:|---:|
| AML Small-LI | 0.45810 | 0.43798 | -0.02012 | Clear loss | 0.47010 | -0.03212 |
| AML Small-HI | 0.76410 | 0.76940 | +0.00530 | Clear gain | 0.76130 | +0.00810 |
| AML Medium-LI | 0.43530 | 0.44711 | +0.01181 | Clear gain | 0.44060 | +0.00651 |
| AML Medium-HI | 0.74220 | 0.77301 | +0.03081 | Clear gain | 0.75930 | +0.01371 |
| AML Large-LI | 0.30440 | 0.33803 | +0.03363 | Clear gain | 0.37430 | -0.03627 |
| AML Large-HI | 0.68640 | 0.78793 | +0.10153 | Clear gain | 0.73340 | +0.05453 |
| Mean | 0.56508 | 0.59224 | +0.02716 | 5 wins / 1 loss | 0.58983 | +0.00241 |

### Table 2b. Internal A2 reference

| Dataset | A2 val-selected | CDVT val-selected | Delta vs A2 | A2 raw-best | CDVT raw-best | Delta vs A2 |
|---|---:|---:|---:|---:|---:|---:|
| AML Small-LI | 0.46247 | 0.43798 | -0.02449 | 0.50667 | 0.47687 | -0.02980 |
| AML Small-HI | 0.77984 | 0.76940 | -0.01044 | 0.79497 | 0.79128 | -0.00369 |
| AML Medium-LI | 0.51163 | 0.44711 | -0.06452 | 0.59031 | 0.51029 | -0.08002 |
| AML Medium-HI | 0.77574 | 0.77301 | -0.00273 | 0.78940 | 0.80088 | +0.01148 |
| AML Large-LI | 0.30108 | 0.33803 | +0.03695 | 0.44720 | 0.44816 | +0.00096 |
| AML Large-HI | 0.72897 | 0.78793 | +0.05896 | 0.76223 | 0.79284 | +0.03061 |
| Mean | 0.59329 | 0.59224 | -0.00104 | 0.64846 | 0.63672 | -0.01174 |

Paper note: bold the better value only after all six manifests pass protocol
validation. Report wins, losses, mean delta, and the number of values within the
0.005 sampling-variation band separately for PE-FraudGT, Multi-FraudGT, and A2.

## Table 3. Representative-Dataset Independent Seeds

| Dataset | Seed | FraudGT val epoch | FraudGT test F1 | CDVT val epoch | CDVT test F1 | Paired val delta | FraudGT raw-best | CDVT raw-best | Paired raw delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AML Small-LI | 42 | 163 | 0.47893 | 75 | 0.43798 | -0.04094 | 0.47985 | 0.47687 | -0.00299 |
| AML Small-LI | 43 | 151 | 0.43515 | 75 | 0.48805 | +0.05291 | 0.46377 | 0.48805 | +0.02429 |
| AML Small-LI | 44 | 151 | 0.46320 | TBD | TBD | TBD | 0.47638 | TBD | TBD |
| AML Medium-LI | 42 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AML Medium-LI | 43 | 87 | 0.36598 | 39 | 0.36957 | +0.00359 | 0.48260 | 0.46113 | -0.02147 |
| AML Medium-LI | 44 | 95 | 0.37642 | TBD | TBD | TBD | 0.43882 | TBD | TBD |
| AML Large-LI | 42 | 99 | 0.21287 | 27 | 0.33803 | +0.12516 | 0.29126 | 0.44816 | +0.15690 |
| AML Large-LI | 43 | 167 | 0.28221 | 83 | 0.34909 | +0.06688 | 0.34659 | 0.48447 | +0.13788 |
| AML Large-LI | 44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 4. Three-Seed Mean and Standard Deviation

Use sample standard deviation across three independent runs and state
`mean +/- std` in the manuscript. FraudGT/account-only and CDVT must both use
seeds 42, 43, and 44; the primary robustness statistic is the same-seed paired
delta, not CDVT mean minus a historical point estimate.

| Dataset | FraudGT val mean +/- std | CDVT val mean +/- std | Paired delta mean +/- std | Delta vs PE | Delta vs Multi |
|---|---:|---:|---:|---:|---:|
| AML Small-LI | TBD | TBD | TBD | TBD | TBD |
| AML Medium-LI | TBD | TBD | TBD | TBD | TBD |
| AML Large-LI | TBD | TBD | TBD | TBD | TBD |

## Table 5. Core Architecture Ablation

These variants are not a simple cumulative ladder. `account_only` and
`event_only` are alternative single-view controls; `dual_view` combines both
views. The consistency variant is a tested but ineffective extension and should
remain supplementary rather than being presented as part of final CDVT.

| Variant | Account view | Causal event view | Fusion rule | Consistency loss | Small-LI F1 | Medium-LI F1 | Large-LI F1 | Mean |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| V0: FraudGT / account-only | Yes | No | None | No | 0.47893 | TBD | 0.21287 | TBD |
| V1: event-only | No | Yes | None | No | 0.33826 | TBD | 0.30530 | TBD |
| V2: additive dual-view | Yes | Yes | Addition | No | TBD | TBD | TBD | TBD |
| V3: final CDVT dual-view | Yes | Yes | Cross-attention | No | 0.43798 | 0.44711 | 0.33803 | 0.40771 |
| V4: dual-view + consistency | Yes | Yes | Cross-attention | Yes | TBD | TBD | TBD | TBD |

## Table 6. Event-Graph Mechanism Controls

| Event condition | Temporal order preserved | Relation types preserved | Event context active | Small-LI F1 | Medium-LI F1 | Large-LI F1 | Mean |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| Normal causal event graph | Yes | Yes | Yes | 0.43798 | 0.44711 | 0.33803 | 0.40771 |
| Shuffled event graph | No | Yes | Yes | 0.09350 | 0.12058 | 0.01596 | 0.07668 |
| Event graph off | N/A | N/A | No | 0.27294 | 0.03012 | 0.00000 | 0.10102 |
| No relation type | Yes | No | Yes | TBD | TBD | TBD | TBD |

Mechanism support requires normal causal events to outperform shuffled and off
controls on at least two representative datasets. The no-relation control
tests whether typed event transitions contribute beyond temporal connectivity.
The completed Small-LI and Large-LI manifests already satisfy the
normal-versus-shuffled/off mechanism criterion on two scales. Medium-LI remains
scheduled so the final table covers all three representative datasets.

## Table 7. History-Size Sensitivity

The frozen default is `K=4`; `K=2` is the single preregistered alternative.
The follow-up queue evaluates both settings on the three representative
datasets while reusing every default run.

| Dataset | K recent events per account | Val-selected test F1 | Delta vs K=4 | Parameters | Peak GPU memory | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| AML Small-LI | 4 | 0.43798 | 0.00000 | 301,147 | 1.97 GB | TBD |
| AML Small-LI | 2 | TBD | TBD | TBD | TBD | TBD |
| AML Medium-LI | 4 | 0.44711 | 0.00000 | 301,147 | 3.02 GB | TBD |
| AML Medium-LI | 2 | TBD | TBD | TBD | TBD | TBD |
| AML Large-LI | 4 | 0.33803 | 0.00000 | 301,147 | 7.11 GB | TBD |
| AML Large-LI | 2 | TBD | TBD | TBD | TBD | TBD |

## Table 8. Computational Cost

Measure both models under the same dataset, hardware, batch size, and sampling
budget. Inference time must come from the dedicated 256-batch normal-only
checkpoint benchmark. Do not use the three-condition normal/shuffled/off
evaluation duration as inference latency.

| Model | Parameters | Peak GPU memory | Training time / epoch | Total training time | Inference time / batch | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| FraudGT / account-only | 182,569 | TBD | TBD | TBD | TBD | TBD |
| CDVT dual-view | 301,147 | TBD | TBD | TBD | TBD | TBD |

## Table 9. Experiment Registry

Every formal result must have one row here or in an automatically generated
machine-readable registry.

| Dataset | Phase | Variant | Seed | Git commit | Config | Checkpoint | Epoch | Val-selected F1 | Raw-best F1 | Delta val vs PE | Delta val vs Multi | Delta val vs A2 | Delta raw vs A2 | Parameters | Peak memory | Runtime | Sampling protocol |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small-LI | Phase 1 | dual_view | 42 | `9c18cfdd` | `AML-Small-LI.yaml` | `Small-LI_dual_view_seed42/best_val.ckpt` | 75 | 0.43798 | 0.47687 | -0.02012 | -0.03212 | -0.02449 | -0.02980 | 301,147 | 1.97 GB | 111,866.76 s | dynamic_random |
| Small-HI | Phase 2 | dual_view | 42 | `2fb3333` | `AML-Small-HI.yaml` | `Small-HI_dual_view_seed42/best_val.ckpt` | 147 | 0.76940 | 0.79128 | +0.00530 | +0.00810 | -0.01044 | -0.00369 | 301,147 | 1.94 GB | 187,481.85 s | dynamic_random |
| Medium-LI | Phase 2 | dual_view | 42 | `2fb3333` | `AML-Medium-LI.yaml` | `Medium-LI_dual_view_seed42/best_val.ckpt` | 95 | 0.44711 | 0.51029 | +0.01181 | +0.00651 | -0.06452 | -0.08002 | 301,147 | 3.02 GB | 140,975.19 s | dynamic_random |
| Medium-HI | Phase 2 | dual_view | 42 | `2fb3333` | `AML-Medium-HI.yaml` | `Medium-HI_dual_view_seed42/best_val.ckpt` | 211 | 0.77301 | 0.80088 | +0.03081 | +0.01371 | -0.00273 | +0.01148 | 301,147 | 3.03 GB | 241,691.53 s | dynamic_random |
| Large-LI | Phase 1 | dual_view | 42 | `9c18cfdd` | `AML-Large-LI.yaml` | `Large-LI_dual_view_seed42/best_val.ckpt` | 27 | 0.33803 | 0.44816 | +0.03363 | -0.03627 | +0.03695 | +0.00096 | 301,147 | 7.11 GB | 67,763.30 s | dynamic_random |
| Large-HI | Phase 2 | dual_view | 42 | `2fb3333` | `AML-Large-HI.yaml` | `Large-HI_dual_view_seed42/best_val.ckpt` | 151 | 0.78793 | 0.79284 | +0.10153 | +0.05453 | +0.05896 | +0.03061 | 301,147 | 7.19 GB | 163,466.93 s | dynamic_random |
| Small-LI | Phase 3 | account_only | 43 | `f7209f2` | `AML-Small-LI-account_only-seed43.yaml` | `Small-LI_account_only_seed43/best_val.ckpt` | 151 | 0.43515 | 0.46377 | -0.02295 | -0.03495 | -0.02732 | -0.04290 | 182,569 | 1.81 GB | 11,148.45 s | dynamic_random |
| Medium-LI | Phase 3 | account_only | 43 | `f7209f2` | `AML-Medium-LI-account_only-seed43.yaml` | `Medium-LI_account_only_seed43/best_val.ckpt` | 87 | 0.36598 | 0.48260 | -0.06932 | -0.07462 | -0.14565 | -0.10771 | 182,569 | 3.00 GB | 13,942.30 s | dynamic_random |
| Small-LI | Phase 3 | account_only | 44 | `f7209f2` | `AML-Small-LI-account_only-seed44.yaml` | `Small-LI_account_only_seed44/best_val.ckpt` | 151 | 0.46320 | 0.47638 | +0.00510 | -0.00690 | +0.00073 | -0.03029 | 182,569 | 1.81 GB | 11,187.52 s | dynamic_random |
| Medium-LI | Phase 3 | account_only | 44 | `f7209f2` | `AML-Medium-LI-account_only-seed44.yaml` | `Medium-LI_account_only_seed44/best_val.ckpt` | 95 | 0.37642 | 0.43882 | -0.05888 | -0.06418 | -0.13521 | -0.15149 | 182,569 | 3.00 GB | 14,711.16 s | dynamic_random |

## Manuscript Claim Gate

The automatic protocol audit passed on 2026-07-31: CDVT achieved 5/6 wins
against PE-FraudGT and a mean val-selected delta of +0.02716. Follow-up
multi-seed, ablation, sensitivity, and runtime experiments are active.

- `PASS`: at least 4/6 val-selected wins against PE-FraudGT and positive mean
  delta against PE-FraudGT. Proceed to seeds 43/44, ablations, mechanism
  controls, K sensitivity, and cost analysis.
- `BORDERLINE`: exactly 3/6 wins or non-positive mean delta. Permit at most one
  validation-driven structural revision; do not return to decoder gates,
  residual logits, prototypes, support coefficients, or rule evidence.
- `FAIL`: fewer than 3/6 wins or an unexplained severe collapse. Diagnose event
  construction, relation encoding, and cross-view fusion before further claims.
