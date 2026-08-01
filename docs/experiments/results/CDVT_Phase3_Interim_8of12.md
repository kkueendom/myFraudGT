# CDVT Representative-Dataset Paired Three-Seed Summary

FraudGT/account-only and CDVT use independent seeds 42, 43, and 44 under the same `dynamic_random` protocol. The primary robustness evidence is the same-seed paired delta. Published PE-FraudGT and Multi-FraudGT means are descriptive references.

**Interim status:** 8 of 12 Phase 3 training manifests are complete, but only
five of the nine required same-seed pairs are complete. Mean and standard
deviation tables are intentionally withheld until all matched pairs finish.

## Seed-level paired results

| Dataset | Seed | FraudGT val-selected | CDVT val-selected | Paired delta | FraudGT raw-best | CDVT raw-best | Paired delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small-LI | 42 | 0.47893 | 0.43798 | -0.04094 | 0.47985 | 0.47687 | -0.00299 |
| Small-LI | 43 | 0.43515 | 0.48805 | +0.05291 | 0.46377 | 0.48805 | +0.02429 |
| Medium-LI | 43 | 0.36598 | 0.36957 | +0.00359 | 0.48260 | 0.46113 | -0.02147 |
| Large-LI | 42 | 0.21287 | 0.33803 | +0.12516 | 0.29126 | 0.44816 | +0.15690 |
| Large-LI | 43 | 0.28221 | 0.34909 | +0.06688 | 0.34659 | 0.48447 | +0.13788 |

## Interim Coverage

| Dataset | Completed account-only seeds | Completed CDVT seeds | Completed pairs |
|---|---|---|---:|
| Small-LI | 42, 43, 44 | 42, 43 | 2/3 |
| Medium-LI | 43, 44 | 42, 43 | 1/3 |
| Large-LI | 42, 43 | 42, 43 | 2/3 |

The completed Val-selected paired deltas are positive for seed 43 on all three
scales: Small-LI `+0.05291`, Medium-LI `+0.00359`, and Large-LI `+0.06688`.
Medium-LI is within the predefined `0.005` sampling-variation band. These rows
are final manifest values, but the cross-seed trend is not final.

Incomplete manifests:
- `/e/yky/FraudGT_cdvt_results/followup_f7209f2/phase3/Small-LI_dual_view_seed44/manifest.json`
- `/e/yky/FraudGT_cdvt_results/followup_f7209f2/ablation/Medium-LI_account_only_seed42/manifest.json`
- `/e/yky/FraudGT_cdvt_results/followup_f7209f2/phase3/Medium-LI_dual_view_seed44/manifest.json`
- `/e/yky/FraudGT_cdvt_results/followup_f7209f2/phase3/Large-LI_account_only_seed44/manifest.json`
- `/e/yky/FraudGT_cdvt_results/followup_f7209f2/phase3/Large-LI_dual_view_seed44/manifest.json`
