# TIER Phase 1b Evidence Family Decomposition Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED manifests audited
- Sampling Protocol: `dynamic_random`
- Historical Baseline: `initial_A2`
- Expected Commit: `c89dc527`
- Fixed-panel A2: excluded

## Per-Task Results

| Dataset | Family | Selection | Seed | Val-selected Test F1 | Delta vs A2 | Raw-best Test F1 | Delta vs A2 | Normal-shuffled | Normal-off | Corrected | Broken | Ratio | Correction rate | Changed | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Large-LI | flow_role | recent | 44 | 0.16162 | -0.13946 | 0.17439 | -0.27281 | 0.16162 | 0.16162 | 16 | 365 | 0.04384 | 0.18391 | 381 | fail |
| Large-LI | flow_role | role_motif | 44 | 0.20884 | -0.09224 | 0.25296 | -0.19424 | 0.20043 | 0.20884 | 7 | 40 | 0.17500 | 0.08046 | 47 | fail |
| Small-LI | flow_role | recent | 42 | 0.32258 | -0.13989 | 0.34275 | -0.16392 | 0.32258 | 0.32258 | 39 | 118 | 0.33051 | 0.14391 | 157 | fail |
| Small-LI | flow_role | role_motif | 42 | 0.31834 | -0.14413 | 0.35810 | -0.14857 | 0.31834 | 0.31834 | 38 | 112 | 0.33929 | 0.14022 | 150 | fail |
| Large-LI | structure | recent | 44 | 0.04545 | -0.25563 | 0.06840 | -0.37880 | 0.04545 | 0.03574 | 31 | 58 | 0.53448 | 0.25410 | 89 | fail |
| Large-LI | structure | role_motif | 44 | 0.06485 | -0.23623 | 0.07323 | -0.37397 | 0.06485 | 0.06485 | 53 | 1248 | 0.04247 | 0.48182 | 1301 | fail |
| Small-LI | structure | recent | 42 | 0.06460 | -0.39787 | 0.10790 | -0.39877 | 0.05859 | 0.06460 | 44 | 962 | 0.04574 | 0.15278 | 1006 | fail |
| Small-LI | structure | role_motif | 42 | 0.10847 | -0.35400 | 0.12745 | -0.37922 | 0.10550 | 0.10847 | 43 | 753 | 0.05710 | 0.15867 | 796 | fail |
| Large-LI | temporal | recent | 44 | 0.03533 | -0.26575 | 0.07601 | -0.37119 | 0.02591 | 0.03150 | 31 | 753 | 0.04117 | 0.36471 | 784 | fail |
| Large-LI | temporal | role_motif | 44 | 0.06897 | -0.23211 | 0.10256 | -0.34464 | 0.05754 | 0.06897 | 13 | 88 | 0.14773 | 0.13830 | 101 | fail |
| Small-LI | temporal | recent | 42 | 0.05318 | -0.40929 | 0.10369 | -0.40298 | 0.04157 | 0.05318 | 31 | 1214 | 0.02554 | 0.11567 | 1245 | fail |
| Small-LI | temporal | role_motif | 42 | 0.09404 | -0.36843 | 0.12644 | -0.38023 | 0.08360 | 0.09404 | 48 | 458 | 0.10480 | 0.17391 | 506 | fail |

## Family Summary

| Family | Tasks | Counterfactually active | Qualified | Small passes | Large passes | Mean Val-selected Delta | Mean Raw-best Delta | Best corrected/broken | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| structure | 4 | 4 | 0 | 0 | 0 | -0.31093 | -0.38269 | 0.53448 | informative_but_unsafe |
| temporal | 4 | 4 | 0 | 0 | 0 | -0.31890 | -0.37476 | 0.14773 | informative_but_unsafe |
| flow_role | 4 | 4 | 0 | 0 | 0 | -0.12893 | -0.19489 | 0.33929 | informative_but_unsafe |

## Gate Conclusion

No evidence family passed the preregistered qualification gate on both Small-LI and Large-LI. Phase 2 is not authorized by these results.

Qualification is recomputed from each manifest. It is not inferred
from headline F1, Raw-best F1, or the stored gate label alone.
