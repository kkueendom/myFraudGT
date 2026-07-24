# TIER Phase 1b Evidence Family Decomposition Results

> **Attribution validity notice:** Post-run code review found that token
> channels were family-masked, but all six support channels were still passed
> to every family. These runs remain valid evidence that historical context is
> counterfactually active and globally unsafe, but they are not a clean
> structure/temporal/flow-role attribution experiment. Family comparisons in
> this file are superseded by the support-masked rerun.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED manifests audited
- Sampling Protocol: `dynamic_random`
- Historical Baseline: `initial_A2`
- Expected Commit: `f43c286`
- Fixed-panel A2: excluded

## Per-Task Results

| Dataset | Family | Selection | Seed | Val-selected Test F1 | Delta vs A2 | Raw-best Test F1 | Delta vs A2 | Normal-shuffled | Normal-off | Corrected | Broken | Ratio | Correction rate | Changed | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Large-LI | flow_role | recent | 44 | 0.09211 | -0.20897 | 0.14768 | -0.29952 | 0.08860 | 0.09211 | 27 | 94 | 0.28723 | 0.22500 | 121 | fail |
| Large-LI | flow_role | role_motif | 44 | 0.14337 | -0.15771 | 0.23404 | -0.21316 | 0.14337 | 0.14337 | 41 | 89 | 0.46067 | 0.33884 | 130 | fail |
| Small-LI | flow_role | recent | 42 | 0.31304 | -0.14943 | 0.35448 | -0.15219 | 0.31304 | 0.31304 | 39 | 100 | 0.39000 | 0.13311 | 139 | fail |
| Small-LI | flow_role | role_motif | 42 | 0.33209 | -0.13038 | 0.35294 | -0.15373 | 0.33209 | 0.33209 | 39 | 95 | 0.41053 | 0.14391 | 134 | fail |
| Large-LI | structure | recent | 44 | 0.04697 | -0.25411 | 0.06100 | -0.38620 | 0.04697 | 0.04184 | 61 | 1192 | 0.05117 | 0.45185 | 1253 | fail |
| Large-LI | structure | role_motif | 44 | 0.06192 | -0.23916 | 0.07678 | -0.37042 | 0.06192 | 0.06192 | 44 | 978 | 0.04499 | 0.40000 | 1022 | fail |
| Small-LI | structure | recent | 42 | 0.08987 | -0.37260 | 0.12043 | -0.38624 | 0.07944 | 0.08987 | 53 | 671 | 0.07899 | 0.19203 | 724 | fail |
| Small-LI | structure | role_motif | 42 | 0.10961 | -0.35286 | 0.14901 | -0.35766 | 0.10654 | 0.10961 | 43 | 735 | 0.05850 | 0.15867 | 778 | fail |
| Large-LI | temporal | recent | 44 | 0.07026 | -0.23082 | 0.07286 | -0.37434 | 0.05598 | 0.06248 | 23 | 315 | 0.07302 | 0.23958 | 338 | fail |
| Large-LI | temporal | role_motif | 44 | 0.05099 | -0.25009 | 0.08147 | -0.36573 | 0.04485 | 0.05099 | 44 | 654 | 0.06728 | 0.36364 | 698 | fail |
| Small-LI | temporal | recent | 42 | 0.04323 | -0.41924 | 0.06885 | -0.43782 | 0.04323 | 0.04323 | 44 | 354 | 0.12429 | 0.15385 | 398 | fail |
| Small-LI | temporal | role_motif | 42 | 0.09949 | -0.36298 | 0.14844 | -0.35823 | 0.09159 | 0.09949 | 47 | 668 | 0.07036 | 0.17279 | 715 | fail |

## Family Summary

| Family | Tasks | Counterfactually active | Qualified | Small passes | Large passes | Mean Val-selected Delta | Mean Raw-best Delta | Best corrected/broken | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| structure | 4 | 4 | 0 | 0 | 0 | -0.30468 | -0.37513 | 0.07899 | informative_but_unsafe |
| temporal | 4 | 4 | 0 | 0 | 0 | -0.31578 | -0.38403 | 0.12429 | informative_but_unsafe |
| flow_role | 4 | 4 | 0 | 0 | 0 | -0.16162 | -0.20465 | 0.46067 | informative_but_unsafe |

## Gate Conclusion

No evidence family passed the preregistered qualification gate on both Small-LI and Large-LI. Phase 2 is not authorized by these results.

Qualification is recomputed from each manifest. It is not inferred
from headline F1, Raw-best F1, or the stored gate label alone.
