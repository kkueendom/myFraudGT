# TIER Phase 1c High-Precision Intervention Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED manifests audited
- Sampling Protocol: `dynamic_random`
- Expected Commit: `27c9e72b`
- Parent Evidence Commit: `c89dc527`
- Fixed-panel A2: excluded
- Scope: paired same-batch mechanism diagnostic

## Per-Task Results

| Dataset | Family | Selection | Direction | Val eligible | Val corrected/broken | Val changed/required | Test corrected/broken | Test ratio | Test changed/required | A2 paired F1 | Routed paired F1 | Paired delta | Gate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Large-LI | flow_role | recent | evidence_negative | 0 | 25/5 | 30/50 | 17/2 | 8.50000 | 19/50 | 0.25676 | 0.26357 | 0.00681 | fail |
| Large-LI | flow_role | role_motif | evidence_negative | 0 | 23/5 | 28/50 | 16/3 | 5.33333 | 19/50 | 0.25676 | 0.24806 | -0.00869 | fail |
| Small-LI | flow_role | recent | evidence_negative | 0 | 19/5 | 24/50 | 32/13 | 2.46154 | 45/50 | 0.43651 | 0.42266 | -0.01385 | fail |
| Small-LI | flow_role | role_motif | evidence_negative | 0 | 16/4 | 20/50 | 29/12 | 2.41667 | 41/50 | 0.43651 | 0.42333 | -0.01318 | fail |
| Large-LI | structure | recent | evidence_negative | 0 | 14/1 | 15/50 | 10/2 | 5.00000 | 12/50 | 0.25676 | 0.25000 | -0.00676 | fail |
| Large-LI | structure | role_motif | evidence_negative | 0 | 17/3 | 20/50 | 12/4 | 3.00000 | 16/50 | 0.25676 | 0.22727 | -0.02948 | fail |
| Small-LI | structure | recent | evidence_negative | 0 | 13/8 | 21/50 | 22/11 | 2.00000 | 33/50 | 0.43651 | 0.42038 | -0.01613 | fail |
| Small-LI | structure | role_motif | evidence_negative | 0 | 15/7 | 22/50 | 26/7 | 3.71429 | 33/50 | 0.43651 | 0.43737 | 0.00086 | fail |
| Large-LI | temporal | recent | evidence_negative | 0 | 26/7 | 33/50 | 21/2 | 10.50000 | 23/50 | 0.25676 | 0.27200 | 0.01524 | fail |
| Large-LI | temporal | role_motif | both | 0 | 26/9 | 35/50 | 22/2 | 11.00000 | 24/50 | 0.25676 | 0.27419 | 0.01744 | fail |
| Small-LI | temporal | recent | evidence_positive | 0 | 2/0 | 2/50 | 1/2 | 0.50000 | 3/50 | 0.43651 | 0.43787 | 0.00136 | fail |
| Small-LI | temporal | role_motif | both | 0 | 21/8 | 29/50 | 38/16 | 2.37500 | 54/50 | 0.43651 | 0.42291 | -0.01360 | fail |

## Family Summary

| Family | Tasks | Validation passes | Test-gate passes | Qualified | Small passes | Large passes | Mean paired delta | Best test ratio | Max test changed | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| structure | 4 | 0 | 0 | 0 | 0 | 0 | -0.01288 | 5.00000 | 33 | not_separable_at_registered_coverage |
| temporal | 4 | 0 | 0 | 0 | 0 | 0 | 0.00511 | 11.00000 | 54 | not_separable_at_registered_coverage |
| flow_role | 4 | 0 | 0 | 0 | 0 | 0 | -0.00723 | 8.50000 | 45 | not_separable_at_registered_coverage |

## Gate Conclusion

No evidence family produced a validation-eligible intervention policy that passed the locked test gate on both scales. Learned ErrorRouter and CrossFusion are not authorized by Phase 1c.

The initial A2 delta is retained in JSON for experiment bookkeeping. Because this phase evaluates only the A2-scored paired subset, it must not replace the full-model headline table.
