# TIER Phase 2a-P0 Directional Utility Probe Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED manifests audited
- Sampling Protocol: `dynamic_random`
- Expected Commit: `3eca8008`
- Parent Evidence Commit: `c89dc527`
- Scope: optimistic in-sample-teacher feasibility probe
- Fixed-panel A2: excluded

## Per-Task Results

| Dataset | Family | Selection | Cal eligible | Cal pass | Val corrected/broken | Val changed/required | Val delta | Val pass | Test corrected/broken | Test changed/required | Test delta | Add/remove | Test pass | Final |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| Large-LI | flow_role | recent | 0 | false | 57/4 | 61/50 | 0.02139 | true | 49/9 | 58/50 | 0.01896 | 1/57 | true | fail |
| Large-LI | flow_role | role_motif | 0 | false | 41/5 | 46/50 | -0.00141 | false | 36/8 | 44/50 | -0.00481 | 0/44 | false | fail |
| Small-LI | flow_role | recent | 0 | false | 17/9 | 26/50 | 0.00738 | false | 24/12 | 36/50 | 0.01219 | 14/22 | false | fail |
| Small-LI | flow_role | role_motif | 0 | false | 10/26 | 36/50 | -0.00578 | false | 13/42 | 55/50 | -0.01175 | 45/10 | false | fail |
| Large-LI | structure | recent | 0 | false | 61/68 | 129/50 | -0.05010 | false | 58/102 | 160/50 | -0.08879 | 96/64 | false | fail |
| Large-LI | structure | role_motif | 0 | false | 43/15 | 58/50 | 0.00013 | true | 44/23 | 67/50 | -0.00341 | 17/50 | false | fail |
| Small-LI | structure | recent | 0 | false | 26/299 | 325/50 | -0.07732 | false | 39/472 | 511/50 | -0.13903 | 487/24 | false | fail |
| Small-LI | structure | role_motif | 0 | false | 2/0 | 2/50 | 0.00144 | false | 1/0 | 1/50 | 0.00070 | 0/1 | false | fail |
| Large-LI | temporal | recent | 0 | false | 67/158 | 225/50 | -0.04468 | false | 65/185 | 250/50 | -0.11439 | 186/64 | false | fail |
| Large-LI | temporal | role_motif | 2 | true | 72/64 | 136/50 | 0.01647 | false | 64/71 | 135/50 | -0.03388 | 67/68 | false | fail |
| Small-LI | temporal | recent | 0 | false | 17/20 | 37/50 | 0.00402 | false | 29/21 | 50/50 | 0.00950 | 23/27 | false | fail |
| Small-LI | temporal | role_motif | 0 | false | 13/67 | 80/50 | -0.02165 | false | 17/127 | 144/50 | -0.06006 | 127/17 | false | fail |

## Family Summary

| Family | Tasks | Cal passes | Val passes | Test passes | Qualified | Positive val+test delta | Mean val delta | Mean test delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| structure | 4 | 0 | 1 | 0 | 0 | 1 | -0.03146 | -0.05763 | optimistic_signal_insufficient |
| temporal | 4 | 1 | 0 | 0 | 0 | 1 | -0.01146 | -0.04971 | optimistic_signal_insufficient |
| flow_role | 4 | 0 | 1 | 1 | 0 | 2 | 0.00540 | 0.00365 | optimistic_signal_insufficient |

## Gate Conclusion

No evidence family passed calibration, frozen validation, and locked test on both scales. Because even the optimistic in-sample-teacher upper bound failed, expensive cross-fitted teacher training and full CrossFusion are not authorized.

This P0 uses in-sample A2 training errors and is an optimistic upper bound. It cannot support a paper result even if a task passes. Raw-best is not defined for this locked diagnostic.
