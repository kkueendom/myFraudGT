# DMPRD P0 Direction Validation Conclusion

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Decision Date: 2026-07-17
- Verification Status: QUICK DIRECTION VALIDATION VERIFIED
- Branch: `feature/support-calibrated-prototype-residual`
- Method Commit: `12383b1`
- Audit Integrity Fix: `8a04509`
- Quick Result and Control Plan Commit: `a1d8633`

## Final Decision

Retain the support- and uncertainty-calibrated prototype residual as the next
formal experimental direction. It passes both the matched A2 quick screen and
the fixed-residual-scale confound check.

This is a direction-level decision, not yet a paper-level six-dataset,
multi-seed claim.

## Evidence 1: Matched A2 Quick Screen

Across Small-LI, Medium-HI, and Large-LI at matched 120-epoch budgets:

- validation-selected test F1 improves on `3/3` datasets;
- mean validation-selected gain is `+0.02180`;
- mean raw-best gain is `+0.00480`;
- no raw-best regression is below `-0.010`.

The pre-registered quick-screen decision is `PASS`.

## Evidence 2: Fixed-0.25 Gate Control

Final logs confirm that every control has `gate=0.2500` and
`proto_conf=0.0000`. Full dynamic P0 versus this fixed control is:

| Dataset | Fixed gate val | Dynamic P0 val | Delta | Fixed raw | Dynamic raw | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.47761 | 0.47059 | -0.00702 | 0.49042 | 0.49520 | +0.00478 |
| Medium-HI | 0.71293 | 0.72253 | +0.00960 | 0.73943 | 0.74317 | +0.00374 |
| Large-LI | 0.16535 | 0.23000 | +0.06465 | 0.23529 | 0.23000 | -0.00529 |
| **Mean delta** | | | **+0.02241** | | | **+0.00108** |

Dynamic P0 wins validation-selected F1 on `2/3` datasets. It passes the
pre-registered rule requiring mean val gain above `+0.005`, at least two wins,
and mean raw delta no lower than `-0.005`.

The Small-LI exception matters: a fixed smaller residual is better there under
val selection. The supported claim is an aggregate robustness improvement, not
universal per-dataset dominance over every scale control.

## Evidence 3: Additional Seed

On Small-LI seed43, matched 120-epoch P0 versus A2 is:

- validation-selected test F1: `0.45149` versus `0.43262`, delta `+0.01887`;
- raw-best test F1: `0.49516` versus `0.49378`, delta `+0.00138`.

This is a positive robustness signal but not a substitute for the planned
three- or five-seed formal evaluation.

## Component Interpretation

On Small-LI seed42, support/variance reliability alone improves val-select by
`+0.00571` but changes raw-best by `-0.00273` versus A2. The full sample gate
adds another `+0.01424` val-select and `+0.00922` raw-best versus the
support/variance-only variant.

Together with the fixed-floor check, the evidence supports retaining both the
prototype quality statistics and sample-conditioned gate for formal testing.

## Scope and Next Gate

The next experiment should be a matched 500-epoch A2/P0 evaluation on all six
AML datasets using validation-selected test F1 as the primary metric. Only
after that six-dataset screen passes should missing seeds be filled and
significance, efficiency, and full component ablations be run.

Raw-best remains a diagnostic and must not be used as the official paper
checkpoint-selection rule.

## Reproducible Evidence

- Quick results: `docs/experiments/DMPRD_P0_QUICKCHECK_RESULTS.tsv`
- Fixed-floor results: `docs/experiments/DMPRD_P0_FLOOR_CONTROL_RESULTS.tsv`
- Seed43 results: `docs/experiments/DMPRD_P0_SEED43_RESULTS.tsv`
- Quick outputs: `results/dmprd_p0_quick120/`
- Control outputs: `results/dmprd_p0_floor_control120/`
- Audits: `run/dmprd_p0_quick_audit.py` and
  `run/dmprd_p0_floor_control_audit.py`
