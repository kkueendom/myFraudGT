# DMPRD P0 Quick-Check Conclusion

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Decision Date: 2026-07-17
- Verification Status: MATCHED 120-EPOCH QUICK SCREEN VERIFIED
- Branch: `feature/support-calibrated-prototype-residual`
- Method Commit: `12383b1`
- Audit Fix Commit: `8a04509`

## Quick-Screen Result

The pre-registered three-dataset quick screen passes. Every result below uses
the same seed, configuration, scheduler budget, and 120 epochs for A2 and P0.

| Dataset | A2 val-select | P0 val-select | Delta | A2 raw | P0 raw | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.45064 | 0.47059 | +0.01995 | 0.48871 | 0.49520 | +0.00649 |
| Medium-HI | 0.71083 | 0.72253 | +0.01170 | 0.74394 | 0.74317 | -0.00077 |
| Large-LI | 0.19626 | 0.23000 | +0.03374 | 0.22131 | 0.23000 | +0.00869 |
| **Mean delta** | | | **+0.02180** | | | **+0.00480** |

P0 wins validation-selected test F1 on `3/3` datasets. Mean raw-best also
improves, and no dataset has a raw regression below `-0.010`.

## Component Diagnostic

The Small-LI three-way diagnostic is:

| Variant | Val-select | Raw-best |
|---|---:|---:|
| A2 | 0.45064 | 0.48871 |
| Support/variance only | 0.45635 | 0.48598 |
| Full P0 | 0.47059 | 0.49520 |

Support/variance alone changes val-select by `+0.00571` and raw-best by
`-0.00273` versus A2. Adding the sample gate changes val-select by another
`+0.01424` and raw-best by `+0.00922` versus support/variance only.

## Required Confound Check

The final test logs show mean P0 gates of `0.2558` on Small-LI, `0.2523` on
Medium-HI, and `0.2682` on Large-LI. These values are close to the configured
floor of `0.25`. Therefore, the quick PASS proves that the P0 training path is
useful, but does not yet prove that sample-varying reliability is better than a
fixed smaller prototype residual.

A matched constant-0.25 gate control is required before attributing the gain to
reliability calibration. That follow-up is pre-registered separately in
`DMPRD_P0_FLOOR_CONTROL_PLAN.md`.

## Reproducible Evidence

- Results: `results/dmprd_p0_quick120/`
- Audit: `run/dmprd_p0_quick_audit.py`
- Machine-readable table: `docs/experiments/DMPRD_P0_QUICKCHECK_RESULTS.tsv`
