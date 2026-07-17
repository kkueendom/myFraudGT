# DMPRD P0 Six-Dataset 500-Epoch Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: formal experiment plan + run
- Plan Date: 2026-07-17
- Verification Status: PRE-REGISTERED / NOT YET STARTED
- Branch: `feature/support-calibrated-prototype-residual`
- Starting Commit: `5da3814`

## Why Proceed

P0 has enough evidence to justify the full-budget screen:

- matched 120-epoch val-selected test F1 improves on `3/3` representative
  datasets, with mean gain `+0.02180` over A2;
- Small-LI seed43 also improves by `+0.01887` val-selected F1;
- dynamic P0 passes the fixed-0.25 residual control with mean val gain
  `+0.02241` and wins on `2/3` datasets.

These results make a 500-epoch improvement plausible but do not guarantee it.
The full experiment is therefore a pre-registered six-dataset screen, not a
confirmation run with post-hoc criteria.

## Matched Reference

The existing A2 runs in `results/dmprd_formal500/` are complete to epoch 499
and use the same parent code, dataset configurations, seeds, and 500-epoch
scheduler. P0 adds only the reliability path and preserves A2 behavior when
disabled, so rerunning A2 would duplicate six expensive tasks without improving
the comparison.

| Dataset | Seed | A2 val-select | A2 raw-best |
|---|---:|---:|---:|
| Small-LI | 42 | 0.46247 | 0.50667 |
| Small-HI | 42 | 0.77984 | 0.79497 |
| Medium-LI | 44 | 0.51163 | 0.59031 |
| Medium-HI | 42 | 0.77574 | 0.78940 |
| Large-LI | 44 | 0.30108 | 0.44720 |
| Large-HI | 43 | 0.72897 | 0.76223 |

## P0 Configuration

- four prototypes per class;
- A3 distribution statistics disabled;
- support/variance reliability enabled;
- sample-conditioned gate enabled;
- `support_tau=16`, `variance_tau=0.25`, `margin_tau=0.10`;
- residual gate floor `0.25`;
- 500 epochs, no early stopping, one matched seed per dataset.

## GPU Schedule

The queue starts Large-LI, Large-HI, Medium-LI, Medium-HI, and Small-LI on all
currently available GPUs. Small-HI starts automatically when the first GPU
finishes. GPUs with any existing compute PID are skipped. Polling is every 1200
seconds and unchanged states are not repeatedly logged.

## Primary Decision Rule

Primary metric: test F1 at the checkpoint selected by validation F1.

The six-dataset formal screen passes only if all conditions hold:

1. mean P0 minus A2 val-selected test-F1 gain is greater than `+0.005`;
2. P0 wins val-selected F1 on at least `4/6` datasets;
3. mean raw-best delta is non-negative;
4. at most one raw-best regression is below `-0.010`.

Raw-best is a secondary diagnostic and is not used for checkpoint selection.

If the screen passes, fill missing seeds for A2 and P0 and then run statistical
tests, efficiency analysis, and complete component ablations. If it fails, do
not launch multi-seed formal runs until the failure pattern is understood.

## Reproduction

```bash
nohup python3 run/dmprd_p0_formal_queue.py \
  > .dmprd_p0_formal500_queue.nohup.log 2>&1 &

python3 run/dmprd_p0_formal_audit.py
```

Expected P0 output: `results/dmprd_p0_formal500/`.
