# DMPRD Branch Experimental Conclusion

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Decision Date: 2026-07-17
- Verification Status: FORMAL BEST-SEED VERIFIED / THREE-SEED RUNNING
- Branch: `feature/dmprd-prototype-residual`

## Decision

Select **A2**, the four-prototype bounded residual without distribution
statistics, as the DMPRD mainline. Reject A3 as an unnecessary extension.

The branch has a clear positive best-seed result under the user's primary
raw-best test-F1 rule: A2 beats the historical FraudGT baseline on all six
datasets after 500 epochs, with a mean gain of `+0.02981`.

This is not yet a three-seed robustness claim. The supplemental seed queue is
still running and must be reported separately.

## A2 Versus FraudGT Baseline

| Dataset | A2 raw-best | Baseline raw-best | Delta |
|---|---:|---:|---:|
| Small-LI | 0.50667 | 0.50474 | +0.00193 |
| Small-HI | 0.79497 | 0.78599 | +0.00898 |
| Medium-LI | 0.59031 | 0.51852 | +0.07179 |
| Medium-HI | 0.78940 | 0.76768 | +0.02172 |
| Large-LI | 0.44720 | 0.41379 | +0.03341 |
| Large-HI | 0.76223 | 0.72120 | +0.04103 |
| **Mean delta** | | | **+0.02981** |

Formal baseline decision: `PASS` (`6/6` raw-best wins).

Val-selected test F1 improves by `+0.01343` on average and wins on four of six
datasets. Small-LI and Large-LI remain weak under val selection, so the paper
must not claim universal val-select improvement.

## A2 Versus Prototype-Only

A2 improves mean raw-best by `+0.00159` and mean val-select by `+0.00883`
against the full-budget `proto_only` reference. The mean result passes the
pre-registered prototype go/no-go rule, but gains are heterogeneous:

- Raw-best is lower on Small-LI, Small-HI, Medium-HI, and Large-HI.
- Only Small-LI and Medium-HI fall by more than `0.005`.
- The evidence supports a modest aggregate improvement, not a uniform win.

## Ablation Decision

The 120-epoch Small-LI ablation first showed that four prototype slots improve
over a single slot:

- A2 minus A1 val-select: `+0.01494`.
- A2 minus A1 raw-best: `+0.00702`.

The full 500-epoch A3 control then tested distribution statistics on all six
datasets. Relative to A2:

- Mean raw-best delta: `+0.00034`.
- Mean val-select delta: `-0.02714`.
- Raw-best wins: `3/6`.
- Val-select wins: `0/6`.
- Decision: `distribution_stats_contribute=FAIL`.

The `+0.00034` raw change is practically negligible and does not compensate
for the consistent val-select degradation. Distribution statistics should not
be presented as a successful module or included in the selected architecture.

## Supported Claim

The current evidence supports this claim:

> A controlled four-prototype bounded residual improves FraudGT raw-best test
> F1 across all six AML datasets under matched 500-epoch best-seed runs, while
> additional prototype-distribution statistics add no reliable benefit.

It does not yet support a three-seed mean claim. Use
`run/dmprd_a2_robust_audit.py` after the supplemental queue finishes before
making a robustness or variance statement.

## Reproducible Evidence

- Quick audit: `docs/experiments/DMPRD_QUICKCHECK_RESULTS.tsv`
- A2 formal audit: `docs/experiments/DMPRD_FORMAL_A2_RESULTS.tsv`
- A3 formal audit: `docs/experiments/DMPRD_FORMAL_A3_RESULTS.tsv`
- A2 formal results: `results/dmprd_formal500`
- A3 formal results: `results/dmprd_a3_formal500`
- Supplemental seeds: `results/dmprd_a2_robust500`
