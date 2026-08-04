# Multi-CDVT Rapid Validation Target

Branch: `feature/cdvt-multi-backbone-screen`

## Current Target

The immediate goal is to determine whether **Multi-CDVT can exceed the
published Multi-FraudGT results in the FraudGT paper**. This is a rapid
directional screen, not the final matched-baseline experiment.

Do not rerun Multi-FraudGT at this stage. Use the published Multi-FraudGT
Val-selected Test F1 values as the screening reference:

| Dataset | Published Multi-FraudGT F1 |
|---|---:|
| AML Small-LI | 0.47010 |
| AML Small-HI | 0.76130 |
| AML Medium-LI | 0.44060 |
| AML Medium-HI | 0.75930 |
| AML Large-LI | 0.37430 |
| AML Large-HI | 0.73340 |
| Mean | 0.58983 |

## Required Experiment Protocol

1. Run **Multi-CDVT only** on all six AML datasets.
2. Use seed 42 for the initial rapid screen.
3. Keep FraudGT's public `dynamic_random` sampling protocol:
   - train, validation, and test use random dynamic sampling;
   - `LinkNeighborLoader` keeps `shuffle=True`;
   - `val.fixed_target_panel=False`;
   - do not fix validation or test target edges;
   - do not restore sampler RNG state before evaluation.
4. Train for the complete **500 epochs**.
5. **Disable early stopping.** No Multi-CDVT task may finish before epoch 500
   merely because validation F1 has stopped improving.
6. Use Val-selected Test F1 as the primary result: select the epoch with the
   highest validation F1 and report the corresponding test F1.
7. Raw-best Test F1 may be reported only as supplementary analysis.
8. Compare Val-selected only with the published Val-selected values above; do
   not compare across metrics.

## Rapid-Screen Decision

For each dataset, report:

- Multi-CDVT Val-selected Test F1;
- published Multi-FraudGT F1;
- `Delta F1 = Multi-CDVT - published Multi-FraudGT`;
- win/loss status;
- whether `|Delta F1| < 0.005`, which must be marked as possible sampling
  variation.

The direction is promising only if Multi-CDVT wins on at least four of six
datasets and has a positive mean delta. A mean delta no greater than `+0.005`
must be described as marginal rather than stable superiority.

## Explicit Non-Goals

- Do not launch or resume `multi_account_only` / Multi-FraudGT training.
- Do not use the incomplete matched Multi-FraudGT runs as the screening
  baseline.
- Do not change the Multi-CDVT architecture or tune it using test results.
- Do not start multi-seed expansion until the six-dataset seed-42 screen has
  been summarized.
- Do not modify or stop unrelated additive-fusion experiments unless the user
  separately requests it.

## Handoff Instruction

The next model should inspect the currently running jobs, then change the
execution workflow to satisfy this Target. Preserve completed artifacts, use a
new result root for any restarted 500-epoch task, make an independent Git
commit before launching changed experiments, and record
`sampling_protocol=dynamic_random` in every manifest.
