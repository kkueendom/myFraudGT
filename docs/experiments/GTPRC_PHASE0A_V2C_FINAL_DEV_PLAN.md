# GTPRC Phase 0A v2c Final Development Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent result: `GTPRC_PHASE0A_V2B_DEV_RESULTS.md`
- Adaptive attempt: final simulator-development attempt
- AML validation/test labels allowed: no

## Single Change

Increase correctable-event base rates:

- standard regimes: `correct_logit` from `-1.75/-1.85` to
  `-0.35/-0.45`;
- rare-duplicate regime: `correct_logit` from `-2.65` to `-0.70`.

The break process is unchanged. The intended controlled relationship is:

- full or high coverage remains slightly harmful;
- positive net utility becomes possible near `alpha=0.40`;
- aligned scores can identify a safer corrective subset;
- shuffled and harmful controls remain invalid.

No other generator, method, bound, dependency group, score coefficient,
policy grid, seed family, or gate may change in v2c.

## Execution

- seven distinct GPU tasks;
- 64 development replicates per regime;
- new output directory;
- selected test-risk diagnostics retained;
- v2 development gate applied unchanged.

## Stop Rule

If v2c fails any development gate, stop the current synthetic GTPRC route.
Do not perform another generator adjustment and do not run formal v2 or CPSE.

If v2c passes, lock the generator and run the existing formal seed set with
512 replicates per regime.

