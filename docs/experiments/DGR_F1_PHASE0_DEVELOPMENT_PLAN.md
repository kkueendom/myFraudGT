# DGR-F1 Phase 0 Development Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent plan mode
- Status: preregistered, not started
- Parent decision: `STOP_TREFIC`
- Experiment type: controlled graph-stream simulation
- FraudGT/evidence training: none
- Validation/test loader use: none
- GPU tasks: seven nonduplicate scenario families

## Fixed Method

- Independent dynamic streams are the statistical units.
- Registered stream checkpoints: 8, 16, 32 and 64.
- Practical margin: `epsilon=0.005`.
- Required replication probability: `pi0=0.75`.
- Family confidence level: `1-delta=0.95`.
- Four sequential checkpoints and three monitored endpoints (mean delta,
  practical improvement probability and practical harm probability) share the
  error budget by Bonferroni correction: each interval uses
  `delta / (4 x 3)`.
- Mean paired-stream intervals use a Studentized interval.
- Practical replication intervals use exact Clopper-Pearson bounds.
- `REPLICABLE_IMPROVEMENT` requires mean Delta F1 LCB above zero and practical
  improvement-probability LCB at least 0.75.
- `REPLICABLE_HARM` is the symmetric decision.
- Otherwise the method returns `INSUFFICIENT_INFORMATION`.

The Studentized interval is not claimed to be a finite-sample guarantee. Its
coverage must be established empirically in this screen.

## Real-Graph Templates

Use only frozen OOF confusion templates from Small-LI and Large-LI. No
validation/test loader is opened. Scenario generators operate on stream-level
confusion/intervention counts and preserve:

- observed rare-positive scale;
- feasible add/remove counts;
- dynamic base-ratio variation;
- graph/time cluster effects.

## Seven Registered Scenario Families

| GPU | Scenario | Ground-truth evaluation target |
|---:|---|---|
| 0 | stable IID improvement | replicable improvement |
| 1 | stable graph-correlated improvement | replicable improvement |
| 2 | stable temporally correlated improvement | replicable improvement |
| 3 | positive mean but low replication | insufficient information |
| 4 | base-ratio sign reversal | insufficient information |
| 5 | null intervention | insufficient information |
| 6 | stable harmful intervention | replicable harm |

Each task evaluates both Small-LI- and Large-LI-like templates.

## Compared Evaluation Rules

- one-stream raw Delta F1;
- row-IID paired-F1 interval;
- mean-only group-sequential interval;
- replication-only exact interval;
- DGR-F1 joint decision.

## Monte Carlo Budget

- 512 Monte Carlo experiments per scenario/template;
- 64 independent streams per experiment;
- four registered checkpoints;
- task seeds `76001` through `76007`;
- no post-run parameter changes.

## Development Gate

All conditions must pass:

1. exact F1 mechanism identity has zero failures;
2. DGR-F1 false improvement is at most 0.05 on null and harmful scenarios for
   both templates;
3. DGR-F1 false harm is at most 0.05 on null and positive scenarios for both
   templates;
4. DGR-F1 improvement power is at least 0.80 by 64 streams in all three stable
   positive scenarios for both templates;
5. median stopping checkpoint is at most 32 for stable IID improvement;
6. DGR-F1 returns insufficient information in at least 0.90 of low-replication
   and ratio-reversal experiments on both templates;
7. one-stream or mean-only evaluation makes a false stable claim in at least
   0.15 of one unstable scenario, demonstrating a nontrivial benchmark;
8. empirical simultaneous coverage of the mean intervals is at least 0.94 in
   null and ratio-reversal scenarios;
9. Large-LI positive power is not more than 0.15 below Small-LI power in the
   stable IID scenario.
10. DGR-F1 harm-detection power is at least 0.80 by 64 streams in the stable
    harmful scenario for both templates.

## Stop Rule

Failure stops DGR-F1. Do not tune:

- `epsilon`;
- `pi0`;
- `delta`;
- stream checkpoints;
- scenario effects;
- correlation strengths;
- template counts.

Passing authorizes a prospective frozen-checkpoint protocol. It does not
authorize a predictive-model claim or validation/test access.
