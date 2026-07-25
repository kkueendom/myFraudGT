# Multi-Model Dynamic Evidence Reliability Benchmark

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Sampling protocol: `dynamic_random`
- Training allowed: no
- Threshold or router tuning allowed: no
- Formal baseline: registered historical initial A2

## Research Question

Does the separation between evidence sensitivity and corrective utility recur
across independently trained fraud-graph evidence models under the same
dynamic-random sampling protocol?

## Completed Benchmark Blocks

### Fixed A2 Checkpoints

- six datasets;
- two dynamic streams per dataset;
- eight full val/test events per stream;
- 96 total events.

Result:
`docs/experiments/A2_DYNAMIC_SAMPLING_STABILITY_RESULTS.md`.

### CET v1

- Small-LI and Large-LI;
- fusion and encoder-only variants;
- repeated normal/shuffled/off/base evaluation;
- 28 total events.

Result:
`docs/experiments/CET_DYNAMIC_RELIABILITY_RESULTS.md`.

## New Block: Fixed TIER Evidence Models

Use the four completed Phase 1 evidence-only checkpoints from commit
`204b3075`:

| Dataset | Evidence selection | Model seed | Checkpoint selected epoch |
|---|---|---:|---:|
| Small-LI | recent | 42 | 39 |
| Small-LI | role_motif | 42 | read from checkpoint |
| Large-LI | recent | 44 | read from checkpoint |
| Large-LI | role_motif | 44 | read from checkpoint |

Each checkpoint receives two independent audit streams. Each stream performs
eight consecutive full validation/test events without RNG restoration.

Total new work:

- 8 tasks;
- 64 full events;
- 16,384 validation loader iterations;
- 16,384 test loader iterations.

## Same-Batch Evaluation

For every sampled batch, compute:

1. TIER normal evidence score;
2. TIER shuffled evidence score;
3. TIER off evidence score;
4. frozen epoch-499 A2 score.

All four scores must refer to the same target edge IDs and labels. The audit
must fail on any edge-ID or label misalignment.

For each event:

- choose the TIER threshold on the dynamic validation normal scores;
- choose the frozen-A2 threshold on the same validation targets;
- apply both thresholds to the immediately following dynamic test event;
- report normal, shuffled, off and base F1;
- report normal-shuffled and normal-off F1 gaps;
- report same-batch normal-base F1 delta;
- report changed, corrected and broken predictions for every evidence
  condition;
- report token coverage, support, target count and unique-edge rate;
- record query and inference time.

The historical initial A2 remains the formal baseline. The epoch-499 A2 is
only a same-batch diagnostic.

## Registered Task Layout

| Task | Dataset | Evidence | Stream |
|---:|---|---|---|
| 0 | Small-LI | recent | 1 |
| 1 | Small-LI | recent | 2 |
| 2 | Small-LI | role_motif | 1 |
| 3 | Small-LI | role_motif | 2 |
| 4 | Large-LI | recent | 1 |
| 5 | Large-LI | recent | 2 |
| 6 | Large-LI | role_motif | 1 |
| 7 | Large-LI | role_motif | 2 |

Seven GPUs start seven distinct tasks. The first worker to finish atomically
claims the eighth task, so no GPU remains idle while another initial task is
still running. No duplicate stream is added merely to occupy a GPU.

## Mechanism Classification

Classify each model/dataset pair after combining 16 events:

- **useful aligned evidence:** normal-shuffled mean F1 >= 0.01,
  same-batch normal-base mean delta > 0, corrected > broken in at least 12/16
  events and mean changed >= 50;
- **sensitive but harmful:** normal-shuffled mean F1 >= 0.01 but
  same-batch delta <= 0 or corrected <= broken in at least 8/16 events;
- **used but unaligned:** normal-off mean F1 >= 0.01 but
  normal-shuffled mean F1 < 0.01;
- **inactive evidence:** both counterfactual gaps < 0.01.

These labels describe fixed checkpoints. They are not statistical guarantees.

## Benchmark Advancement Gate

Advance the evaluation-method paper only if:

1. the same-batch audit completes without alignment or protocol violations;
2. at least three independently trained evidence model families exhibit a
   repeatable sensitivity-versus-utility mismatch;
3. the mismatch appears on both Small-LI and Large-LI, or the scale difference
   itself is stable and explainable;
4. dynamic event variation changes at least one single-run conclusion;
5. the final paper explicitly distinguishes sampling variation, evidence
   sensitivity, corrective utility and checkpoint selection.

If TIER is uniformly inactive, the benchmark still records a negative result
but does not yet establish cross-model generality. In that case, add one
preregistered COSTAR fixed-checkpoint block before making a paper-level claim.

## Engineering Requirements

- Commit runner, spec and tests before execution.
- Smoke all eight tasks with one repeat and two val/test steps.
- Use no dedicated evaluation generator or RNG restoration.
- Preserve every previous output directory.
- Record dataset, evidence selection, model seed, audit seed, commit, config,
  checkpoint, checkpoint epoch, protocol and all diagnostics in each manifest.
- Poll infrequently and keep logs concise.
