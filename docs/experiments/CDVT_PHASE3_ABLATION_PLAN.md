# CDVT Phase 3, Ablation, and Runtime Plan

## Execution Gate

No task in this plan may start until the frozen six-dataset Phase 2 summary
passes both preregistered conditions:

1. at least four of six positive val-selected test F1 deltas; and
2. a positive six-dataset mean val-selected delta against initial A2.

`run/cdvt_phase3_gate.py` recomputes this decision from the authoritative Phase
1 and Phase 2 manifests. Every follow-up queue calls the gate before creating
its output root. Raw-best F1 is never used by the gate.

## Standard-Model Identity

The standard CDVT path remains the model frozen at commit `9038f85`:

- `dual_view`;
- `lambda_cons=0`;
- history size `K=4`;
- relation types enabled; and
- dynamic-random train, validation, and test sampling.

The follow-up branch adds ablation and orchestration interfaces. When relation
types are enabled, the temporal layer preserves the frozen operation order
`base + relation + edge`; the standard forward calculation is unchanged.

## Phase 3: Independent Seeds

Six new training tasks complete real seeds 42, 43, and 44 on representative
scales:

| Dataset | Reused seed | New seeds |
|---|---:|---:|
| AML Small-LI | 42 from Phase 1 | 43, 44 |
| AML Medium-LI | 42 from Phase 2 | 43, 44 |
| AML Large-LI | 42 from Phase 1 | 43, 44 |

`run/cdvt_phase3_queue.sh` materializes one immutable config per new task,
uses every genuinely idle GPU, polls every 300 seconds, and refuses an existing
result root. `run/cdvt_phase3_summary.py` verifies all nine manifests and
reports every real seed plus sample mean and sample standard deviation.

## Minimal Training Ablation

Only eight missing seed-42 tasks are trained:

1. Medium-LI account-only;
2. Medium-LI event-only;
3. no-relation CDVT on Small-LI;
4. no-relation CDVT on Medium-LI;
5. no-relation CDVT on Large-LI;
6. `K=2` CDVT on Small-LI;
7. `K=2` CDVT on Medium-LI; and
8. `K=2` CDVT on Large-LI.

The summary reuses Small-LI and Large-LI account/event controls from Phase 1
and standard CDVT seed-42 runs from Phase 1/2. It does not retrain these tasks.

### No-Relation Semantics

The no-relation control removes all explicit account-role transition identity:

- relation key embeddings are not used;
- relation value embeddings are not used;
- four relation one-hot transition columns are zeroed; and
- the role-change transition indicator is zeroed.

Time gap, amount change, currency change, payment-format change, event
connectivity, event ordering, model width, and registered parameter count are
held fixed. This prevents relation identity from leaking through continuous
edge features after the embedding pathway is disabled.

### Mechanism Controls Require No Retraining

Every standard CDVT validation-selected checkpoint already evaluates normal,
shuffled, and off event conditions on aligned test batches. The ablation
summary extracts these three F1 values from `best_event.test` and checks that
normal exceeds the stronger counterfactual by at least 0.005. Mechanism support
is required on at least two representative datasets.

## Computational Cost

The training runner's `test_seconds` includes normal, shuffled, and off forward
passes and is not a fair inference-latency measure. It must be described as
three-condition evaluation time only.

After the ablation manifests are complete, `run/cdvt_runtime_queue.sh` loads
the validation-selected account-only and CDVT checkpoints on Small-LI,
Medium-LI, and Large-LI. It runs four warm-up batches followed by 256
normal-only dynamic-random test batches. The resulting six benchmark records
report:

- parameter count;
- peak GPU memory;
- seconds per batch;
- seconds per target; and
- CDVT-to-account ratios.

The runtime queue refuses to start until both the Phase 2 gate and complete
ablation summary pass.

## Task Budget

| Stage | New training tasks | Checkpoint-only tasks |
|---|---:|---:|
| Phase 3 real seeds | 6 | 0 |
| Core/relation/K ablation | 8 | 0 |
| Normal-only runtime | 0 | 6 |
| Total | 14 | 6 |

No full grid search, duplicate final-model seed-42 training, or separate
normal/shuffled/off training is permitted.

For formal execution, `run/cdvt_followup_queue.sh` owns all 14 training tasks
through one GPU allocation table. This avoids a race in which independent
Phase 3 and ablation queues could observe the same GPU as idle. After all
training manifests and both summaries are complete, the same orchestrator
starts the six normal-only checkpoint benchmarks. The stage-specific queues
remain available only for controlled recovery or isolated reruns.

## Expected Outputs

- `phase3_summary.json` and `phase3_summary.md`;
- `ablation_summary.json` and `ablation_summary.md`;
- `runtime_summary.json` and `runtime_summary.md`;
- per-task materialized YAML, manifest, checkpoint, progress, and logs; and
- queue manifests containing Git commit and `sampling_protocol=dynamic_random`.
