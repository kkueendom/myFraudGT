# CDVT Follow-up Deployment Record

Status: Phase 2 passed and the corrected self-contained follow-up Bundle is
running. Queue PID: `2595374`.

The original watcher executed portable launcher `e364827` after Phase 2
completed. Its 42 tests passed, but the launcher stopped before creating a
result root because direct execution of `cdvt_phase3_gate.py` could not import
the repository-level `run` package. No follow-up training started from that
Bundle. Source commit `34456ab` adds location-independent repository bootstrap
to the affected CLI scripts and a regression test that invokes them from
outside the repository.

## Version Identity

- Source branch: `feature/cdvt-phase3-experiments`
- Architecture freeze: `9038f85d63634a91c712302813aad221a4e907e5`
- Follow-up source commit: `34456abdbd979e71a30bda25ed2619287dfbea81`
- Portable execution commit: `f7209f2b947daa032890df9b1212552264b50b0a`
- Source and portable tree:
  `a5abface6f0abbc732f05df44ab2f3f2b5786024`

The portable commit is parentless because the local source repository is a
shallow clone. Its tree is byte-identical to source commit `34456ab` and its
commit message records the source and architecture-freeze identities.

The source includes the validation-cost optimization and the revised baseline
policy. PE-FraudGT from FraudGT Table 2 is the formal primary baseline because
it is the direct parent architecture used by CDVT's account view.
Multi-FraudGT is reported as the strongest published FraudGT reference, and A2
is retained as an internal strong comparator. Raw-best remains supplementary
and is compared only with A2 raw-best.

The source also includes a dated baseline-policy amendment, corrected Phase 2
and follow-up execution documents, and a Phase 0 GPU smoke that now verifies
fixed-mini-batch loss reduction in addition to gradients and event
interventions. None of these changes modifies the frozen model architecture.

The source includes the mixed-class Phase 0 correction and its authoritative
JSON audit record. The final remote worktree passed all 43 CDVT tests, Python
compilation, shell syntax, commit identity, and Git-cleanliness checks.

## Upload Artifacts

| Artifact | Local path | SHA-1 |
|---|---|---|
| Self-contained follow-up Bundle | `/Users/kun/CDVT_followup_portable_f7209f2.bundle` | `180e2c1fb5167fc6a5b4c33ea0295168efe778f7` |
| Conditional remote launcher | `/Users/kun/CDVT_followup_portable_f7209f2_remote_launch.sh` | `622188a22af96d26930d3f653379502e4b45be46` |

The Bundle passed `git bundle verify`, a real fresh clone, source-to-portable
tree diff, runtime branch rename, Python compilation, and shell syntax checks.
The local machine lacks torch, so the complete suite was run in the remote
FraudGT environment. The older `c3919ed` Bundle is `nonportable`; `2513edc`
predates unified GPU scheduling; and `44f9caf` predates the validation-cost
optimization. The `4095597` Bundle predates the paper-baseline correction.
The `2321e05` Bundle lacks matched three-seed FraudGT controls. All five are
superseded. The `eb17e95` and `db54795` Bundles predate the final
baseline-policy and mixed-class Phase 0 audit record. The `e364827` Bundle
lacks the location-independent CLI import fix. None of these older Bundles may
be used for a new launch.

The 1,232-byte incremental Bundle
`/Users/kun/CDVT_smoke_fix_9be1737.bundle` has SHA-1
`b3cb8a16887b27d2d077483bfa845c829bd2cd83`. It requires `db54795`, was used
only to update the dedicated remote audit worktree, and is not a
self-contained deployment artifact.

## Remote Targets

- Host: `yky@10.168.1.101`
- Bundle: `/e/yky/CDVT_followup_portable_f7209f2.bundle`
- Worktree: `/e/yky/FraudGT_cdvt_followup_f7209f2`
- Phase 1 root:
  `/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd`
- Required Phase 2 root:
  `/e/yky/FraudGT_cdvt_results/phase2_2fb3333`
- Follow-up root:
  `/e/yky/FraudGT_cdvt_results/followup_f7209f2`

## Hard Gate

The launcher and the unified queue independently recompute the Phase 2 summary
from authoritative manifests. They stop before creating the follow-up result
root unless:

1. all six seed-42 datasets are complete;
2. at least four val-selected deltas against PE-FraudGT are positive; and
3. the six-dataset mean val-selected delta is positive.

Multi-FraudGT and A2 comparisons are reported but cannot open or close this
gate. Raw-best F1 cannot open this gate.

The gate passed on 2026-07-31 with 5/6 wins against PE-FraudGT and mean
val-selected delta `+0.0271604333`.

## Unified Workload

`run/cdvt_followup_queue.sh` owns one GPU allocation table for all follow-up
training, preventing two queues from selecting the same idle GPU.

| Stage | New training tasks | Checkpoint-only tasks |
|---|---:|---:|
| CDVT seeds 43/44 on Small-LI, Medium-LI, Large-LI | 6 | 0 |
| Matched FraudGT/account-only seeds 43/44 | 6 | 0 |
| Missing account/event, no-relation, and K=2 controls | 8 | 0 |
| Normal-only runtime benchmarks | 0 | 6 |
| Total | 20 | 6 |

The queue polls every 300 seconds, uses every genuinely idle GPU, never kills
another process, refuses every existing output root, and runs runtime
benchmarks only after training and both summaries complete successfully.

## Launch Verification

The remote environment passed all 43 `test_cdvt*.py` tests, Python
compilation, shell syntax checks, Git cleanliness, Bundle hashes, fresh clone,
the mixed-class GPU smoke, and the formal Phase 2 gate. The unified queue
created its authoritative manifests and assigned its first seven tasks to GPUs
0-6 without reusing an output directory.
