# CDVT Additive-Fusion Control Deployment Record

## Purpose

The additive control isolates the contribution of representation-level
cross-attention. It retains the FraudGT account view and causal event encoder,
but replaces account-query/event-key-value attention with projected-state
addition. No model-selection or training setting differs from frozen CDVT.

## Source Identity

- Source branch: `feature/cdvt-phase3-experiments`
- Audited source commit: `92711440e8c39b6f741b3ff2decfd4eaf776fa61`
- Architecture freeze: `9038f85d63634a91c712302813aad221a4e907e5`
- Portable execution commit: `7acb1eb3bb93d1dbe00217e3c00baa381e70ba57`
- Source and portable tree: `ba5d578f682f9c8b02ba052bedc422a75569eb9f`
- Portable ref: `portable/cdvt-additive-7acb1eb`
- Runtime branch after clone: `feature/cdvt-phase3-experiments`

## Portable Artifacts

- Local Bundle: `/Users/kun/CDVT_additive_portable_7acb1eb.bundle`
- Bundle SHA-1: `e03beb12c99296186a5f8618ce7fb97e492e5783`
- Local launcher:
  `/Users/kun/CDVT_additive_portable_7acb1eb_remote_launch.sh`
- Launcher SHA-1: `2b56b25ab6c99a0365adf9d5efdcb3cd69bbc78a`
- Fresh-clone verification: passed
- Local shell/Python syntax checks: passed
- Local non-PyTorch follow-up tests: 11 passed; four PyYAML-dependent tests
  skipped because the system Python lacks PyYAML

## Remote Contract

- Host: `yky@10.168.1.101`
- Remote Bundle: `/e/yky/CDVT_additive_portable_7acb1eb.bundle`
- Remote launcher:
  `/e/yky/CDVT_additive_portable_7acb1eb_remote_launch.sh`
- Remote worktree: `/e/yky/FraudGT_cdvt_additive_7acb1eb`
- Result root: `/e/yky/FraudGT_cdvt_results/additive_7acb1eb`
- Required completed follow-up root:
  `/e/yky/FraudGT_cdvt_results/followup_f7209f2`
- Python: `/d/miniconda3/envs/fraudGT/bin/python`

The launcher refuses to run unless the unified follow-up queue has a successful
`queue_complete.json`. This prevents the three additive tasks from racing the
active multi-seed, ablation, and runtime allocator.

## Workload

| Dataset | Seed | Experiment label | Architecture | Protocol |
|---|---:|---|---|---|
| Small-LI | 42 | `causal_event_add` | `additive_view` | `dynamic_random` |
| Medium-LI | 42 | `causal_event_add` | `additive_view` | `dynamic_random` |
| Large-LI | 42 | `causal_event_add` | `additive_view` | `dynamic_random` |

All tasks use 500 maximum epochs, validation-selected checkpointing, early
stopping after epoch 80 with ten stale evaluation events, batch size 2048, 256
iterations per split, `K=4`, typed relations, and `lambda_cons=0`.

## Launch Status

Portable files are prepared and uploaded. Training is intentionally not
started until the existing `followup_f7209f2` queue completes successfully.
The remote launcher reruns the complete `test_cdvt*.py` suite before launch;
those remote results must be recorded before any additive result is accepted.
