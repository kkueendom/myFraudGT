# Multi-CDVT Screen Deployment Record

## Version Identity

- Branch: `feature/cdvt-multi-backbone-screen`
- Core screen implementation: `2d9dc11`
- Upstream Multi configuration audit: `b0dc86e`
- Paper baseline alignment descendant: `49bc6fb`
- Final execution commit: `8dac98d5bc5bf7c6ec908d72cae9a9032ddc41dc`
- Frozen PE-CDVT architecture ancestor: `9038f85`
- Multi-CDVT changes only the account backbone by enabling reverse message
  passing while retaining Ports, Ego ID, the frozen event graph, relation-aware
  event encoder, cross-attention and `lambda_cons=0`.

## Portable Artifacts

| Artifact | Local path | Remote path | SHA-1 |
|---|---|---|---|
| Self-contained bundle | `/Users/kun/CDVT_multi_screen_8dac98d.bundle` | `/e/yky/CDVT_multi_screen_8dac98d.bundle` | `3c0a05c4af5e0ee3b2f88fce32ba803d79ee65b8` |
| Watch-and-launch script | `/Users/kun/CDVT_multi_screen_8dac98d_remote_watch_and_launch.sh` | `/e/yky/CDVT_multi_screen_8dac98d_remote_watch_and_launch.sh` | `010b3a1d073961d243f49f54ae436d81ac766e6c` |

The bundle passed `git bundle verify`, a real fresh clone, exact commit
identity, Python compilation and shell syntax checks. Full PyTorch tests remain
an enforced remote precondition because the local Python installation does not
contain PyTorch or PyTorch Geometric.

## Remote Control

- Host: `yky@10.168.1.101`
- Watcher PID: `707963`
- Earliest follow-up inspection: `2026-08-02 22:35 CST`
- Follow-up inspection interval if still running: 8 hours
- Watcher status:
  `/e/yky/.CDVT_multi_screen_8dac98d_watcher_status.json`
- Watcher log: `/e/yky/CDVT_multi_screen_8dac98d_watcher.log`
- Fresh-clone target: `/e/yky/FraudGT_cdvt_multi_8dac98d`
- Multi GPU smoke root:
  `/e/yky/FraudGT_cdvt_results/multi_smoke_8dac98d`
- Unified post-followup root:
  `/e/yky/FraudGT_cdvt_results/post_followup_8dac98d`
- Unified allocator PID file:
  `/e/yky/FraudGT_cdvt_results/.post_followup_8dac98d_queue.pid`

The watcher was observed in `initial_sleep` state after launch. It does not
inspect or modify the active follow-up queue before the registered time.

## Enforced Launch Sequence

1. Require the existing `followup_f7209f2/queue_complete.json` to contain
   `complete=true` and `status=0`.
2. Refuse every existing code, smoke or result output path.
3. Verify the remote bundle SHA-1 and fresh-clone exact commit identity.
4. Run all `test_cdvt*.py` tests, compile the four execution scripts and run
   shell syntax validation.
5. Materialize a `multi_cdvt` mixed-class GPU smoke config and require RMP,
   Ports, Ego ID, positive account/event/fusion gradients, positive overfit
   loss reduction and complete train/val/test reverse-relation audits.
6. Start exactly one allocator containing three additive controls and six
   matched Multi-FraudGT/Multi-CDVT training tasks.

Any failed infrastructure, protocol, test or smoke check stops before the
allocator is created. The later scientific Multi Gate is not treated as an
engineering failure: its result determines whether the experiment expands.
