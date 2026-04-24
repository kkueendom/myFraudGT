# FraudGT Autonomous Experiment Rules

## Objective
- The target is no longer just AML Small-HI.
- Relative to the original FraudGT paper, achieve at least `+2.00` F1 points in **raw peak test F1** on **every** AML dataset in Table 2.
- Use the paper's strongest published FraudGT-family result on each dataset as the baseline target unless the user explicitly changes the comparison rule.
- The completion criterion is raw peak, not formal best-by-val.
- Formal best should still be tracked, but it is now a secondary reference metric.
- Current per-dataset raw-peak thresholds are:
- `Small-HI > 78.13`
- `Small-LI > 49.01`
- `Medium-HI > 77.93`
- `Medium-LI > 46.06`
- `Large-HI > 75.34`
- `Large-LI > 39.43`

## Autonomy
- Operate with high autonomy.
- Do not pause between experiments for approval.
- Only ask the user when there is a real branching decision that cannot be resolved from code, logs, papers, or prior results.

## Research Standard
- Read and understand the FraudGT paper before proposing new model changes.
- Favor structural improvements that transfer across HI / LI and graph scales, not just one dataset.
- When progress stalls, consult relevant primary-source papers and borrow structural ideas.
- Favor model-side or architecture-side innovations over pure tuning.

## Experiment Policy
- Monitor every launched training job to completion or explicit early-stop.
- While one run is training, prepare the next structural candidate instead of waiting idly.
- If a run does not beat the relevant paper baseline trajectory, launch the next experiment automatically.
- Use short pilot runs only as screening; do not treat them as final evidence when the model is known to peak late.
- Prefer one coherent model family with dataset-specific configs over unrelated per-dataset hacks.
- Do not judge a run from only the first evaluation point.
- If a run is clearly dead, allow `1-3` evaluation points before stopping.
- If a run is borderline, watch at least `5` evaluation points before deciding.
- If a run is promising, let the `40`-epoch pilot finish and judge primarily by raw peak, with formal best recorded as secondary evidence.

## Git Discipline
- Every code change must be tracked with git on the server repository.
- Every code change must also be tracked in the local editing repository.
- Before starting a new experiment, make sure the tracked server state is clean and intentional.
- If an experiment fails, revert the failed commit explicitly instead of leaving stale code in place.
- If an experiment fails, revert the failed commit on both local and remote branches.
- After a revert, resync local scratch from the current tracked server files before making more edits.
- Do not rewrite history or use destructive reset commands.

## Server Context
- Remote repo: `/e/yyk/FraudGT_multi6`
- Local editing repo: `/Users/kun/FraudGT_multi6`
- Conda env: `fraudgt_dual_gate`
- Data root: `/e/yyk/data/archive`

## Monitoring
- Estimate runtime before long runs.
- Use sleep-based polling to watch logs and processes during training.
- Keep track of both raw epoch metrics and the formal best-by-val checkpoint metric.
- For `AML-Small-LI` within the `supportmixconsis` family, do not early-stop from only `epoch 1` or `epoch 3`.
- The known baseline trajectory can remain near-zero through `epoch 3` and only becomes clearly informative around `epoch 7+`.
- For low-risk `supportmixconsis` structural variants on `Small-LI`, watch at least through `epoch 7`; if still ambiguous, extend to `epoch 9` before stopping.

## Innovation Filter
- Prefer changes that alter representation, context fusion, temporal modeling, subgraph reasoning, flow / role transition modeling, or edge decoding structure.
- Do not rely on hyperparameter-only changes as the main contribution.
- Reject candidates that help only AML Small-HI while collapsing on LI or larger datasets.
