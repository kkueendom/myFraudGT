# FraudGT Autonomous Experiment Rules

## Objective
- Continue structural model experimentation on AML Small-HI until the formal best-by-val test F1 reaches at least 0.80.
- Use the paper-reported baseline as the reference point. Do not spend time re-running the original baseline unless strictly required.

## Autonomy
- Operate with high autonomy.
- Do not pause between experiments for approval.
- Only ask the user when there is a real branching decision that cannot be resolved from code, logs, papers, or prior results.

## Research Standard
- Read and understand the FraudGT paper before proposing new model changes.
- When progress stalls, consult relevant papers and borrow structural ideas from primary sources.
- Favor model-side or architecture-side innovations over pure tuning.

## Experiment Policy
- Monitor every launched training job to completion.
- While one run is training, prepare the next structural candidate instead of waiting idly.
- If a run does not beat the current target, launch the next experiment automatically.
- Use short pilot runs only as screening; do not treat them as final evidence when the model is known to peak late.

## Git Discipline
- Every code change must be tracked with git on the server repository.
- Before starting a new experiment, make sure the tracked server state is clean and intentional.
- If an experiment fails, revert the failed commit explicitly instead of leaving stale code in place.
- After a revert, resync local scratch from the current tracked server files before making more edits.
- Do not rewrite history or use destructive reset commands.

## Server Context
- Remote repo: `/e/yyk/FraudGT`
- Conda env: `fraudgt_dual_gate`
- Data root: `/e/yyk/data`

## Monitoring
- Estimate runtime before long runs.
- Use sleep-based polling to watch logs and processes during training.
- Keep track of both raw epoch metrics and the formal best-by-val checkpoint metric.

## Innovation Filter
- Prefer changes that alter representation, context fusion, temporal modeling, subgraph reasoning, or edge decoding structure.
- Do not rely on hyperparameter-only changes as the main contribution.
