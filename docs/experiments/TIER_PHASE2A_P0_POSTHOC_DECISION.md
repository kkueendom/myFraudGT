# TIER Phase 2a-P0 Post-Hoc Decision

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: transparent post-hoc protocol amendment
- P0 experiment commit: `3eca8008`
- Sampling protocol: `dynamic_random`
- Formal P0 decision: fail
- Paper-result eligibility: none

## Formal Result

No task passed all three registered stages:

1. train-only calibration;
2. frozen validation;
3. locked test.

Therefore P0 does not authorize CrossFusion, a final ErrorRouter, or any paper
claim. Its test results cannot be used as formal confirmation.

## Exploratory Signal

One family and selection had the same paired-F1 direction on both scales:

| Dataset | Family/selection | Validation delta | Test delta | Validation changed | Test changed |
|---|---|---:|---:|---:|---:|
| Small-LI | flow-role/recent | +0.00738 | +0.01219 | 26 | 36 |
| Large-LI | flow-role/recent | +0.02139 | +0.01896 | 61 | 58 |

Large-LI passed the locked validation and test gates but failed train-only
calibration because the selected calibration policy changed only 6 samples,
below the registered minimum of 10. Small-LI remained below the 50-change
evaluation coverage gate.

## Why P0 Is Not a Strict Upper Bound

P0 uses errors from an A2 model evaluated on the same training split on which
A2 was fitted. This is optimistic because the utility probes can see unusually
clean A2 confidence patterns. It is simultaneously pessimistic for error
coverage: an in-sample A2 makes fewer errors and produces fewer disagreement
examples than an out-of-fold A2.

The calibration failures therefore do not prove that an out-of-fold utility
target is unlearnable. They do prove that the current in-sample training
protocol is inadequate and cannot be used as formal router supervision.

## Protocol Amendment

The preregistered P0 stop rule is not relaxed. Instead, one new exploratory
experiment is registered after observing P0:

> Run target-edge cross-fitting only for the preselected
> `flow_role/recent` evidence path to determine whether realistic OOF A2 errors
> provide enough direction-specific utility supervision.

This is a protocol correction, not a retrospective P0 pass.

## Contamination Control

Because Small-LI and Large-LI test outcomes informed this amendment:

- cross-fit development uses training OOF targets and validation only;
- the current Small-LI/Large-LI test split is not used for architecture or
  threshold selection in the cross-fit development phase;
- formal confirmation requires new seeds and the untouched expansion datasets;
- all P0 findings are labeled exploratory in the paper record.

## OOF Feasibility Gate

Use three deterministic target-edge folds. For each fold, both A2 and the
flow-role evidence encoder exclude held-out target edges from supervised loss
while retaining them as unlabeled graph context. Combine held-out predictions
across folds.

Before any validation evaluation, require:

- at least 50 OOF add/remove disagreement candidates on each scale;
- both utility classes represented for at least one direction;
- utility AUPRC above its OOF prevalence by at least `0.05`;
- normal evidence utility AUPRC at least `0.02` above shuffled evidence;
- a train-OOF calibrated policy with positive paired F1 delta,
  corrected/broken at least `1.5`, and at least 50 interventions.

Only an OOF-qualified probe is evaluated on validation. Validation must then
show positive paired F1 delta, corrected/broken at least `1.5`, and at least
50 interventions on Small-LI and Large-LI. Test remains unopened during this
development decision.

If this amended OOF feasibility experiment fails, TIER stops without another
router or decoder variant.
