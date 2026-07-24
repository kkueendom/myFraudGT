# TIER Phase 2a-P0 Directional Utility Feasibility Probe

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: preregistered optimistic feasibility screen
- Parent evidence commit: `c89dc527`
- Sampling protocol: `dynamic_random`
- Status as evidence: diagnostic upper bound, not a paper result

## Purpose

Phase 1c showed that fixed confidence rules can find high-ratio interventions,
but not at the registered coverage, and that negative and positive overrides
have different effects on fraud F1. This probe asks:

> Is direction-specific override utility learnable at all from detached A2
> confidence, independent evidence confidence, disagreement, and raw support?

It is intentionally cheaper than cross-fitted teacher training. The original
A2 and Phase 1b evidence checkpoints are frozen. Because A2 was trained on the
same training split used to form utility targets, this is an optimistic
feasibility upper bound. A pass authorizes proper target-edge cross-fitting; it
does not authorize a paper claim or full CrossFusion.

## Directional Targets

For samples on which frozen A2 and frozen evidence disagree:

- `add`: A2 predicts normal and evidence predicts fraud;
- `remove`: A2 predicts fraud and evidence predicts normal.

Each probe predicts whether taking that direction-specific override would be
correct. The two probes never share their final action threshold.

Inputs are detached and contain no labels:

- A2 score and confidence margin;
- evidence score and confidence margin;
- score difference and interaction;
- log context support count.

No validation or test labels update the probes or their thresholds.

## Train-Only Fitting and Calibration

Dynamic training samples are aggregated by global edge ID. A deterministic hash
of global edge ID assigns each candidate to:

- 80% probe fitting;
- 20% train-only threshold calibration.

The add/remove probes are small MLPs trained with class-balanced binary
cross-entropy. Train-only calibration jointly selects the add and remove
thresholds. A calibration policy must:

- improve paired F1 over A2;
- correct more predictions than it breaks;
- have corrected/broken at least `1.5`;
- correct at least `10%` of A2 errors;
- change at least 10 calibration predictions.

If no policy passes, the best diagnostic policy is retained but the task fails.

## Locked Validation and Test Gates

The calibrated probe and thresholds are frozen before validation. Both
validation and test must independently satisfy:

- paired F1 delta greater than zero;
- correction rate on A2 errors at least `10%`;
- corrected/broken at least `1.5`;
- corrected minus broken greater than zero;
- at least `max(50, 10% of A2 errors)` changed predictions.

The full task passes only when calibration, validation, and test all pass.

## Matrix and Stop Rule

Run the same 12-task matrix as Phase 1b/1c:

- Small-LI seed 42 and Large-LI seed 44;
- structure, temporal, and flow-role evidence;
- recent and role-motif selection.

All tasks may run concurrently across available GPUs.

| Outcome | Decision |
|---|---|
| Same family passes both scales | Train proper target-edge cross-fitted A2/evidence teachers |
| Passes one scale only | Diagnose scale-conditioned utility; no universal router |
| No cross-scale pass | Stop TIER before expensive cross-fitting or CrossFusion |

The preregistered gates may not be relaxed after viewing results.
