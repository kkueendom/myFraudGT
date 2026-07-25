# TIER Phase 2b OOF Utility Qualification

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: preregistered OOF qualification
- Input teacher commit: `c78214c1`
- Sampling protocol: `dynamic_random`
- Validation/test labels: not accessed

## Purpose

Merge the three target-edge folds for each dataset and determine whether
flow-role/recent evidence can predict direction-specific override utility from
training-only OOF errors.

## Three-Way OOF Split

Global edge ID deterministically assigns every merged OOF row to:

- 60% probe fitting;
- 20% action-threshold calibration;
- 20% locked OOF evaluation.

The add and remove probes are independent. Probe inputs are limited to detached
A2 score, evidence score, both margins, score disagreement, score interaction,
and support count. Labels never enter the feature vector.

Probe initialization and optimization are deterministic from the dataset seed
and direction, and the derived probe seeds are stored in the checkpoint.

The threshold calibration subset may select add/remove action thresholds, but
the final qualification gate is evaluated only on the locked OOF subset.

## Counterfactual Evaluation

Normal, shuffled, and off evidence scores are evaluated on the same normal
direction-candidate rows. Holding the candidate population fixed prevents a
counterfactual from appearing better merely because it creates an easier set
of disagreements.

The separately saved support count is held fixed in this comparison. The
shuffled/off evidence-model scores already include the corresponding
counterfactual context path; therefore the reported AUPRC gap isolates whether
that evidence score contributes beyond detached A2 confidence and support.

## Qualification Gate

Each dataset must satisfy all of:

- at least 50 normal direction candidates;
- at least one direction has both positive and negative utility labels;
- that direction's normal utility AUPRC exceeds utility prevalence by `0.05`;
- normal utility AUPRC exceeds shuffled utility AUPRC by `0.02`;
- the locked OOF policy has:
  - paired F1 delta greater than zero;
  - corrected/broken at least `1.5`;
  - corrected greater than broken;
  - at least 50 changed predictions.

Small-LI and Large-LI must both pass. Otherwise TIER stops and no validation
loader, test loader, additional router threshold, or CrossFusion experiment is
authorized.
