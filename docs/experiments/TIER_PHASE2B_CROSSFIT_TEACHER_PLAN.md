# TIER Phase 2b Cross-Fitted Teacher Generation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: post-hoc exploratory protocol correction
- Parent decision: `TIER_PHASE2A_P0_POSTHOC_DECISION.md`
- Sampling protocol: `dynamic_random`
- Validation/test labels: not accessed

## Objective

Generate realistic out-of-fold A2 errors and out-of-fold flow-role evidence
scores for direction-specific utility qualification. This stage does not train
the final router and does not evaluate validation or test.

## Cross-Fit Construction

For Small-LI seed 42 and Large-LI seed 44:

1. assign every training target edge to one of three deterministic folds using
   its global edge ID;
2. for fold `k`, train A2 while excluding fold `k` outputs from supervised
   loss;
3. train a flow-role/recent EvidenceEncoder with the same held-out loss mask;
4. retain held-out edges as unlabeled graph context in both models;
5. calibrate A2 and evidence thresholds only on non-held-out training outputs;
6. score only held-out fold `k`;
7. save normal, shuffled, and off evidence scores with edge IDs and support.

The three held-out outputs form one OOF training table per dataset.

## Fast Feasibility Budget

- three folds per dataset, six tasks total;
- A2: 80 epochs;
- EvidenceEncoder: 60 epochs;
- original batch size, 256 dynamic training iterations per epoch;
- fixed epoch budget, no held-out-label early stopping;
- class weight 6 and existing optimizer/scheduler settings.

This budget trains utility teachers, not a headline baseline. Any later formal
run must test teacher-budget sensitivity.

## Integrity Requirements

- `sampling_protocol=dynamic_random`;
- `LinkNeighborLoader(shuffle=True)`;
- no fixed target panel or sampler RNG restoration;
- fold membership derives only from global training edge ID;
- held-out fold contributes zero supervised loss;
- no validation or test loader iteration;
- one row per unique OOF edge after aggregation;
- normal/shuffled/off scores use the same held-out A2-scored edge set;
- each task records commit, config, checkpoints, fold, coverage, and thresholds.

## Continue/Stop Rule

After all six tasks finish, merge by dataset and run the separately committed
OOF utility audit. Do not train a validation-facing router unless the registered
OOF feasibility gate in the P0 post-hoc decision passes on both scales.
