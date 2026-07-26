# GTPRC Phase 0A v2b Development Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent result: `GTPRC_PHASE0A_V2A_DEV_RESULTS.md`
- Scope: simulator identifiability only
- AML validation/test labels allowed: no

## Failure-Driven Changes

v2a separated row-IID and GTPRC coverage but row-IID never violated the harm
limit. v2b changes exactly two generator-interface properties:

1. increase candidate policies from 16 to 64 quantile thresholds so a policy
   can lie closer to the harm boundary;
2. remove the latent shared-cluster effect from the aligned score. The score
   may use noisy outcome-correlated information, but it cannot directly
   observe the variable that induces dependence.

The following remain fixed:

- all seven outcome-generating regimes;
- `alpha=0.40` and `delta=0.05`;
- calibration and evaluation row counts;
- row-IID, time-block, entity-block, and graph-time methods;
- group Hoeffding formula;
- shuffled and harmful controls;
- all v2 development and formal gates.

The manifest additionally records mean and median selected test break risk for
each method. This is diagnostic only and does not select a policy.

## Development Run

- seven GPUs;
- 64 replicates per regime;
- development seeds remain the v2 development seed family;
- output directory is new and cannot overwrite v2a.

The v2 development gate from
`GTPRC_PHASE0A_V2_STRESS_PLAN.md` applies unchanged.

## Formal Consequence

If v2b passes, formal v2 uses the locked 64-policy grid and new formal seeds.
If v2b fails, formal v2 remains prohibited. Any further change requires
another explicit failure analysis and preregistration.

