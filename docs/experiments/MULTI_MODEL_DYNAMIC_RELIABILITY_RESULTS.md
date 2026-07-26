# Multi-Model Dynamic Evidence Reliability Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: ANALYZED
- Sampling protocol: `dynamic_random`
- Manifests: 31
- Dynamic val/test events: 220
- Evidence model families: 3
- Evidence model/dataset units: 10
- Training or tuning in this benchmark: none
- A2 input: `docs/experiments/A2_DYNAMIC_SAMPLING_STABILITY_RESULTS.json`
- CET input: `docs/experiments/CET_DYNAMIC_RELIABILITY_RESULTS.json`
- TIER input: `docs/experiments/TIER_DYNAMIC_RELIABILITY_RESULTS.json`
- COSTAR input: `docs/experiments/COSTAR_DYNAMIC_RELIABILITY_RESULTS.json`

## Family-Level Results

| Family | Units | Events | Mismatch units | Mismatch datasets | Repeatable mismatch units | Repeatable datasets | Useful aligned | Inactive | Cross-scale repeatable mismatch | Event reversal |
|---|---:|---:|---:|---|---:|---|---:|---:|---|---:|
| CET | 4 | 28 | 4 | Large-LI, Small-LI | 3 | Large-LI, Small-LI | 0 | 0 | true | 1 |
| TIER | 4 | 64 | 4 | Large-LI, Small-LI | 4 | Large-LI, Small-LI | 0 | 0 | true | 0 |
| COSTAR | 2 | 32 | 1 | Large-LI | 0 | none | 0 | 1 | false | 2 |

Mismatch includes `sensitive_but_harmful` and `used_but_unaligned`. It does not relabel inactive evidence as a sensitivity-versus-utility mismatch. A family counts toward the advancement gate only when at least one mismatch unit satisfies its sensitivity and harmful-utility conditions in at least 75% of dynamic events.

## Advancement Gate

| Requirement | Result |
|---|---|
| All fixed evidence blocks completed paired normal/shuffled/off/base evaluation | **pass** |
| At least three independently trained evidence families show repeatable sensitivity/utility mismatch | **fail** |
| Mismatch occurs on both Small-LI and Large-LI in at least one family | **pass** |
| Dynamic sampling changes at least one single-run conclusion | **pass** |
| Sampling variation, sensitivity, utility and checkpoint selection are reported separately | **pass** |

## Aggregate Mechanism Counts

- Mismatch units: 9/10
- Useful-aligned units: 0/10
- Inactive units: 1/10
- A2 datasets whose repeated epoch-499 diagnostic delta crosses both +/-0.005: Large-HI, Large-LI, Medium-LI, Small-LI
- Evidence units with a sensitivity-threshold or utility-sign reversal: 3

## Decision

`STOP_PAPER_LEVEL_MULTI_MODEL_GENERALITY_CLAIM`

The preregistered cross-model generality gate is not fully satisfied. The completed work supports a rigorous negative benchmark and mechanism taxonomy, but not yet a paper-level claim that one new reliability method generalizes across three evidence families.

Repeated events are not independent model seeds. No IID p-value or finite-sample guarantee is claimed. A strong method-paper claim still requires graph/time-blocked risk units, explicit dependence assumptions and nonzero out-of-sample corrective coverage.

