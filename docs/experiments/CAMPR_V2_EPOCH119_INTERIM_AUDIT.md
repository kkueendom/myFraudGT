# CAMPR V2 Epoch-119 Interim Hard-Gate Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Decision Date: 2026-07-18
- Verification Status: MATCHED EPOCH-119 INTERIM / 500-EPOCH RUNNING
- Branch: `feature/campr-counterfactual-prototype-router`
- V2 Commit: `e48a093`
- Decision: `HIGH RISK / CONTINUE TO 500`

## 1. Interim Decision

CAMPR V2 passes on Large-LI but fails materially on Small-LI at epoch 119. This
does not satisfy an early two-dataset screen, but it is not the formal decision:

> A candidate advances only when validation-selected test F1 exceeds matched
> A2 on both Small-LI and Large-LI after the complete 500-epoch schedule.

The jobs were briefly stopped under the earlier epoch-119 rule, then resumed
from checkpoints that include model, optimizer, and scheduler state. Large-LI
resumes from epoch 100 and Small-LI from epoch 150. No short-budget scheduler is
introduced, so the final comparison remains a 500-epoch matched test.

## 2. Matched Epoch-119 Results

| Dataset | Variant | A2 val-select | Current val-select | Delta | A2 raw | Current raw | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Large-LI | CAMPR full | 0.08955 | 0.14118 | +0.05163 | 0.21583 | 0.23656 | +0.02073 |
| Large-LI | CAMPR no-aux | 0.08955 | 0.16162 | +0.07207 | 0.21583 | 0.24359 | +0.02776 |
| Small-LI | CAMPR full | 0.47308 | 0.42379 | -0.04929 | 0.47445 | 0.49904 | +0.02459 |
| Small-LI | CAMPR no-aux | 0.47308 | 0.45253 | -0.02055 | 0.47445 | 0.47212 | -0.00233 |

The primary metric is validation-selected test F1. Raw-best is diagnostic and
cannot rescue the failed Small-LI result because it uses test labels for epoch
selection.

## 3. Auxiliary-Loss Ablation

Full CAMPR minus no-aux validation-selected delta:

- Large-LI: `-0.02044`;
- Small-LI: `-0.02874`;
- mean: `-0.02459`;
- wins: `0/2`.

The scale-normalized target fixed V1's numerical target collapse, but the
counterfactual supervision itself did not improve the primary metric.

## 4. Mechanism Diagnosis

### 4.1 Benefit is not sufficiently predictable from current router inputs

V2 target standard deviation remained nonzero (`0.056-0.153` in later sampled
diagnostics), but validation route standard deviation was usually only
`0.001-0.012`. The router learned an almost constant allocation even though its
training target varied. The ten base/prototype scalar inputs therefore do not
appear sufficient to predict which samples benefit from the residual.

### 4.2 Mean preservation prevented P0 collapse but did not create useful routing

The route mean remained exactly `1.0`, so CAMPR avoided P0's global residual
attenuation. This invariant is successful and should be retained. However, an
almost constant route reduces CAMPR to A2 plus optimization noise and cannot
provide the claimed sample-level benefit.

### 4.3 The effect is regime dependent

Both full and no-aux variants helped Large-LI but hurt Small-LI. This pattern
suggests that the current router/residual interaction changes training dynamics
rather than learning a dataset-independent correction rule.

### 4.4 The counterfactual target and inference objective are mismatched

The target uses current labels to measure realized loss reduction, while the
inference router sees only coarse label-free scalar summaries. Much of the
per-sample target may be irreducible from those summaries. Increasing the
auxiliary weight alone is therefore unlikely to solve the problem.

## 5. Preserved Evidence and Resume State

- V1 fixed-temperature pilot: `results/campr_formal500/`;
- V2 scale-normalized pilot: `results/campr_formal500_v2/`;
- audit: `run/campr_formal_audit.py --epoch-limit 119`;
- V1 failure: target standard deviation `0.0004-0.0008`;
- V2 repair smoke: target standard deviation `0.2375` after prototype startup;
- pause points before resume: approximately Large-LI `120`, Small-LI `194`;
- scheduler-preserving resume checkpoints: Large-LI epoch `99`, Small-LI epoch
  `149`;
- `run/campr_formal_audit.py` deduplicates resumed JSONL rows by epoch and keeps
  the last record, preventing old validation rows from being paired with the
  resumed trajectory's test rows;
- the unrelated Medium-HI pilot remains paused and is not part of the current
  first-stage decision.

## 6. Constraints for CAMPR V2 and the Next Candidates

The next method must:

1. preserve exact A2 fallback and mean residual scale;
2. use richer label-free evidence than ten scalar summaries;
3. demonstrate nontrivial but non-saturated sample routing;
4. beat matched A2 at epoch 499 on both Small-LI and Large-LI before expansion;
5. use the same 500-epoch scheduler and matched cutoff for every comparison;
6. be implemented and committed as a distinct, reproducible variant.
