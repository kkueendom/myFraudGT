# TIER Phase 1b/1c Failure Analysis and Next Step

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: post-experiment mechanism diagnosis
- Phase 1b evidence commit: `c89dc527`
- Phase 1c diagnostic commit: `27c9e72b`
- Sampling protocol: `dynamic_random`
- Decision: current evidence-replacement and confidence-rule routes stop

## What the Experiments Established

The support-masked Phase 1b matrix completed all 12 registered tasks. Every
family had a material normal-versus-shuffled/off gap, so historical raw-graph
context is genuinely used. No family safely replaced A2 predictions:

| Family | Best corrected/broken | Mean Val-selected delta | Decision |
|---|---:|---:|---|
| structure | 0.53448 | -0.31093 | informative but unsafe |
| temporal | 0.14773 | -0.31890 | informative but unsafe |
| flow_role | 0.33929 | -0.12893 | informative but unsafe |

Phase 1c then froze A2 and the evidence models and searched validation-only
high-precision intervention policies. All 12 tasks completed, but no task had
one validation-eligible policy. The strongest selected rules changed only
2-35 validation predictions, below the preregistered minimum of 50.

## Why a High Corrected/Broken Ratio Did Not Guarantee Better F1

Most selected policies used the `evidence_negative` direction: A2 predicted
fraud, while the evidence model predicted normal. This can correct many A2
false positives. In a highly imbalanced fraud dataset, however, breaking one
true-positive fraud prediction can cost more F1 than correcting one normal
false positive helps.

Examples:

- Large-LI temporal/role-motif: 22 corrected, 2 broken, paired F1 `+0.01744`;
- Small-LI flow-role/recent: 32 corrected, 13 broken, paired F1 `-0.01385`;
- Small-LI temporal/role-motif: 38 corrected, 16 broken, paired F1 `-0.01360`.

The count ratio is therefore a safety diagnostic, not a sufficient routing
objective. A single symmetric override gate conflates two different decisions:

1. rescue an A2 false negative by adding a fraud prediction;
2. suppress an A2 false positive by removing a fraud prediction.

Their class prevalence, error costs, and useful evidence differ.

## Method Decision

The current TIER implementation does not authorize a learned ErrorRouter or
CrossFusion. Implementing them now would add capacity after both registered
qualification stages failed.

The independent-evidence hypothesis is not rejected: counterfactual evidence
effects were present in every task, and Large-LI temporal evidence produced a
positive mean paired F1 delta. What failed is the formulation of evidence as a
standalone fraud classifier followed by a symmetric replacement decision.

## Recommended Next Method: Directional Utility Routing

The next minimal method should replace the binary override gate with two
direction-specific utility estimates learned only from training data:

```text
u_fn = P(A2 is a false negative and evidence can rescue it)
u_fp = P(A2 is a false positive and evidence can suppress it)

q_add    = RouterAdd(h_base, h_evidence, disagreement, support)
q_remove = RouterRemove(h_base, h_evidence, disagreement, support)
```

The final decision uses `q_add` only on A2-negative candidates and `q_remove`
only on A2-positive candidates. The loss must weight the two utilities by their
effect on fraud-class precision and recall rather than treating every corrected
sample as equally valuable.

This remains one unified contribution:

> independent raw-graph evidence is converted into direction-specific utility,
> then used to repair distinct A2 error types without globally replacing A2.

## Required Training Supervision

Router labels must be generated from training targets only. To avoid teaching
the router from in-sample A2 confidence, use target-edge cross-fitting:

1. deterministically assign training target edges to folds by global edge ID;
2. train one A2 teacher per fold while excluding that fold from supervised
   loss, while allowing its edges to remain unlabeled graph context;
3. score only the excluded fold;
4. combine all held-out predictions into out-of-fold A2 error labels;
5. train the directional utility probe on those labels;
6. never use validation or test labels in router state.

## Phase 2a Qualification Before Full Fusion

First run a low-cost directional utility probe on Small-LI seed 42 and Large-LI
seed 44. The probe may use frozen raw-evidence representations but may not train
CrossFusion.

Required outputs:

- false-negative-rescue AUPRC and precision at registered coverage;
- false-positive-suppression AUPRC and precision at registered coverage;
- routed paired F1 against cross-fitted A2;
- normal, shuffled, and off evidence comparisons;
- direction-specific corrected, broken, and intervention counts;
- calibration error and reliability bins;
- coverage by dataset scale and fraud class.

Advance only when the same direction-aware design:

- improves validation-selected paired F1 on both Small-LI and Large-LI;
- changes at least 50 predictions per evaluation stream;
- has positive direction-weighted net utility;
- retains a normal-versus-shuffled F1 gap of at least `0.010`;
- shows no validation/test label leakage;
- remains positive under at least two independent cross-fit assignments.

Only then implement representation-level CrossFusion and the full directional
ErrorRouter. If Phase 2a fails, stop TIER and change the paper question rather
than adding more decoder modules.
