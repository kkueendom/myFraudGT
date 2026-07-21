# CF-RCR: Cross-Fitted Risk-Controlled Residual

## 1. Motivation

A2 is the fixed reference. Previous post-hoc corrections failed mainly because
they learned a nearly global score shift, kept growing after useful gains were
exhausted, or behaved differently across temporal regimes. A validation
threshold can absorb a global shift, so such corrections do not improve the
ranking that determines F1.

CF-RCR keeps the complete A2 path and adds a small committee whose only job is
to propose sample-dependent **margin** corrections. It explicitly learns a
finite ranking gain and falls back exactly to A2 when committee agreement is
insufficient.

## 2. Method

The chronological training edge IDs are divided into three contiguous
environments. Residual expert `k` is optimized using labels from the other two
environments only. All experts receive the same detached input:

- normalized A2 edge representation;
- eight A2 prototype evidence signals;
- bounded base-decoder and A2 margins.

Each expert proposes a bounded correction:

\[
c_i^{(k)}=b\tanh f_k(x_i),\qquad k\in\{1,2,3\}.
\]

For a fixed set of hard positive-negative pairs, the expert compares its
candidate ranking loss with the detached A2 ranking loss. Optimization stops
after a finite target gain is reached:

\[
R_k=\left[L_{rank}(m^{A2}+c^{(k)})
-L_{rank}(m^{A2})+\gamma\right]_+.
\]

The objective combines all included samples, the worst valid temporal group,
and a small correction norm. A microbatch without both classes contributes
zero ranking loss. The ordinary training prediction remains exactly A2; the
auxiliary objective trains only the residual heads.

At inference, the committee mean and disagreement are

\[
\mu_i=\operatorname{mean}_k c_i^{(k)},\qquad
\sigma_i=\operatorname{std}_k c_i^{(k)}.
\]

The final margin is

\[
m_i^{final}=m_i^{A2}
+\operatorname{sign}(\mu_i)
\left[|\mu_i|-\kappa\sigma_i-\delta\right]_+.
\]

Thus uncertain or negligible proposals produce an exact A2 fallback. For
two-logit classifiers, half of the accepted correction is subtracted from the
negative logit and half is added to the positive logit, preserving the logit
midpoint.

## 3. Differences from failed directions

1. No label is used by an expert on its held-out temporal environment.
2. The loss optimizes ranking improvement relative to A2, not an unconstrained
   pointwise score shift.
3. The hinge has a finite gain target, so there is no incentive for correction
   magnitude to grow indefinitely.
4. Cross-expert disagreement directly reduces the deployed correction.
5. The fallback is algebraically exact, rather than a soft gate that always
   perturbs A2.

## 4. Prospective screening protocol

- Fixed code version and one hyperparameter set for both datasets.
- Small-LI seed 42 and Large-LI seed 44.
- 500 epochs, no early stopping.
- Fixed validation target panel with seed 1729.
- Primary endpoint: test F1 at the best validation-F1 epoch.
- Required improvement over matched A2: at least `+0.005` on both datasets.
- A2 references: Small-LI `0.45507`, Large-LI `0.37143`.
- Passing thresholds: Small-LI `0.46007`, Large-LI `0.37643`.
- Raw-best test F1 is diagnostic only.

If and only if both pair thresholds pass, expand the unchanged commit to
Medium-HI and Large-HI, then to the remaining datasets.

## 5. Reproduction

Use `run/prospective_pair_queue.py` with:

```bash
PAIR_SPEC=run/cfrcr_pair_spec.json \
PAIR_GPU_ALLOWLIST=3,4 \
PAIR_OUT_BASE=/e/yky/FraudGT_pair_results \
PAIR_RUNTIME_BASE=/e/yky/FraudGT_pair_runtime \
python run/prospective_pair_queue.py
```

Audit with `run/prospective_pair_audit.py` against commit `fdbfffd3` and the
reference root
`/e/yky/FraudGT_pair_results/fixedevala2_fdbfffd3_pair500`.
