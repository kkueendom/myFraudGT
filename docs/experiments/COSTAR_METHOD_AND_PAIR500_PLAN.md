# COSTAR: Orthogonal Threshold-Robust Prototype Adapter

## Status

- Decoder: `costar`
- Branch: `feature/costar-dynamic-random`
- Scope: matched Small-LI seed 42 and Large-LI seed 44, 500 epochs
- Primary metric: validation-selected Test F1
- Training labels: used only by the original A2 loss and the COSTAR adapter
  objective
- Validation/test labels: never used by COSTAR gradients, EMA, orthogonal
  center, or routing

## Motivation

CPAR-K4 and ACDR Help alternate between gains and losses across datasets and
training checkpoints. Their local route also multiplies the same prototype
residual as A2's global beta, so global scale and sample allocation are not
identifiable. COSTAR addresses both issues without changing the A2 optimizer
target.

## Implemented method

For binary A2 logits, define the anchor margin and effective prototype margin:

\[
m_i^{A2}=m_i^{base}+e_i,
\qquad
e_i=\beta r_i(d_{i,1}-d_{i,0}).
\]

The no-dropout router consumes detached, label-free inference features:

\[
\xi_i=[h_i,m_i^{base},m_i^{A2},e_i,s_i^+,s_i^-,
m_i^{proto},r_i,\operatorname{agree}_i].
\]

Current and EMA routers produce bounded values:

\[
v_i=\tanh f_\phi(\xi_i),\qquad
\bar v_i=\tanh f_{\bar\phi}(\xi_i).
\]

Temporal disagreement produces a confidence and consensus value:

\[
c_i=\exp(-|v_i-\bar v_i|/\tau_c),\qquad
q_i=c_i(v_i+\bar v_i)/2.
\]

The train-only EMA center is

\[
b=\frac{\mathbb E_{train}[q_i e_i^2]}
        {\mathbb E_{train}[e_i^2]+\epsilon}.
\]

The final correction and prediction are

\[
\Delta_i=\frac12(q_i-b)e_i,
\qquad
m_i=m_i^{A2}+\Delta_i.
\]

With an exact train-distribution center,

\[
\mathbb E_{train}[\Delta_i e_i]=0,
\]

so the adapter cannot reproduce a uniform change to A2's global prototype
coefficient. There is no second learned global residual scale.

The router output layer is zero initialized. At initialization, or when the
adapter is disabled, `q=b=Delta=0`, so COSTAR logits equal A2 logits exactly.

## A2 optimization anchor

The head stashes both final COSTAR logits and pre-adapter A2 logits. The custom
training loop computes:

\[
L=L_{A2}(z^{A2},y)+L_{adapter}(z^{COSTAR},
\operatorname{stopgrad}(z^{A2}),y).
\]

- `L_A2` updates the original encoder, base decoder, prototype head, and beta.
- `L_adapter` updates only the current COSTAR router.
- All A2 features and residuals are detached in the adapter path.
- A2 and router gradients are clipped separately, so the router cannot change
  the A2 update through a shared gradient norm.
- The EMA router is updated after optimizer steps and is stored in checkpoints.
- Prototype banks are updated from the current training batch only after its
  prediction has been formed.

## Threshold-transfer objective

### Honest implementation boundary

This version does **not** claim strict historical cross-fold training. The
current loader exposes sampled training batches but no stable target-edge ID or
epoch-complete prediction cache. Fabricating fold names would not make the
procedure cross-fold.

The implemented approximation deterministically splits each training batch
within each class into alternating calibration and evaluation halves. A hard-F1
threshold is selected from detached COSTAR calibration scores using training
labels only. The adapter is optimized on the other half at three thresholds:

\[
\{t-0.1\sigma_{cal},\ t,\ t+0.1\sigma_{cal}\}.
\]

If a batch does not contain at least two examples from both classes, the code
falls back to the full batch and records `batch_split_valid=0`. This is a
train-batch threshold-transfer approximation, not an independent validation
estimate.

For an evaluation-half threshold `t`,

\[
\widetilde F1(t)=
\frac{2\sum_i y_i\sigma((m_i-t)/T)+\epsilon}
     {\sum_i y_i+\sum_i\sigma((m_i-t)/T)+\epsilon}.
\]

The platform loss is the mean of the worst half of the three threshold losses.
The adapter also receives pairwise positive-negative ranking loss and hinge
safeguards that penalize worse soft-F1 or ranking loss than the detached A2
anchor on the same evaluation half.

\[
L_{adapter}=w_a\left[
L_{plateau}+w_rL_{rank}+w_s(L_{F1-safe}+L_{rank-safe})
+w_oL_{orth}+w_tL_{time}\right].
\]

## Inference contract

Inference uses only sample features, current/EMA router parameters, and the
frozen train-EMA center. It does not use:

- labels;
- validation/test statistics;
- batch means, extrema, ordering, or batch size;
- current-batch prototype updates.

Changing inference batch partition or order must preserve logits within
`1e-6`.

## Diagnostics

Low-frequency logs record:

- orthogonal cosine error `|<Delta,e>| / (||Delta|| ||e||)`;
- three-threshold platform-width proxy;
- current/EMA consistency;
- fallback ratio;
- correction RMS;
- train-center value and update count;
- EMA update count.

The training objective additionally keeps worst soft-F1, safe-regret terms,
rank loss, selected training threshold, perturbation size, and whether the
batch stratified split was valid in `_costar_diag`.

## Pair-500 protocol

All splits use FraudGT's original dynamic-random sampling with
`LinkNeighborLoader(shuffle=True)`. No target panel, independent evaluation
generator, or sampler RNG restoration is used. The comparison baseline is the
initial A2 table in `run/dynamic_random_a2_baseline.json`.

| Dataset | Seed | Matched A2 selected Test F1 |
|---|---:|---:|
| Small-LI | 42 | 0.46247 |
| Large-LI | 44 | 0.30108 |

Both jobs start with `optim.max_epoch=500` and the same cosine-with-warmup
scheduler as A2. Intermediate epochs are reads from that trajectory, never
separate short-scheduler runs.

Formal release requires both selected Test F1 values to exceed matched A2 by at
least `0.005`. Raw-best Test F1 is diagnostic only.

`run/costar_formal_queue.py` refuses a dirty worktree, excludes GPU0, embeds the
short commit in every result name, and writes a full-revision lock under the Git
worktree metadata on first launch. The queue is provided but is not launched by
this implementation task.

## Known risks

1. The batch-stratified objective is not strict cross-fold and depends on the
   sampled training batch composition.
2. Rare positive classes may make some batches fall back to the full-batch
   approximation.
3. The EMA orthogonal center is exact only in expectation; distribution drift
   can create nonzero instantaneous orthogonal error.
4. Zero initialization protects A2 initially but does not guarantee that the
   learned adapter improves unseen data.
5. The soft-F1 temperature and threshold perturbation remain method constants
   that require later sensitivity analysis if COSTAR passes the pair screen.
