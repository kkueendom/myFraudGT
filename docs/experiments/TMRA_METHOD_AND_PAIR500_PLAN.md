# TMRA: Temporal Memory Rank Alignment

## 1. Problem addressed

Large-LI diagnostics show that about half of training microbatches contain no
fraud-positive target edge. EPRA correctly returns zero rank loss when either
class is absent, but this means many Large-LI updates receive no rare-class
ranking supervision. Increasing the loss weight cannot recover information
that is missing from the current batch.

TMRA extends EPRA with a training-only temporal ranking memory. It preserves
the complete A2 decoder and all EPRA safeguards while allowing a one-class
microbatch to form valid pairs with recent examples of the opposite class.

## 2. Method

Training edge IDs are chronological. The training range is split into three
contiguous environments. For environment `e`, TMRA maintains bounded FIFO
memories of recent normalized positive and negative A2 margins:

\[
M_e^+,\;M_e^-.
\]

Margins are centered and divided by a detached batch scale before insertion.
Every memory tensor is detached, so old batches never retain an autograd graph.
The memory capacity is 256 per class and environment; only the most recent
values are retained.

For the current batch, TMRA forms up to three kinds of rank pairs per temporal
environment:

1. current positives versus current negatives;
2. current positives versus recent negative memory;
3. recent positive memory versus current negatives.

If an environment memory has not been initialized, the corresponding global
class memory is used as a temporary fallback. The finite-margin loss is

\[
L_{mem}=\frac{1}{|\mathcal P|}
\sum_{(p,n)\in\mathcal P}
\tau\,\operatorname{softplus}
\left(\frac{\gamma-(m_p-m_n)}{\tau}\right).
\]

Only current-batch margins receive gradients. Memories are updated after the
loss is constructed, preventing a batch from serving as its own historical
anchor.

The full training-only objective is

\[
L=L_{A2}+\lambda_g L_{global-rank}
+\lambda_p L_{prototype}+\lambda_m L_{mem}.
\]

The auxiliary weights use the same 5-epoch warm-up and 25-epoch linear ramp as
EPRA. Inference discards the memories and all auxiliary losses, so the deployed
model is exactly the A2 decoder with no additional parameters or latency.

## 3. Main innovation

- It targets the observed rare-positive microbatch sparsity directly rather
  than adding another inference-time corrector.
- Temporal memories prevent the majority period from dominating rank pairs.
- Detached FIFO anchors provide cross-batch supervision without label leakage
  into evaluation or unbounded stale computation graphs.
- Missing-class batches become useful while retaining a finite ranking margin.
- The method changes representation learning but preserves A2 inference.

## 4. Prospective pair500 protocol

- Branch: `feature/temporal-memory-rank-alignment`.
- One fixed commit and one hyperparameter set for both datasets.
- Small-LI seed 42 and Large-LI seed 44, 500 epochs each.
- Validation-selected test F1 is the primary endpoint.
- Required deltas versus matched A2: at least `+0.005` on both datasets.
- Thresholds: Small-LI `0.46007`, Large-LI `0.37643`.
- Raw-best test F1 remains diagnostic only.
- Expansion order after passing: Medium-HI + Large-HI, then the remaining
  datasets without changing the method commit.

## 5. Reproduction

```bash
PAIR_SPEC=run/tmra_pair_spec.json \
PAIR_GPU_ALLOWLIST=1,5 \
PAIR_OUT_BASE=/e/yky/FraudGT_pair_results \
PAIR_RUNTIME_BASE=/e/yky/FraudGT_pair_runtime \
python run/prospective_pair_queue.py
```
