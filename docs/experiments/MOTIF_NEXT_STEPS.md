# Motif-Aware Graph Transformer — host run guide

New research direction (encoder architecture innovation). The decoder line
(v1/v2/M1/v3) is deprioritized — under the honest val-selected F1 metric it gave
~0 gain over the FraudGT baseline. Branch: `feature/motif-aware-gt`.

## ⚡ FAST go/no-go first (do this before the full sweep)

One command runs motif-ON vs motif-OFF on the two smallest datasets at a short
budget and prints the val-selected test F1 delta. Both arms are identical except
`gt.motif_bias`, so the delta is purely the mechanism.

```bash
git pull                        # branch feature/motif-aware-gt
PY=/d/miniconda3/envs/fraudGT/bin/python EPOCHS=80 bash run/motif_quickcheck.sh
# (set GPU=<n> / DATASETS="Small-HI Small-LI" / SEED=42 as needed)
```

It prints a table ending with `mean Δ Test@Val (motif - baseline)`. **Decision:**
- clearly **positive Δ** on both small datasets → the mechanism works; proceed to
  the full matched sweep below (and add more seeds / all 6 datasets).
- **≈0 or negative Δ** → motif-biased attention (this structural version) is not
  enough; report back before investing more. Likely next: amount-weighted / longer
  motifs, or the motif *node encoder*, or a different architectural angle.

Also confirm the mechanism is actually active (not zero): after a run, the learned
`motif_proj.weight` norm should be > 0 (it is zero-initialised).

---

## Full run (only if the fast check is positive)

## What this is

A **laundering-motif-biased attention** added to FraudGT's `SparseNodeTransformer`.
Money laundering is defined by two structures the FraudGT paper names as the core
difficulty (and that GTs provably struggle with): **scatter-gather / smurfing**
(fan-in/out) and **round-tripping** (short cycles / reciprocity). We give the
attention an explicit, learned per-head bias driven by a cheap, parameter-free
per-edge motif descriptor, so attention is amplified along edges that participate
in laundering structure.

- Mechanism: `edge_scores += motif_proj(LayerNorm(motif_e))` per head, right after
  the existing edge-attr bias / temporal bias — `fraudGT/layer/gt_layer.py`
  (`motif_bias_enabled`, ~L1797). Starts at zero (identical to baseline attention),
  learns from there.
- Descriptor: `fraudGT/transform/motif_stats.py::compute_motif_edge_features`
  (src/dst in-out degrees, scatter/gather ratios, reciprocity). Computed on the
  sampled 2-hop subgraph via degree scatter + a reciprocity lookup — negligible
  latency, preserving FraudGT's efficiency claim.
- Config knob: `gt.motif_bias` (`fraudGT/config/gt_config.py`, default False).

Scope note: this version uses **structural** motifs only (degrees + reciprocity).
Amount-weighted motifs, longer cycles (need a >2-hop sampler), and a separate
motif *node encoder* are documented follow-ups.

## Configs

`configs/AML-*/AML-*-SparseNodeGT+ports+Ego+Motif.yaml` — the FraudGT PE-baseline
(`SparseNodeGT+ports+Ego`, `edge_decoding: dot`, 500 epochs, no early stop) **plus**
`gt.motif_bias: True`. Data dir set to `/e/yyk/data/archive`, wandb off.

**The fair baseline is the SAME config with `gt.motif_bias False`** (a CLI override),
so the two arms are identical except the mechanism.

## Run (matched pair, 3 seeds, all 6 datasets)

```bash
git pull

# 1) smoke test (motif on), few epochs — expect no NaN, loss falls, small overhead:
python -m fraudGT.main \
  --cfg configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+Motif.yaml \
  --repeat 1 --gpu 0 seed 42 optim.max_epoch 4 train.iter_per_epoch 16

# 2) full matched pair (parallelise across GPUs by adapting your M1 queue):
for ds in Small-HI Small-LI Medium-HI Medium-LI Large-HI Large-LI; do
  for seed in 42 43 44; do
    cfg=configs/AML-$ds/AML-$ds-SparseNodeGT+ports+Ego+Motif.yaml
    # Motif arm:
    python -m fraudGT.main --cfg $cfg --repeat 1 --gpu 0 seed $seed \
      out_dir /e/yyk/FraudGT_multi6/motif/AML-$ds
    # Baseline arm (identical config, mechanism off):
    python -m fraudGT.main --cfg $cfg --repeat 1 --gpu 0 seed $seed \
      gt.motif_bias False out_dir /e/yyk/FraudGT_multi6/motif_baseline/AML-$ds
  done
done
```

## Verify / evaluate

1. **Mechanism is active (not collapsed like the decoder gate):** after training,
   confirm the learned `motif_proj.weight` is non-zero (e.g. print its norm), and
   ideally that removing it (`gt.motif_bias False`) changes predictions.
2. **Metric:** report **val-selected test F1** (`test_at_val`), 3-seed mean ± std,
   all 6 datasets incl. Large-LI. Compare the motif arm vs the matched baseline arm.
   (Reuse the val-select audit logic from `run/evidence_gate_v2_audit.py`, pointing
   `--roots` at the two out dirs.)
3. **Success =** motif beats the matched FraudGT baseline on the honest metric,
   especially on cycle-heavy / LI / Large cases. Report per-dataset deltas.

## Follow-ups (for the paper, after the core result)
- **Generic-PE baseline** (`Hetero_RWSE` / LapPE, plumbing already in the repo): show
  the *fraud-specific* motif bias beats a *generic* structural encoding — the key
  novelty defense against "this is just feature engineering".
- **Ablate motif families** (scatter-gather vs reciprocity/cycles) and add
  amount-weighted / longer-cycle motifs (larger-receptive-field sampler).
- **Efficiency table**: latency motif-on vs motif-off (expected negligible).
