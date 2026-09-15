# Large-HI B0 Memory-Safe Reproduction

This record reproduces the remaining Multi-CDVT ablation task:

- dataset: `AML Large-HI`
- variant: `multi_account_only` (B0)
- implementation commit: `ba2ea2c6c5dd921a7bc5b30a8d7490fc7e3541fb`
- sampling protocol: `dynamic_random`
- seed: `42`
- primary metric: Val-selected Test F1

The committed configuration is:

```text
configs/CDVT/ablation/AML-Large-HI-multi_account_only-seed42-retry10-offload-chunk65536.yaml
```

Its SHA-256 is:

```text
f9fca9c43493a67fdea2576eb9b2560e60cba16b5e1ca5a934c219b1341d1dfa
```

## Formal protocol

| Setting | Value |
|---|---|
| train/val/test sampling | dynamic random |
| `train.batch_size` | 2048 |
| train iterations per epoch | 256 |
| val/test iterations per evaluation | 256 |
| evaluation period | 4 epochs |
| maximum epochs | 500 |
| early stopping | disabled |
| fixed validation panel | disabled |
| edge FF chunk size | 65536 |
| edge FF checkpoint | enabled |
| checkpoint activation CPU offload | enabled |

The chunking and activation-storage changes preserve the model equations. They
reduce peak CUDA memory and make the account-only Large-HI run practical on the
2080 experiment host.

## Launch command

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True \
python run/cdvt_phase1_screen.py \
  --config configs/CDVT/ablation/AML-Large-HI-multi_account_only-seed42-retry10-offload-chunk65536.yaml \
  --device cuda:0 \
  --variant account_only \
  --experiment-label multi_account_only \
  --lambda-cons 0.0 \
  --output-dir /e/yky/FraudGT_cdvt_results/multi_ablation_seed42_accel_v2_691c0c7/Large-HI_multi_account_only_seed42 \
  --max-epochs 500 \
  --disable-early-stop \
  --phase CDVT_multi_ablation
```

## Acceptance checks

Use only the completed `manifest.json`. A valid formal result must contain:

```text
epochs_completed=500
sampling_protocol=dynamic_random
seed=42
git_commit=ba2ea2c6c5dd921a7bc5b30a8d7490fc7e3541fb
edge_ff_chunk_size=65536
edge_ff_checkpoint=true
edge_ff_offload=true
early_stopping_enabled=false
```

Do not report intermediate log F1 values as the formal result.
