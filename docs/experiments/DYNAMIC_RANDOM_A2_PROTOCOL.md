# Dynamic-Random A2 Experiment Protocol

## Sampling contract

All subsequent baseline, full-model, and ablation runs use the sampling behavior
of the public FraudGT implementation at commit `65706ed`:

- train, validation, and test loaders sample dynamically;
- `LinkNeighborLoader` receives `shuffle=True` for every split;
- `val.fixed_target_panel=False` is explicit in every experiment spec;
- validation and test target edges are not fixed;
- no independent evaluation generator is created;
- sampler RNG state is not restored before evaluation;
- dataset-specific `train.batch_size` and `val.iter_per_epoch` remain unchanged.

The Fixed-panel A2 runs are historical diagnostics only. They must not appear in
the main result table and must not be used as a screening baseline.

## Baseline and comparison

The only baseline is `run/dynamic_random_a2_baseline.json`, containing the six
initial A2 results. Val-selected Test F1 is compared only with A2 Val-selected
Test F1; Raw-best Test F1 is compared only with A2 Raw-best Test F1.

For either metric:

```text
delta_f1 = candidate_f1 - initial_a2_f1
```

An absolute delta below `0.005` is marked `sampling_noise_range`. Formal claims
prioritize Val-selected Test F1. Raw-best is supplementary.

## Screening and reporting

Single-seed runs may screen candidates. A candidate advances to multi-seed runs
only after it shows useful evidence under this same protocol. Reports include
per-dataset deltas, mean delta, wins, failures, and the sampling-risk label.

Every completed experiment manifest records dataset, model/variant, seed, Git
commit, config, checkpoint, Val-selected Test F1, Raw-best Test F1, both deltas,
and `sampling_protocol=dynamic_random`.
