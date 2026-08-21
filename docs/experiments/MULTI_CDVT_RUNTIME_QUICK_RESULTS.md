# Multi-CDVT Runtime 快速验证结果

## Material Passport

- Experiment ID: `multi_runtime_quick_2c31bba3`
- Type: runtime benchmark
- Status: completed and audited
- Git branch: `experiment/multi-cdvt-runtime`
- Git commit: `2c31bba35601d3a28d51e3969df6944cf0da5fc7`
- Hardware: NVIDIA GeForce RTX 2080 Ti
- Software: PyTorch 2.5.1, CUDA 11.8
- Sampling protocol: `dynamic_random`
- Result root: `docs/experiments/results/multi_runtime_quick_2c31bba3`

## 1. 验证问题

快速判断当前 Multi-CDVT 相对同一 backbone 的 Multi-FraudGT 增加了多少
参数、峰值显存和端到端推理延迟，并据此决定是否值得运行三个 LI 数据集
的正式 256-batch runtime 实验。

## 2. 实验设计

| 项目 | 设置 |
|---|---|
| 数据集 | AML Small-LI |
| 对照模型 | `multi_account_only`，即 Multi-FraudGT |
| 完整模型 | `multi_cdvt` |
| Seed | 42 |
| Batch size | 2048 |
| 采样 | train/val/test 均为动态随机采样，`shuffle=True`，无独立 generator |
| 推理模式 | normal-only，端到端计时，包含动态采样、CPU 数据准备和 GPU forward |
| 预热 | 4 batches |
| 测量 | 3 个连续窗口，每个窗口 32 batches，共 96 batches/模型 |
| 执行方式 | 两个模型在同一张 GPU 上串行运行，避免并发干扰 |
| 权重 | 新初始化，不加载 checkpoint；算子规模和 runtime 不依赖训练后的权重值 |

## 3. 初步结果

| 模型 | 参数量 | 峰值显存 | 每 batch 时间，Mean +/- Std | 吞吐量 |
|---|---:|---:|---:|---:|
| Multi-FraudGT | 243,561 | 4.824 GiB | 0.5673 +/- 0.0072 s | 3,577.90 targets/s |
| Multi-CDVT | 362,139 | 4.841 GiB | 2.8130 +/- 0.0156 s | 721.65 targets/s |

| 对比项 | Multi-CDVT / Multi-FraudGT | 预设理想目标 | 预设最低标准 | 初步判断 |
|---|---:|---:|---:|---|
| 参数量 | 1.4869x | <= 1.5x | <= 2.0x | 达到理想目标 |
| 峰值显存 | 1.0034x | <= 1.2x | <= 1.5x | 达到理想目标 |
| 推理延迟 | 4.9583x | <= 2.0x | <= 5.0x | 仅边界通过 |
| 吞吐量 | 0.2017x | 越高越好 | 不单独设 Gate | 下降约 79.8% |

## 4. 初步结论

1. Multi-CDVT 的参数和显存开销可控。参数增加约 48.7%，峰值显存只增加
   约 0.34%，说明新增事件分支并未造成明显 GPU 显存压力。
2. 主要代价是端到端延迟。Small-LI 上每 batch 时间由 0.567 秒增加到
   2.813 秒，约为原模型的 4.958 倍。
3. 延迟只比 5 倍最低门槛低约 0.8%，因此应标记为边界结果，不能据此
   宣称 Multi-CDVT 高效。当前更准确的论文表述是“以有限参数和显存增量
   换取更高 F1，但事件上下文构建带来明显延迟”。
4. 三个 32-batch 窗口的波动较小，但本次只有一个数据集和一台 GPU，仍
   不能替代正式效率表。

## 5. 下一步正式实验

在相同 2080 Ti、batch size、动态随机采样和 normal-only 条件下，对
Small-LI、Medium-LI、Large-LI 分别运行 4-batch 预热和 256-batch 测量。
正式表格至少报告参数量、峰值显存、秒/batch、targets/s 和相对开销。
当 B0 checkpoint 可用时优先加载 Full/B0 的 Val-selected checkpoints；
若继续使用新初始化权重，必须保留“runtime 与权重值无关”的明确说明。

## 6. 证据路径

- 汇总：`docs/experiments/results/multi_runtime_quick_2c31bba3/runtime_summary.json`
- 人类可读表：`docs/experiments/results/multi_runtime_quick_2c31bba3/runtime_summary.md`
- 原始 benchmark：两个模型目录下的 `benchmark.json`
- 配置：`docs/experiments/results/multi_runtime_quick_2c31bba3/configs/`
- 完成标记：`docs/experiments/results/multi_runtime_quick_2c31bba3/queue_complete.json`
