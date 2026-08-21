# Multi-CDVT Runtime 快速验证结果

## Material Passport

- Experiment ID: `multi_runtime_quick_2c31bba3`
- Type: runtime benchmark
- Status: completed and audited
- Git branch: `experiment/multi-cdvt-runtime`
- Small-LI code commit: `2c31bba35601d3a28d51e3969df6944cf0da5fc7`
- Medium/Large-LI code commit: `848df40747e6912c6ed352c8f2a937ec0f286872`
- Hardware: NVIDIA GeForce RTX 2080 Ti
- Software: PyTorch 2.5.1, CUDA 11.8
- Sampling protocol: `dynamic_random`
- Result roots: `docs/experiments/results/multi_runtime_quick_*`

## 1. 验证问题

快速判断当前 Multi-CDVT 相对同一 backbone 的 Multi-FraudGT 增加了多少
参数、峰值显存和端到端推理延迟，并观察计算开销是否随数据规模变化。

## 2. 实验设计

| 项目 | 设置 |
|---|---|
| 数据集 | AML Small-LI、Medium-LI、Large-LI |
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

| 数据集 | 模型 | 参数量 | 峰值显存 | 每 batch 时间，Mean +/- Std | 吞吐量 |
|---|---|---:|---:|---:|---:|
| Small-LI | Multi-FraudGT | 243,561 | 4.824 GiB | 0.5673 +/- 0.0072 s | 3,577.90 targets/s |
| Small-LI | Multi-CDVT | 362,139 | 4.841 GiB | 2.8130 +/- 0.0156 s | 721.65 targets/s |
| Medium-LI | Multi-FraudGT | 243,561 | 7.153 GiB | 1.4343 +/- 0.0051 s | 1,313.60 targets/s |
| Medium-LI | Multi-CDVT | 362,139 | 7.220 GiB | 3.4516 +/- 0.0602 s | 546.55 targets/s |
| Large-LI | Multi-FraudGT | 243,561 | 11.416 GiB | 3.8387 +/- 0.5172 s | 149.04 targets/s |
| Large-LI | Multi-CDVT | 362,139 | 11.514 GiB | 3.9849 +/- 0.0929 s | 141.95 targets/s |

| 数据集 | 参数比 | 显存比 | 延迟比 | 吞吐量比 | 延迟判断 |
|---|---:|---:|---:|---:|---|
| Small-LI | 1.4869x | 1.0034x | 4.9583x | 0.2017x | 仅边界通过最低标准 |
| Medium-LI | 1.4869x | 1.0093x | 2.4064x | 0.4161x | 通过最低标准，未达理想目标 |
| Large-LI | 1.4869x | 1.0086x | 1.0381x | 0.9524x | 达到理想目标 |
| 三数据集宏平均 | 1.4869x | 1.0071x | 2.8009x | 0.5234x | 仅作补充，不替代逐数据集判断 |

三个数据集共 288 个测量 batch 的总时间比为 1.7549x。该 pooled 数值受
Large-LI 较长的原模型耗时影响，因此论文应优先报告上表中的逐数据集比值。

## 4. 初步结论

1. Multi-CDVT 的参数和显存开销可控。参数固定增加约 48.7%，三个数据集
   的峰值显存仅增加 0.34%-0.93%。
2. 相对延迟随数据规模增大而明显降低，从 Small-LI 的 4.958x 降到
   Medium-LI 的 2.406x，再降到 Large-LI 的 1.038x。合理解释是大图中
   原模型自身的邻居采样和图计算已经占据主要时间，事件分支的边际占比下降。
3. 三个数据集均满足 <=5x 的最低标准，但只有 Large-LI 达到 <=2x 的理想
   目标。Small-LI 仍是边界结果，因此不能笼统宣称模型高效。
4. Large-LI 原模型的三个窗口波动较大，标准差为 0.5172 s/batch；正式
   结果需要扩大到 256 batches，以降低动态采样造成的不确定性。
5. 当前准确表述仍是“以有限参数和显存增量换取预测提升，但在 Small 和
   Medium 规模上存在明显的事件上下文处理延迟”。

## 5. 下一步正式实验

当前三个 LI 数据集都完成了 96-batch 快测。论文正式效率表仍应把每个模型
扩大到 256 batches，并保持相同 2080 Ti、batch size、动态随机采样和
normal-only 条件。正式表格至少报告参数量、峰值显存、秒/batch、targets/s
和相对开销。当 B0 checkpoint 可用时优先加载 Full/B0 的 Val-selected
checkpoints；若继续使用新初始化权重，必须保留“runtime 与权重值无关”的
明确说明。

## 6. 证据路径

- Small-LI：`docs/experiments/results/multi_runtime_quick_2c31bba3/`
- Medium-LI：`docs/experiments/results/multi_runtime_quick_Medium-LI_848df407/`
- Large-LI：`docs/experiments/results/multi_runtime_quick_Large-LI_848df407/`
- 原始 benchmark：两个模型目录下的 `benchmark.json`
- 配置：各结果目录下的 `configs/`
- 完成标记：各结果目录下的 `queue_complete.json`
