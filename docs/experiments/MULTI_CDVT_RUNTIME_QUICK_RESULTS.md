# Multi-CDVT Runtime 快速验证结果

## Material Passport

- Experiment ID: `multi_runtime_quick_six_dataset`
- Type: runtime benchmark
- Status: completed and audited
- Git branch: `experiment/multi-cdvt-runtime`
- Small-LI code commit: `2c31bba35601d3a28d51e3969df6944cf0da5fc7`
- Medium/Large-LI code commit: `848df40747e6912c6ed352c8f2a937ec0f286872`
- Small/Medium/Large-HI code commit: `17e6f06517fbfe8e480cfbdca078ee2bdb8cf9e1`
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
| 数据集 | AML Small/Medium/Large-LI 和 Small/Medium/Large-HI |
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
| Small-HI | Multi-FraudGT | 243,561 | 4.740 GiB | 0.5749 +/- 0.0020 s | 3,530.72 targets/s |
| Small-HI | Multi-CDVT | 362,139 | 4.734 GiB | 2.6939 +/- 0.0250 s | 753.43 targets/s |
| Medium-HI | Multi-FraudGT | 243,561 | 7.184 GiB | 1.3584 +/- 0.0024 s | 1,388.84 targets/s |
| Medium-HI | Multi-CDVT | 362,139 | 7.214 GiB | 3.6255 +/- 0.2662 s | 522.85 targets/s |
| Large-HI | Multi-FraudGT | 243,561 | 11.557 GiB | 3.3555 +/- 0.0269 s | 168.70 targets/s |
| Large-HI | Multi-CDVT | 362,139 | 11.584 GiB | 4.0239 +/- 0.0418 s | 140.86 targets/s |

| 数据集 | 参数比 | 显存比 | 延迟比 | 吞吐量比 | 延迟判断 |
|---|---:|---:|---:|---:|---|
| Small-LI | 1.4869x | 1.0034x | 4.9583x | 0.2017x | 仅边界通过最低标准 |
| Small-HI | 1.4869x | 0.9987x | 4.6858x | 0.2134x | 通过最低标准，接近边界 |
| Medium-LI | 1.4869x | 1.0093x | 2.4064x | 0.4161x | 通过最低标准，未达理想目标 |
| Medium-HI | 1.4869x | 1.0042x | 2.6690x | 0.3765x | 通过最低标准，未达理想目标 |
| Large-LI | 1.4869x | 1.0086x | 1.0381x | 0.9524x | 达到理想目标 |
| Large-HI | 1.4869x | 1.0024x | 1.1992x | 0.8350x | 达到理想目标 |
| 六数据集宏平均 | 1.4869x | 1.0044x | 2.8261x | 0.4992x | 仅作补充，不替代逐数据集判断 |

六个数据集共 576 个测量 batch 的总时间比为 1.8504x。该 pooled 数值受
两个 Large 数据集较长的原模型耗时影响，因此论文应优先报告上表中的逐
数据集比值。LI 和 HI 的宏平均延迟比分别为 2.8009x 和 2.8513x。

## 4. 初步结论

1. Multi-CDVT 的参数和显存开销可控。参数固定增加约 48.7%，六个数据集
   的显存比位于 0.9987x-1.0093x，几乎没有额外峰值显存压力。
2. 相对延迟主要由数据规模决定，而不是 LI/HI 强度决定。Small 的延迟比为
   4.686x-4.958x，Medium 为 2.406x-2.669x，Large 为 1.038x-1.199x。
3. 合理解释是大图中原模型自身的邻居采样和图计算已占据主要时间，事件
   分支的边际占比随规模增大而下降。
4. 六个数据集均满足 <=5x 的最低标准，两个 Large 数据集达到 <=2x 的理想
   目标。两个 Small 数据集仍接近边界，因此不能笼统宣称模型高效。
5. Large-LI 原模型和 Medium-HI Multi-CDVT 的窗口波动相对较大，正式
   结果需要扩大到 256 batches，以降低动态采样造成的不确定性。
6. 当前准确表述是“以有限参数和显存增量换取预测提升；Large 规模的边际
   延迟较小，但 Small/Medium 规模仍有明显的事件上下文处理开销”。

## 5. 下一步正式实验

当前六个数据集都完成了 96-batch 快测。论文正式效率表最低应把三个 LI
代表数据集扩大到 256 batches；理想情况下六个数据集全部扩展。正式实验
应保持相同 2080 Ti、batch size、动态随机采样和 normal-only 条件，并
报告参数量、峰值显存、秒/batch、targets/s 和相对开销。当 B0 checkpoint
可用时优先加载 Full/B0 的 Val-selected checkpoints；若继续使用新初始化
权重，必须保留“runtime 与权重值无关”的明确说明。

## 6. 证据路径

- Small-LI：`docs/experiments/results/multi_runtime_quick_2c31bba3/`
- Medium-LI：`docs/experiments/results/multi_runtime_quick_Medium-LI_848df407/`
- Large-LI：`docs/experiments/results/multi_runtime_quick_Large-LI_848df407/`
- Small-HI：`docs/experiments/results/multi_runtime_quick_Small-HI_17e6f065/`
- Medium-HI：`docs/experiments/results/multi_runtime_quick_Medium-HI_17e6f065/`
- Large-HI：`docs/experiments/results/multi_runtime_quick_Large-HI_17e6f065/`
- 原始 benchmark：两个模型目录下的 `benchmark.json`
- 配置：各结果目录下的 `configs/`
- 完成标记：各结果目录下的 `queue_complete.json`
