# FraudGT 消融实验阶段性审计

更新时间：2026-09-14

## 1. 当前状态

- 正式任务：12 项
- 已完成并通过 manifest 审计：11 项
- 运行中：Large-HI B0（`multi_account_only`）
- 主指标：Val-selected Test F1
- 随机种子：42
- 训练轮数：500 epoch
- 采样协议：`dynamic_random`
- 已完成任务代码版本：`691c0c7f`
- Large-HI B0 当前低显存等价实现：`ba2ea2c6`
- Large-HI B0 当前正式配置：`AML-Large-HI-multi_account_only-seed42-retry10-offload-chunk65536.yaml`
- Large-HI B0 当前进程：PID `862141`，物理 GPU 1
- 低频监控：每 12 小时一次

Large-HI B0 的新版本只调整张量分配、残差缓冲区复用、checkpoint 激活的 CPU 临时存储位置和 edge FF 分块大小，不改变模型公式、采样协议、batch size 或指标计算。

## 2. Variant 含义

| 代号 | 实验名称 | 含义 |
|---|---|---|
| B0 | `multi_account_only` | 只保留账户图主干，不使用高阶事件视图 |
| B1 | `multi_causal_event_add` | 加入因果事件证据，但采用较简单的加法融合 |
| B2 | `multi_dual_view_no_relation` | 使用双视图融合，但关闭事件关系类型 |
| Full | `multi_cdvt` | 完整 Multi-CDVT 模型 |

## 3. 已完成结果

`Full - Variant` 表示完整模型相对于该消融模型的变化。正值表示被移除的模块有正贡献；负值表示消融模型反而更高。

| 数据集 | Variant | Variant F1 | Full F1 | Full - Variant |
|---|---|---:|---:|---:|
| Small-LI | B0 | 0.43191 | 0.44841 | +0.01651 |
| Small-LI | B1 | 0.46486 | 0.44841 | -0.01645 |
| Small-LI | B2 | 0.46185 | 0.44841 | -0.01343 |
| Small-HI | B0 | 0.76932 | 0.78393 | +0.01461 |
| Medium-LI | B0 | 0.43923 | 0.49882 | +0.05959 |
| Medium-LI | B1 | 0.47031 | 0.49882 | +0.02851 |
| Medium-LI | B2 | 0.47150 | 0.49882 | +0.02732 |
| Medium-HI | B0 | 0.78195 | 0.78731 | +0.00535 |
| Large-LI | B0 | 0.38871 | 0.48824 | +0.09952 |
| Large-LI | B1 | 0.49080 | 0.48824 | -0.00256 |
| Large-LI | B2 | 0.35366 | 0.48824 | +0.13458 |
| Large-HI | B0 | 运行中 | 0.79741 | 待计算 |

## 4. 阶段性均值

| Variant | 已完成数据集数 | Variant 平均 F1 | 平均 Full - Variant |
|---|---:|---:|---:|
| B0 | 5/6 | 0.56223 | +0.03912 |
| B1 | 3/3 | 0.47532 | +0.00317 |
| B2 | 3/3 | 0.42900 | +0.04949 |

B0 的均值尚未包含 Large-HI，不能作为最终均值。B1 和 B2 只在三个 LI 数据集上设计并运行。

## 5. 当前结论

1. B0 在已完成的五个数据集上都低于 Full，说明高阶事件视图整体有稳定正贡献。
2. B1 的平均差值仅为 `+0.00317`，并且 Small-LI、Large-LI 上 B1 高于 Full，说明完整交互融合相对简单加法融合的优势不稳定。
3. B2 在 Medium-LI 和 Large-LI 上明显低于 Full，但在 Small-LI 上高于 Full，说明关系类型在中大型数据集更有价值，在小数据集上可能引入噪声。
4. 当前结果可支持“事件视图总体有效”和“关系类型对较大图更重要”，但不能宣称所有新增子模块在所有数据集上都单调提升。
5. 目前均为单 seed 正式结果，不应在论文中虚构多 seed 均值或标准差。

## 6. 剩余工作

1. 等待 Large-HI B0 完成 500 epoch 并生成 `manifest.json`。
2. 核验 `sampling_protocol=dynamic_random`、seed、epoch、commit 和 Val-selected Test F1。
3. 运行正式汇总脚本，生成 `ablation_summary.json` 和 `ablation_summary.md`。
4. 补充 Large-HI B0 行，并重新计算 B0 的六数据集平均值。

## 7. Large-HI B0 加速与复现记录

原 `edge_ff_chunk_size=4096` 正式运行约需 27 至 30 分钟完成一个训练 epoch，预计 500 epochs 需要约 10 天。为避免单纯重启后仍然过慢，先在空闲 GPU 上进行了三个 4-epoch 等价配置基准；三者均保持动态随机采样、seed 42、batch size 2048、每轮 256 次迭代、每 4 epochs 评估一次以及关闭早停。

| edge FF chunk size | 4 epochs 总耗时 | 训练耗时 | 推理耗时 | GPU 峰值显存 |
|---:|---:|---:|---:|---:|
| 16384 | 4164.7 秒 | 3676.7 秒 | 488.0 秒 | 22.137 GB |
| 32768 | 3662.5 秒 | 3144.8 秒 | 517.7 秒 | 22.158 GB |
| 65536 | 3289.0 秒 | 2780.1 秒 | 508.9 秒 | 22.200 GB |

因此选择 `edge_ff_chunk_size=65536` 作为正式 retry10 配置。其训练部分相对 4096 配置约快 2.5 倍，按短基准估计完整任务约需 4.8 天。慢速 retry10 前的输出已归档为：

```text
/e/yky/FraudGT_cdvt_results/multi_ablation_seed42_accel_v2_691c0c7/Large-HI_multi_account_only_seed42.slow_chunk4096_epoch4_20260914T083544Z
```

当前正式输出目录保持不变：

```text
/e/yky/FraudGT_cdvt_results/multi_ablation_seed42_accel_v2_691c0c7/Large-HI_multi_account_only_seed42
```

截至 2026-09-14 12:29（服务器时间），正式 retry10 已运行至 epoch 16，最近一个训练 epoch 约 725 秒，未发现 OOM。该中间状态仅用于运行健康检查，不能作为论文正式结果。
