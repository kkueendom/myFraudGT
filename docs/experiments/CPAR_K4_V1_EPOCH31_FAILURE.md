# CPAR-K4 V1 Epoch 31 失败结论

## 结论

CPAR-K4 V1（commit `1ee2a2d0`）在同一条 500-epoch scheduler 轨迹的 epoch 31
检查点已出现一致性负信号，停止作为候选方法。该结果不能用继续训练、raw-best
或重新选择 checkpoint 解释为成功。

| 数据集 | matched seed | A2 val-selected Test F1 | V1 | delta | raw-best delta |
|---|---:|---:|---:|---:|---:|
| Small-LI | 42 | 0.39051 | 0.33452 | **-0.05599** | -0.04377 |
| Large-LI | 44 | 0.10929 | 0.08284 | **-0.02645** | -0.03909 |

这里的 A2 和 V1 都只读取各自 500-epoch 轨迹截至 epoch 31 的记录。正式论文指标
仍是最佳 validation F1 epoch 对应的 Test F1；raw-best 只作诊断。

## 关键机制证据

Large-LI 的低频训练诊断为：

```text
best_action: a0=0.744, a0.5=0.013, a1=0.018, a1.5=0.225
normalized_gap: mean=0.00378, p90=0.01907
effective=0.225
route_aux=0.00734
confidence=0.9958
dose_mean=1.4961
```

验证集进一步显示：

```text
argmax a1.5=1.000
mean_q approximately 0.005/0.000/0.000/0.995
dose_mean approximately 1.488
```

这说明最终路由不是在区分 `a0` 与 `a1.5`，而是几乎对所有样本使用最大剂量。

## 根因

V1 对每个样本使用 oracle argmin 动作，但又把辅助权重设为：

\[
w_i=\operatorname{clip}(\tilde g_i/0.05,0,1).
\]

大量 `a0` 样本的最佳与次佳损失几乎相同，数值 gap 为 0 或接近 0，因此它们
虽然出现在 `best_action` 统计中，却没有形成有效的反向监督。只有 gap 较强的
`a1.5` 子集持续更新 router。辅助损失随后快速下降到 `0.00734`，但这是单一
动作监督被拟合后的假收敛，不代表路由判断正确。

因此，V1 的失败链条是：

```text
弱 gap 样本 -> 权重接近 0 -> a0/neutral 证据退出损失
强 a1.5 子集 -> 独占有效监督 -> router 全局预测 a1.5
高 confidence -> dose 接近 1.5 -> 两个困难数据集同时退化
```

## V2 修复与可证伪条件

1. `normalized_gap < 0.005` 的样本不再使用带噪声的 argmin，目标固定为 action
   index 2（dose 1，A2 neutral）。
2. 只有 `normalized_gap >= 0.005` 的 strong 样本使用 oracle action。
3. 类别 sample weight 先在最终 target 动作组内保留，再把每个出现的动作组归一到
   相等总监督质量，避免任一 strong 子集独占梯度。
4. neutral argmax 在前向精确使用 dose 1；straight-through soft dose 保留任务梯度。
5. 记录 strong coverage、abstention、strong action、最终 target 组、监督质量及
   dose 上下边界。
6. V2 从 epoch 0 新跑，并使用同时包含 `v2` 与短 commit 的结果和 marker 路径。

若 V2 再次出现任一情况，应立即判定机制失败：

- 某个出现的最终 target 组监督质量明显偏离其他组；
- target 中 neutral 占多数，但验证 route 仍长期全局卡在 dose 1.5；
- strong coverage 很低且模型仍高 confidence 地偏离 A2；
- Small-LI 或 Large-LI 在预注册中途节点持续明显低于 matched A2。

## 证据位置

- V1 结果：`results/cpar_k4_formal500/`
- V1 run commit：`1ee2a2d0`
- matched A2：`/e/yky/FraudGT_dmprd_quickcheck/results/dmprd_formal500/`
- V1 audit：`run/cpar_k4_formal_audit.py --epoch-limit 31`
