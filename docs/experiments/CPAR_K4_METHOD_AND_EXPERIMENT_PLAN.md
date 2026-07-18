# CPAR-K4 V2 方法与第一阶段实验计划

## 1. 研究问题

A2 使用四槽类别原型生成有界残差，并以一个全局系数控制残差总强度：

\[
z_i^{A2}=z_i^{base}+\beta r_i d_i,
\]

其中，\(r_i\) 表示正、负类别原型均已建立，\(d_i\) 是原型残差，
\(\beta\) 是全局可学习系数。

P0 的失败表明，累计 support、基础熵和原型绝对置信度组成的固定乘法 gate
会在长训练中退化：熵接近 0，support 接近 1，gate 接近下限，最终只会统一缩小
A2 残差。CAMPR V2 虽改为学习路由，但它比较 base 与 A2，而实际决策发生在 A2
附近；其批均值归一化还会让同一样本的预测依赖同批样本。

CPAR-K4 要回答更直接的问题：

> 对当前样本，A2 原型残差应该关闭、减半、保持还是增强？

V1 在 epoch 31 已被证伪：Small-LI 与 Large-LI 的 val-selected Test F1
分别低于 matched A2 `-0.05599` 和 `-0.02645`。Large 的训练诊断同时出现
`best_action a0=0.744, a1.5=0.225`，但只有 `0.225` 样本获得有效监督，最终
router 却以 `0.9958` confidence 把平均 dose 推到 `1.4961`。V2 因此不再让
弱 gap 样本退出损失，而是把它们明确监督为 A2 neutral action。

## 2. 方法：Counterfactual Prototype Action Router with K=4

### 2.1 A2 中心的反事实动作

动作空间固定为：

\[
\mathcal A=\{0,0.5,1.0,1.5\}.
\]

训练时，对每个样本计算四个不参与反向传播的反事实损失：

\[
L_{ik}=\ell\left(
z_i^{base}+a_k\beta r_i d_i, y_i
\right),\quad a_k\in\mathcal A.
\]

oracle 动作是 \(k_i^*=\arg\min_k L_{ik}\)。令标准化的最佳与次佳动作差为
\(\tilde g_i\)，V2 的最终监督目标是：

\[
t_i=\begin{cases}
k_i^*, & \tilde g_i\ge \gamma,\\
2, & \tilde g_i<\gamma,
\end{cases}
\qquad \gamma=0.005.
\]

action index 2 对应 dose 1，即 A2 neutral。阈值 `0.005` 依据 V1 日志设定：
失败批次的 normalized gap 均值仅 `0.00378`，这些弱差异不应被当成可靠 oracle；
P90 为 `0.01907`，仍保留了一部分明确 strong 样本。目标、base logits、encoder
表示、原型残差和 \(\beta\) 在辅助路径中全部 `detach`。标签只生成训练期目标，
不进入推理特征或最终 logits。

### 2.2 无标签、无批依赖路由器

路由器输入由 `h.detach()` 和九个逐样本标量组成：

- 未再次 `tanh` 的 base margin 及其绝对值；
- 未再次 `tanh` 的实际 A2 residual margin 及其绝对值；
- 正、负原型相似度及 prototype margin；
- prototype ready；
- base 与 residual 的方向一致性。

网络为：

```text
Linear(dim_h + 9, 32) -> GELU -> Linear(32, 4)
```

不使用 dropout。最后一层权重和偏置均初始化为 0，所以初始动作概率严格均匀。
所有运算均按样本执行，不使用 batch mean、batch extrema 或批内中心化。

### 2.3 不确定时严格回退 A2

设动作概率为 \(q_{ik}\)，动作熵与置信度为：

\[
H(q_i)=-\sum_k q_{ik}\log q_{ik},\qquad
c_i=1-\frac{H(q_i)}{\log 4}.
\]

先计算期望动作，再围绕 A2 剂量 1 做置信度收缩：

\[
\bar a_i=\sum_k q_{ik}a_k,
\qquad
\hat a_i=1+c_i(\bar a_i-1).
\]

最终预测为：

\[
\boxed{
z_i=z_i^{base}+\hat a_i\beta r_i d_i
}
\]

当路由器均匀时，\(c_i=0\)、\(\hat a_i=1\)，模型精确回退 A2。与 CAMPR
不同，这一保证不依赖当前 batch 的组成或大小。V2 进一步把 argmax action 2
视为显式 abstention：前向 dose 直接置为 1，保证学习后的 neutral 决策也精确
回退 A2；反向使用 straight-through soft dose，主分类损失仍能纠正错误 abstention。

### 2.4 稳健的动作辅助损失

令最小和次小反事实损失的差为 \(g_i\)，并用该样本四个动作损失的平均绝对值
归一化：

\[
\tilde g_i=
\frac{L_i^{(2)}-L_i^{(1)}}
{\max(\frac14\sum_k |L_{ik}|,\epsilon)}.
\]

V1 使用连续 gap weight，弱样本权重接近 0，结果只有单一 strong action 子集在
训练 router。V2 改为“阈值 abstention + 最终 target 动作组平衡”。设动作组
\(G_k=\{i:t_i=k\}\)，先在组内保留类别 sample weight \(\omega_{y_i}\)：

\[
u_i=\frac{\omega_{y_i}}
{\sum_{j\in G_{t_i}}\omega_{y_j}}.
\]

每个当前 batch 中出现的最终动作组此时总权重都等于 1，再把全部权重归一化为
batch mean 1，得到 \(\hat u_i\)。因此 neutral 组与每个出现的 strong action 组
获得相等、受控的总监督质量，组内仍保留类别不平衡修正。动作损失为：

\[
\mathcal L_{action}=\frac1N\sum_i
\hat u_i\operatorname{CE}(s_i,t_i).
\]

总损失为：

\[
\mathcal L=\mathcal L_{task}+0.05\mathcal L_{action}.
\]

辅助损失只在 epoch 10 至 150（含端点）启用；主分类损失在完整 500 epoch 内
仍可训练 router。默认 \(\epsilon=10^{-4}\)、\(\gamma=0.005\)。

## 3. 相对已有方法的创新点

1. **从二值 help gate 改为可解释的多剂量决策。** 模型直接学习“关闭、减半、
   保持、增强”，而不是间接预测一个难以解释的连续 gate。
2. **决策点与部署点一致。** 四个反事实候选直接作用于 A2 residual；剂量 1 就是
   被比较和被回退的原模型。
3. **弱证据显式弃权。** normalized gap 低于阈值时，监督目标就是 A2 neutral，
   而不是给带数值噪声的 argmin 一个接近零但方向偏置的权重。
4. **动作组质量守恒。** 类别加权后再把每个出现的最终 target 组归一到相等总质量，
   防止单一 strong action 子集像 V1 一样支配 router。
5. **推理无标签且无 batch 依赖。** 推理只使用当前样本的表示、margin 和原型相似度；
   batch 顺序、大小和同批样本不会改变输出。
6. **双重 A2 fallback。** uniform router 和学习后的 neutral argmax 都在前向精确使用
   dose 1；straight-through 路径保留主任务梯度。

## 4. 第一阶段：困难 LI 数据集配对证伪

所有任务从一开始使用 A2 相同的 500-epoch scheduler；中途 epoch 119/239/349
只读取同一条轨迹，不启动短 scheduler 实验。V2 必须从 epoch 0 新跑，不能恢复
V1 checkpoint。

| 数据集 | CPAR seed | matched A2 seed | A2 val-select Test F1 | A2 raw-best（仅诊断） |
|---|---:|---:|---:|---:|
| Small-LI | 42 | 42 | 0.46247 | 0.50667 |
| Large-LI | 44 | 44 | 0.30108 | 0.44720 |

正式放行条件预先固定为：

- 两个任务都完成 epoch 499；
- 两个数据集的“最佳 validation F1 epoch 对应 Test F1”均比 matched A2 至少高
  `+0.005`；
- raw-best 只作诊断，不参与模型选择；
- 只有两者同时通过，才进入 Medium-HI seed42 与 Large-HI seed43。

建议中途证伪规则：epoch 199 时任一数据集相对 matched A2 的 validation peak
低于 `-0.015`，且机制诊断显示动作置信度或有效 gap 覆盖率已塌缩，可成对停止。
不能用中途 Test F1 调参。

## 5. 运行与审计

分支：

```text
feature/cpar-k4-counterfactual-action-router
```

队列脚本：

```bash
CPAR_V2_GPU_ALLOWLIST=3,4 \
python run/cpar_k4_v2_formal_queue.py
```

默认允许 GPU1-6，始终排除 GPU0。与其他方案并发时必须显式传入本方案独占的
两张卡。默认将命令行含 `ocr` 的进程视为非阻塞，但仍要求至少 9000 MiB 空闲显存。

独立输出：

```text
results/cpar_k4_v2_<shortSHA>_formal500/
.cpar_k4_v2_<shortSHA>_formal500_queue.events
.cpar_k4_v2_<shortSHA>_formal500_active/
.cpar_k4_v2_<shortSHA>_formal500_failed/
```

V2 脚本、目录和 run name 都包含版本与当前短 commit。它们不会读取 V1 的
`results/cpar_k4_formal500`、events 或 marker。队列启动前检查分支和干净工作区，
禁止跨版本、跨 commit 自动恢复。

正式审计：

```bash
python run/cpar_k4_v2_formal_audit.py --epoch-limit 499
```

audit 优先读取当前 worktree 的 `results/dmprd_formal500`；若不存在，则自动读取
服务器上保存权威 A2 结果的兄弟 worktree `FraudGT_dmprd_quickcheck`。也可用
`--a2 PATH` 显式指定。

中途只读审计示例：

```bash
python run/cpar_k4_v2_formal_audit.py --epoch-limit 119
```

## 6. 必查机制诊断

- uniform router 是否使 dose 精确等于 1；
- 同一样本改变 batch 分块和顺序后，route/logits 是否不变；
- strong coverage 与 abstain/neutral 比例；
- 四个 strong oracle action 比例、最终 target 组比例和组平衡后的监督质量；
- normalized gap 均值、P90 与阈值；
- route confidence、dose 均值/标准差/极值、精确 A2 比例及上下边界比例；
- A2 相对动作 oracle 的平均损失差；
- prototype bank 是否仍只在当前 batch 预测完成后更新。

若 CPAR-K4 最终失败，保留 commit、配置、结果和失败机制，不通过修改 scheduler、
读取 raw-best Test F1 或跨版本续训挽救结论。
