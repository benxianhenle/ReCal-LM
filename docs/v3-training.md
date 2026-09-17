# V3：同步递归与独立训练阶段

状态日期：2026-09-17。实际阶段是 **A-final → V3-E1 warm-start**。E2/E3 是有条件的后续实验，当前 launcher 只运行 E1。

## 现有模型与继承关系

`TextStateLM` 在四模块模型上运行 `A(x) → h0 → R1 → R2 → R3`。共享 R 对整条序列同步更新；第 k 轮的输出由共享 Decoder 对 `Δk = hk − h(k−1)` 解码。旧 R 内部仍使用实际的 `input_norm(state + input_projection(state))` 和 Transformer stack，E1 不替换它。

```text
旧主线 step 33291：136,359,936 tokens，保留最终权重
    ↓ A 对照分支增加 3,899,392 tokens
A-final step 34243：140,259,328 tokens，只读
    ├── 上一轮 E1：继承旧 scheduler 进度，门槛失败，权重已清理
    └── 本轮 E1：新 AdamW + 新 scheduler + 等权 CE
            ↓ 仅通过后
           E2：函数保持的 Residual Gate
            ↓ 仅通过后
           E3：Block Drift Refresh
```

A 的 SHA256：`696cf16d1fb08b564ab8d8a4397fd4f554756db54249005c0cbfd5d51666d1a2`。加载时校验 A 及两组验证集 hash，恢复 A 的训练数据游标和 RNG，并重算 Fixed/Extra 的三个深度 CE；与已存基线逐项误差须 ≤1e-5。

## E1 的训练目标

```text
L_text = (CE1 + CE2 + CE3) / 3
L_drift = mean_k MSE(P(stopgrad(hk)),
                     clamp(1 − cosine(stopgrad(hk), stopgrad(h(k−1))), 0, 1))
L = L_text + 0.05 × L_drift
```

文本梯度穿过完整 A→R→D 图；层间没有停止梯度。Drift 的输入和监督目标停止梯度，因此这一项只训练 Drift 头。旧 `core.embedding` 已不被状态递归使用，继续冻结以兼容 checkpoint。

关闭 Attention 辅助文本项 `alpha`、Attention alignment `lambda_A`、R fixed-point `lambda_R` 和强制 CE 单调项。旧文本权重 `[0.2,0.4,1.0]` 总和为 1.6，新权重总和为 1；比较时需考虑总梯度尺度也发生变化。

## 新 optimizer 与 scheduler

| 参数 | 设置 |
|---|---|
| AdamW | 全新状态；betas=(0.9,0.95)，weight_decay=0.1，foreach=False |
| 新阶段索引 | phase_step 1 起，与旧 global_step 分开记录 |
| 前 300 步 | LR(step)=5e-5 × step/300；首步约 1.6667e-7 |
| 第 300 步以后 | LR=5e-5 恒定 |
| 梯度裁剪 | 全局 L2 norm 上限 1；记录裁剪前 norm |
| 预算 | 3M 上限内向下取完整更新：732 步、2,998,272 tokens |
| 时间上限 | 从 plan 的 started_at 算起 2 小时；监督进程负责终止 |

每步为 8 次累积、每次 1×512 tokens，BF16 autocast，FP32 参数和优化器状态。A 没有序列化 Adam 状态，本次也明确要求重置。scheduler 不再引用 A 的历史步数。

## 验证与阶段门槛

在起点及第 245、489、732 步评估：Fixed 为 8×512 tokens，Extra 为 24×512 tokens。所有路径在整条序列同一深度上真实执行。

两组都必须满足：

1. CE3 相对 A 的增幅不超过 0.03。
2. `max(CE1,CE2,CE3) − min(CE1,CE2,CE3)` 比 A 缩小超过 1e-5。
3. `Var(h3)/Var(h1)` 和 `Var(Δ3)` 都保留 A 的至少 90%。

方差先沿各序列 token 维计算，再对 hidden 维与验证批次取均值。这些数值沿用前轮的操作化门槛，不是统计显著性检验；不强制 CE3<CE2<CE1。

出现非有限 loss/梯度则报错退出；裁剪前梯度范数超过 1000 时停止更新；任一验证组 CE3 相对 A 恶化超过 0.03 连续两次则提前停止。不完整阶段不能通过。

## 后续结构演化

**E2：**用 `hk + gk × (R_old(hk) − hk)`，初始 `gk=1`，保留旧 R 的参数。在实数代数上等价于旧映射；浮点实现需做误差容限内的输出校验。现有小模型测试验证轨迹兼容和因果性。不能以全新随机 R 替换旧 R，也不能直接改成 `h + R_old(h)`。

**Anchor branch：**当前 E1 没有独立的固定 Anchor 输入支路。将来可增加 `Wa × a` 且 `Wa=0` 初始化，以保留初始函数；它仍属于待实现/验证的演化方案。

**E3：**Drift 的职责改为判断 block 是否需要 Attention refresh。已有实验原型采用 128-token block，通过 Attention 与 R 候选路径的 CE 产生训练标签。在线特征与决定必须遵守因果性；小模型因果测试已存在，实模型收益尚未验证。旧 per-token STOP/CONTINUE 不作为当前主线。

## 存储与可恢复性

本轮只在通过门槛时保存一份 E1 终点；失败则只保留指标、最终数据游标/RNG，不保存实验模型。没有中间权重或 optimizer checkpoint，因此遇到中断可能需要从 A 重新开阶段，不能承诺精确恢复。

历史实验已在新训练首次更新后归档。数据、验证 tensor、分析缓存及元数据保留；旧实验模型清理。后续阶段须明确指定新的输出目录和新的 plan，不能覆盖 A。
