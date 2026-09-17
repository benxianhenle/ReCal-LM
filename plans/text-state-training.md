> 历史训练方案，已于 2026-09-17 由 [V3-E1 独立阶段](../docs/v3-training.md) 接替。本文的 10B 主计划、原调度和多 checkpoint 保留策略不再作为活动配置。

# 文本主导的状态循环策略（2026-09-14）

启动基线：旧正式训练 step 12810，52,469,760 输入 tokens。旧权重没有 Adam 状态；不能恢复已丢失的动量。保存 original-baseline-metadata.pt 中的数据游标、随机状态、原配置和可重建的余弦学习率调度信息；优化器重新初始化，保持原 10B token 学习率 horizon。

## 前向与梯度

- A(x)：原 token embedding + 注意力 Transformer stack。
- A(hidden)：同一个注意力 stack 接收隐藏状态，不重复 token embedding。
- R(hidden)：input_norm(hidden + input_projection(hidden)) 后接原 26 层 R stack；三个推理深度使用同一组 R 权重、同一 token 位置，不再读入下一个 token。旧 R token embedding 保留但冻结，因此总 R 参数 3,052,308,480，其中活跃参数 2,954,004,480。
- 每轮 D 解码相邻状态差 delta，三轮都预测相同位置的真实下一 token，深度加权 CE 为 0.2、0.4、1.0 的未归一化求和。日志 loss_lm 是第三轮 CE，loss_text 是加权总和，不混用两者。
- A 文本辅助：A(stack) 接收 detached R 状态；减去 detached 前一状态；冻结 D 参数但保留输入梯度，CE 仅更新 A。
- A 追随 R：LN 后余弦距离 + 0.1 SmoothL1，目标和输入 R 状态都 detached。
- R 固定点：单独重算以 detached 初始 A 状态为起点的完整三轮 R（无逐轮 detach），穿过冻结参数的 A 求距离。只更新 R；原主文本轨迹完整 BPTT 不受影响。dropout 为 0，重算轨迹与主轨迹数值一致。此额外计算只在第三阶段开启，届时显存和吞吐还需观察。
- P 保留独立训练：预测相邻状态的余弦偏移，输入与标签 detached，系数 0.05。旧 P 标签是 R 与对齐位置 A 的偏移，新定义匹配状态循环语义。

## 启用时点（按输入 tokens 计，不乘循环次数）

1. 累计 52,469,760 → 102,469,760：alpha 0 → 0.10，lambda_A=lambda_R=0。
2. 累计 102,469,760 → 202,469,760：alpha=0.10，lambda_A 0 → 0.05，lambda_R=0。
3. 累计至少 202,469,760，且连续三次无退化警报、主 CE 不差于新基线、状态余弦优于新基线后，lambda_R 用接下来的 100M tokens 从 0 增至 0.01。条件未满足时保持关闭，不能只按 token 数强行开启。
4. 最终系数 0.10 / 0.05 / 0.01，R 系数上限始终为当前 A 对齐系数的 1/5。

## 验证与保护

复用原固定 8 批，以及前一天生成的额外固定 24 批 held-out 数据；新路径重新测量起点，不能与旧状态解码路径 CE 混排。记录三轮 CE、A(R) 与 R 余弦、范数比、状态范数、delta 范数、token 方差、A/R/D/P 梯度范数。前三轮均是相同标签，深度 CE 可以直接比较。

初期在第 1、20、100、200、300、400、500 次更新验证；同时保留原每 500 全局步周期。正式 checkpoint 从第 20 次更新起保存。

保护：状态范数低于基线 70%、token 方差低于 50%、delta 范数低于 30%、第三轮 CE 比第一轮差 2% 以上，或余弦 >0.995 且 CE 未改善，关闭 R 约束并在 A 对齐已开启时减半其系数。主验证 CE 连续三次恶化超过 0.2% 时额外减半 alpha，连续五次时保存并停止。基线较小，警报属于运行保护，不构成理论上的坍缩证明。

## 权重和恢复

继续共用 runs/four-model-r3b-10b/checkpoints 存储，最多三套：latest、按新路径固定验证 CE 排名的 best、新策略原始基线 step 12810。第一次成功保存新权重时，原 4083 阶段边界和旧目标的 best 将退出保留集合。新基线无需复制大文件；切换清单随新 checkpoint 原子发布。

新日志位于 runs/four-model-text-state-10b。启动脚本 scripts/train_text_state.py 会从共享清单 latest 恢复，严格检查新配置和两组验证数据 hash，恢复数据、RNG 和策略调度；仍只保存权重，因此每次重启 Adam 会重置。原始基线元信息单独保留，不能冒充完整优化器恢复点。

启动：OMP_NUM_THREADS=4 /workspace/ai-training/.venv/bin/python -u scripts/train_text_state.py
停止：向 launcher.json 中训练 PID 发送 SIGTERM，等当前更新、验证和保存完成。不要直接杀死写权重中的进程。

检查：41 项测试通过。实际 H20 全规模验证和运行结果见新日志、initial_validation.json、status.json。

2026-09-14 日志频率调整：常规 training.log、metrics.jsonl 和 status.json 每 20 步写入一次（--log-every 可配置），验证和退出状态立即写入。日志附带窗口平均 loss_lm 和最大裁剪前梯度范数，保留窗口内波动信息。早期加密验证按策略累计更新数计算，重启不会重复触发第 1、20 步验证与保存。
