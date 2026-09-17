> 历史训练方案，已于 2026-09-17 由 [V3-E1 独立阶段](../docs/v3-training.md) 接替。本文的 10B 主计划、原调度和多 checkpoint 保留策略不再作为活动配置。

# 已确认的正式训练方案

目标：R 核心约 3B 参数，训练 10B tokens，单卡 H20。

## 四个模型和参数

| 模块 | 参数量 |
|---|---:|
| 独立注意力 A | 324,811,776 |
| R 循环核心 | 3,052,308,480 |
| R 专属解码器 D | 324,811,776 |
| 偏移预测 P | 2,363,136 |
| 四模块合计 | 3,704,295,168 |
| 前期额外 EMA 教师 | 324,811,776 |

新实现：recal/model/four_model.py，训练入口 scripts/train_four_models.py。
各模块参数不共享，无第五个 Router；各自的 embedding/normalization 属于各自模块。teacher 是独立注意力的冻结 EMA 副本，沿用用户选择，不加载外部预训练教师。

## 前期

n→n+1 指读入新的 token 后推进状态，不是同一个输入上的内部迭代深度。当前实现每次展开 2 个相邻 token 的状态推进，可配置 rollout_steps。

- R_(n+1) = R(R_n, x_(n+1))；目标为 sg(T_(n+1))。
- A_(n+1) 学习 sg(R_(n+1))。
- D 每个 micro-batch 以 50/50 概率接收 A_n 或 R_n 的停止梯度状态，用真实下一 token 做 CE。n 从本次展开中随机选择，验证固定用 R 的最后一步。
- P 学习 R 与参考状态的 cosine 偏差；参考和输入均停止梯度，仅更新 P。
- R 初始状态来自停止梯度的 A；R 梯度沿展开序列传播。
- 每次成功 optimizer.step 后更新 EMA teacher，默认 decay=0.9995。

上述目标均在同一更新前的模型状态下计算，互相监督时目标一端不接收梯度。

## 自动切换与后期

运行更新：已按用户指令在第 4083 步手动提前切换，保留 transition_before 节点。手动切换标记持久化到 manifest，若从该早期节点重启，会自动完成已决定的切换。此次重启按既定权重保存策略重新初始化 Adam。

“轮”定义为固定验证集上的一次验证，每 500 个优化步骤执行。连续 5 次相邻验证 LM loss 的相对变化不超过 0.1% 则切换（首次判定需 6 个验证值）。任何一次超过阈值就重新计数。可以调整配置中的 validation_interval、plateau_patience 和 plateau_relative_tolerance。

切换前保存一个阶段节点，然后真正删除 teacher 模块。后期 CE 梯度沿 A→R→D 联合回传；R/attention 蒸馏损失关闭。P 的偏差参考变为停止梯度的独立注意力重算状态。这表示状态一致性偏差，不代表语义正确性。

## 数据与训练规模

清洗数据盘：/cloud/cloud-ssd1/training-pool-v1/cleaned-v1/pretrain。

按既有训练池规划使用 token 窗口配比：CCI3 52%、中文维基 8%、FineWeb-Edu 16%、英文维基 4%、FineMath 10%、代码合计 10%（当前八种语言均分）。CCI3 尚未单独拆分新闻类别。SFT 数据不混入此次预训练。

基于文档文本 SHA256 的固定 0.1% 验证划分；分词器训练也排除这些验证文档。正式 tokenizer 为 32000 byte-BPE，含完整 256 byte alphabet 和 ByteLevel decoder，训练样本约 6400 万字符；每种来源按既有比例抽样。样本和 hash 位于 artifacts/four-model-v1。

训练器逐批读取 Parquet、在线分词、按 token 窗口混合；记录每个来源的文件/batch/行游标、尚未消费的 tokens 和随机状态。若训练池不足 10B tokens，会循环读取训练划分；10B 表示累计读取量，不保证唯一 tokens。

默认 BF16 autocast、FP32 参数与 AdamW 状态；R 为 hidden_size=3072、26 层、FFN=8192，attention/decoder 各 2 层。启用非重入梯度检查点。batch=1、seq_len=512、grad_accum=8，每个完整优化步骤 4096 个输入 tokens。模型位置容量 2048，但当前正式长度为 512，不能宣称已训练 2048 上下文。

## 三套保存：用户已确认的空间取舍

位置：系统盘 /workspace/ReCal-LM/runs/four-model-r3b-10b/checkpoints。

只保留 latest、best、transition_before 三个角色，对应同一步则复用同一套；最多三套，不保留异常 loss 节点。best 按同一固定验证集的 LM loss 选择，避免早晚期不同总损失混比。

每套包含四模型 FP32 权重、阶段、训练配置、step、tokens_seen、数据游标、随机状态、分词器和验证集 hash；早期另外包含 EMA 教师权重。按用户确认，不保存 Adam 优化器状态。重启后重新初始化优化器，并明确打印警告；这不是精确续训。不中断的阶段切换保留当前优化器状态。

单套前期权重约 16.12GB（15.01GiB）。三套加一次原子写入临时文件及 2GiB 余量预算约 62.04GiB。写入完成并更新 manifest 后才清理无引用旧套；磁盘不足时保留旧 checkpoint 并报错。系统盘空间不能再用于大型额外产物。

latest 每次验证或正常终止时保存，默认最多丢失一个验证周期内的训练进度。SIGTERM/SIGINT 会在当前优化步骤完成后保存；强制断电/SIGKILL 无法保证保存。系统盘随实例删除可能丢失，保存位置按用户要求使用系统盘。

## 验证与运行证据

tests/test_four_model.py：梯度归属、模块参数独立性、EMA 更新、后期教师删除、token 位置因果性、五轮平台期、三套角色保留、失败写入和数据游标恢复。旧项目测试保留。

runs/four-model-transition-smoke：GPU 真实数据跨阶段短测（专用放宽阈值，不用于正式配置），第 6 个验证点切换，第 7/8 步后期训练，第 9 步权重恢复。

r3b-initial-check.log：实际完整 3B 核心的首次有界运行，检查 GPU 显存和大权重保存。正式运行续用同一训练目录；由于权重保存不含优化器，首次有界运行后启动长任务也会重置 Adam。
