# ReCal-LM

ReCal-LM 是一个可运行的 PyTorch 原型，用来验证“模块化循环认知架构”
在语言模型上的第一阶段：单一语言 R 循环体、完整注意力教师、状态漂移估计、
Router/Executor 调度信号，以及与普通 Transformer Baseline 的对照训练流程。

这个仓库不是最终的多模态系统，而是完整设计路线里的第一块地基。

## 核心想法

设计目标不是制造一个必须同时运行所有能力的巨大单体模型，而是建立一个
**持续存在的统一认知空间（Shared Workspace）**。

语言、视觉、音频、控制、记忆、工具调用等能力未来都可以成为独立但可联合训练
的 `R` 循环模块。它们不需要使用完全相同的网络结构，但需要学会在同一个
Workspace 中读写、融合和更新状态。

一句话概括：

```text
统一状态、模块计算、动态路由、局部更新、漂移监测、全局校正
```

这套架构希望把“思考”从一次前向传播，变成一个可持续闭环：

```text
当前状态 S_t
  ↓
Router / Executor 判断下一步
  ↓
选择要运行的 R 模块、循环次数和算力预算
  ↓
局部 / 联合循环计算得到 ΔS_t
  ↓
更新为 S_(t+1)
  ↓
Decoder / Action 输出语言、视觉结果或控制动作
  ↓
环境反馈 / 新输入
  └──────────────↺
```

## 最小闭环

当前最小闭环可以理解为：

```text
输入 / 当前状态
        ↓
      主 R
        ↓
当前执行模块解码器
        ↓
执行器 / Router
        ↓
选择下一轮要激活的 R 模块或循环深度
        └──────────────↺
```

在完整设计里，Router 会决定下一轮激活哪些 `R_i` 模块、每个模块运行几次、
分配多少算力，以及是否需要全局校正。

在当前 ReCal-LM 原型里，先不引入多个 `R_i`。Router/Executor 被限制为
**单一 R 的执行器**：它只预测当前语言 R 应该运行多少轮，以及是否接近需要
全局校正。

## 当前已经实现

当前代码实现的是设计路线的第 1 阶段：**单一语言 R**。

已实现内容：

- ReCal 模型：`E -> A/front -> R/recurrent -> B/back -> LMHead`
- Baseline 模型：`E -> Transformer -> LMHead`
- RMSNorm、RoPE、SwiGLU、embedding/head 权重绑定
- 完整注意力 / full-calibration 教师路径
- recurrent state 局部更新路径
- 语言建模损失 `loss_lm`
- 状态蒸馏损失 `loss_state`
- logits 蒸馏损失 `loss_kd`
- DriftEstimator 漂移估计头
- Router/Executor 单 R 调度头
- 循环深度选择：`N in {1, 2, 4, 8}`
- 20M / 150M ReCal 与 Baseline 配置
- 3 次 500M tokens 试验 gate，再决定是否进入 3B tokens 完整实验

当前还没有实现：

- 多个专业 `R_i` 模块
- Vision / Audio / Control / Memory / Tool-use 等模块
- 跨模块 Router
- 真正运行时的 `d_t > tau` 自动触发 Full Global Attention
- 环境反馈 / 控制动作闭环
- 图像或视频逐层生成系统

所以目前项目不是最终多模态架构，而是先把语言版最基础闭环跑通。

## 当前代码中的架构对应关系

设计中的概念和当前代码对应如下：

| 设计概念 | 当前实现 |
| --- | --- |
| Shared Workspace | `full["state"]` / recurrent state |
| 主 R 循环体 | `self.recurrent` |
| 完整注意力教师 | `full_calibration()` |
| 局部状态更新 | `recurrent_step()` |
| 从状态解码 | `decode_from_state()` |
| Router / Executor | `RouterExecutor` |
| 漂移估计 | `DriftEstimator` |
| 全局校正信号 | `router_calibration_prob`，目前只是预测信号 |

重要文件：

- `recal/model/recal_model.py`：ReCal-LM 主模型、Router/Executor、DriftEstimator
- `recal/model/layers.py`：Transformer block、RoPE、RMSNorm、SwiGLU
- `recal/model/baseline.py`：普通 Transformer Baseline
- `scripts/train.py`：训练入口
- `scripts/evaluate.py`：验证入口
- `scripts/run_gate_experiment.py`：500M tokens gate 试验编排
- `configs/recal_20m.yaml`：小模型 smoke 配置
- `configs/recal_150m.yaml`：150M ReCal 配置
- `configs/baseline_150m.yaml`：150M Baseline 配置

## Attention Teacher 训练思路

完整注意力作为高成本教师：

```text
S_t* = FullAttention(history + new input)
```

循环体预测：

```text
S_hat_t = R(S_(t-1), delta_input)
```

训练目标不是只让模型预测下一个 token，而是让 R 学会：

```text
新信息进入以后，统一认知状态应该怎样变化
```

因此 ReCal-LM 同时训练：

- token 预测能力
- recurrent state 对齐 full-calibration state 的能力
- recurrent logits 贴近 teacher logits 的能力
- drift estimator 判断状态偏移的能力
- router/executor 预测循环深度和校正需求的能力

## Router/Executor 当前计划

完整设计里的 Router/Executor 最终应该做到：

- 根据当前 Workspace 判断下一轮该激活哪些 `R_i`
- 决定每个 R 运行几轮
- 判断应该把算力集中到哪种模态、对象或局部区域
- 判断是否需要触发全局注意力校正
- 让系统形成“思考 - 执行 - 反馈 - 再思考”的闭环

当前项目先实现最小版本：

- 不选择多个 R 模块
- 只在单一语言 R 上预测 loop-depth
- 输出 `router_expected_loop_steps`
- 输出 `router_selected_loop_steps`
- 输出 `router_calibration_prob`
- 使用 `loss_router` 训练调度头

这样可以先验证 Router 信号、训练稳定性和指标记录，而不会过早把系统扩散到
多模态复杂度里。

## DriftEstimator 当前计划

完整设计中，连续局部 R 循环会产生状态漂移：

```text
d_t = DriftEstimator(S_t)
```

当：

```text
d_t > tau
```

触发：

```text
Full Global Attention / Calibration
```

当前项目已经实现 DriftEstimator，但还没有把它接成真正的运行时强制校正。

现在的训练方式是：

- recurrent loop 产生 `loop_state`
- full-calibration teacher 产生 `target_state`
- 用 cosine drift 构造 `drift_target`
- DriftEstimator 预测 `drift_pred`
- 使用 `loss_drift` 训练 drift 预测

后续计划是把 `router_calibration_prob` 和 `drift_pred` 接入运行时逻辑：

- 漂移低：继续局部 R 循环
- 漂移中：增加循环次数或降低步长
- 漂移高：触发 full-calibration
- 关键决策：强制 full-calibration

## 扩展路线

### Stage 1：单一语言 R

当前仓库正在做这一阶段。

目标：

- 跑通 recurrent state update
- 跑通 Attention Teacher
- 跑通 DriftEstimator
- 跑通单 R Router/Executor
- 与 Baseline 做可复现实验对比

### Stage 2：Language + Vision

下一阶段加入视觉模块：

- Text Encoder 和 Vision Encoder 写入同一个 Workspace
- Language R 和 Vision R 联合训练
- 先做图像问答或场景理解
- 验证共享认知空间是否能表达跨模态状态

### Stage 3：Language + Vision + Audio

加入音频模块：

- 专注听内容：`Language + Audio (+ Memory)`
- 视频理解：`Text + Vision + Temporal + Audio`
- 训练 Router 学会按任务激活双模块或三模块组合

### Stage 4：加入 Control

形成现实闭环：

```text
感知 → 思考 → 执行 → 新感知
```

例如摄像机控制：

- Vision：判断画面和目标位置
- Language / Core：表示目标、计划和当前状态
- Control：生成移动、旋转、缩放、对焦等控制意图

可以先两两训练：

- `Vision + Control`：视觉闭环跟踪
- `Language + Control`：指令到动作策略
- `Vision + Language`：场景理解和空间关系

之后再做三模块联合训练。

### Stage 5：更多专业 R

后续可以扩展：

- 长期记忆
- 空间推理
- 触觉
- 机械控制
- 工具调用
- 专业推理模块
- 图像 / 视频逐层生成模块

关键不是把所有能力塞进一个固定网络，而是让新模块学会围绕同一个 Workspace
读写和协同。

## 图像 / 视频逐层生成计划

完整设计中，生成不要求一次性完成全部内容，而是逐轮关注：

```text
整体布局
→ 人物主体
→ 背景
→ 图层
→ 局部细节
→ 动作
→ 时间连续性
→ 音频
→ 全局校准
```

Router 可以根据当前误差、用户目标或模型自身判断，把下一轮计算集中到特定
图层、对象或模态。

当前 ReCal-LM 尚未实现这一部分，只保留为后续多模态路线。

## 模型规模

150M ReCal 配置：

- hidden size：768
- physical transformer layers：18
- `front_layers: 7`
- `recurrent_layers: 4`
- `back_layers: 7`
- `ffn_dim: 2048`
- Router/Drift hidden size：384

已验证参数量：

- `configs/recal_20m.yaml`：`23,317,056`
- `configs/recal_150m.yaml`：`153,632,256`

## 快速验证

从 `F:\PyTorch_venv\PyTorch` 运行：

```powershell
.\.venv\Scripts\python.exe -m compileall .\ReCal-LM\recal .\ReCal-LM\scripts .\ReCal-LM\tests
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_20m.yaml --dry-run
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --dry-run
```

CPU 一步 smoke 训练：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_20m.yaml --steps 1 --batch-size 1 --seq-len 16 --random-data --device cpu --output .\ReCal-LM\runs\smoke-router-drift --save-interval 1
```

训练指标会包含：

- `loss_lm`
- `loss_state`
- `loss_kd`
- `loss_drift`
- `loss_consistency`
- `loss_router`
- `drift_pred`
- `drift_target`
- `router_expected_loop_steps`
- `router_selected_loop_steps`
- `router_calibration_prob`

评估 checkpoint：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\evaluate.py --config .\ReCal-LM\configs\recal_20m.yaml --checkpoint .\ReCal-LM\runs\smoke-router-drift\checkpoint_last.pt --batch-size 1 --seq-len 16 --batches 1 --loop 2 --device cpu
```

## 150M 本地演示

小显存 GPU 可以从短序列、小步数开始：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\recal-150m-demo
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\baseline_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\baseline-150m-demo
```

如果 CUDA 显存紧张，可以把 `--seq-len` 降到 32。`--device cpu` 适合正确性检查，
不适合评估真实训练速度。

## 3x500M Gate，再进入 3B

进入 3B tokens 完整实验之前，先做 3 组配对试验：

- ReCal 训练 3 次
- Baseline 训练 3 次
- 每次 pilot 目标为 500M training tokens
- 所有 checkpoint 使用同一份验证集评估
- 只有当 ReCal 平均验证 loss 更低，并且 3 次中至少 2 次胜出，才进入 3B

生成计划但不启动长训练：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --plan-only --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --output .\ReCal-LM\runs\gate_500m
```

启动真实 6 个 pilot 任务：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --grad-accum 1 --output .\ReCal-LM\runs\gate_500m
```

说明：

- `path\to\train.txt` 和 `path\to\val.txt` 是占位路径。
- `--plan-only` 不训练，只输出计划和警告。
- 真训练模式遇到占位路径或缺失文件会立即停止。
- 已有 `checkpoint_last.pt` 时默认继续训练。
- 使用 `--skip-existing` 可以跳过已有 checkpoint 的 run。
- gate 通过后脚本写出 `start_full_3b.ps1`。
- 只有明确需要自动启动 3B 时才加 `--auto-full`。

## Tokenizer 和数据

训练 tokenizer：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train_tokenizer.py --input path\to\text.txt --output .\ReCal-LM\artifacts\tokenizer.json --vocab-size 32000
```

抓取 FineWeb-Edu 小样本：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\prepare_data.py --conf .\ReCal-LM\.conf --output .\ReCal-LM\data\fineweb_edu_sample.jsonl
```

下载 500MB FineWeb-Edu 本地样本：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\download_fineweb_edu.py --target-bytes 500M --val-bytes 10M --output-dir .\ReCal-LM\data\fineweb_edu_500m
```

`.jsonl` 训练数据使用 streaming 方式读取，不会一次性全部加载进内存。

## 完整全参数训练参数说明

完整训练入口是 `scripts/train.py`。它默认训练 ReCal 或 Baseline 的全部参数；如果发现有参数被冻结，会在启动阶段报错。只有明确做局部训练实验时，才使用 `--allow-frozen-params`。

启动时会打印 `trainability` JSON，用来确认每个主要模块是否都参与训练：

- `embed_tokens`：输入 token embedding；当 `tie_embeddings: true` 时也和 `lm_head` 共享权重。
- `front`：前段 full-calibration Transformer blocks，用于把输入序列编码成初始隐状态。
- `recurrent`：主循环体 R blocks，用于 recurrent state update。
- `back`：后段解码 blocks，把 state 转成可预测 token 的 hidden。
- `state_input`：把新 token embedding 注入旧 state 的更新投影。
- `state_norm`：recurrent state 的归一化。
- `final_norm`：输出 logits 前的最终归一化。
- `lm_head`：语言模型输出头。
- `router_executor`：预测循环步数和是否需要校准。
- `drift_estimator`：预测 recurrent state 相对 full-calibration state 的漂移。

常用训练命令：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py `
  --config .\ReCal-LM\configs\recal_150m.yaml `
  --data .\ReCal-LM\data\train.jsonl `
  --val-data .\ReCal-LM\data\val.jsonl `
  --tokenizer .\ReCal-LM\artifacts\tokenizer.json `
  --target-tokens 500M `
  --seq-len 2048 `
  --batch-size 1 `
  --grad-accum 1 `
  --device cuda `
  --save-interval 100 `
  --val-interval 100 `
  --val-batches 20 `
  --output .\ReCal-LM\runs\recal_150m_full
```

参数作用：

| 参数 | 作用 |
|---|---|
| `--config` | 选择模型结构和默认训练超参，例如 `configs/recal_150m.yaml`。 |
| `--data` | 训练文本，支持 UTF-8 `.txt` 或每行含 `text` 字段的 `.jsonl`。 |
| `--val-data` | 独立验证集；提供后 `checkpoint_best.pt` 按 `val_loss_lm` 选择。 |
| `--tokenizer` | HuggingFace tokenizers JSON；不提供时使用字节级 fallback tokenizer。 |
| `--target-tokens` | 按训练 token 数控制停止点，例如 `500M`、`3B`。 |
| `--steps` | 按 step 数控制训练；如果同时设置 `--target-tokens`，会取更早停止点。 |
| `--seq-len` | 每条样本上下文长度，不能超过配置里的 `context_length`。 |
| `--batch-size` | 每个 micro-batch 的样本数。 |
| `--grad-accum` | 梯度累积步数；有效 tokens/step = `batch-size * seq-len * grad-accum`。 |
| `--lr` | 覆盖配置里的学习率。 |
| `--device` | `auto`、`cuda` 或 `cpu`。 |
| `--resume` | 从 checkpoint 恢复模型、优化器、step 和 `tokens_seen`。 |
| `--save-interval` | 每隔多少 step 写 `checkpoint_last.pt`。 |
| `--val-interval` | 每隔多少 step 跑验证；提供 `--val-data` 后默认等于 `--save-interval`。 |
| `--val-batches` | 每次验证使用多少个 batch。 |
| `--val-loop` | ReCal 验证时固定循环步数；不设置则使用模型路由选择。 |
| `--keep-interval-checkpoints` | 额外保留 `checkpoint_步数.pt`；默认只保留 best 和 last。 |
| `--random-data` | 随机 token smoke 测试，只验证程序路径，不代表模型质量。 |
| `--compile` | 尝试使用 `torch.compile`。 |
| `--allow-frozen-params` | 允许部分参数冻结；完整全参训练不要使用。 |
| `--dry-run` | 只构建模型并输出参数量/可训练覆盖，不进入训练。 |

训练日志指标：

| 指标 | 含义 |
|---|---|
| `loss` | 总训练目标，包含 LM loss 和配置权重下的辅助 loss。 |
| `loss_lm` | 下一 token 预测交叉熵；判断语言建模质量时优先看它。 |
| `loss_state` | recurrent state 向 detached EMA teacher state 对齐的 cosine + normalized-MSE 损失。 |
| `loss_kd` | recurrent logits 向 EMA teacher state 解码出的 logits 对齐的温度 KL 蒸馏损失。 |
| `loss_drift` | DriftEstimator 对 EMA 平滑 drift accumulator 的预测误差。 |
| `loss_consistency` | R state 与 Student attention state 解码分布之间的功能一致性 KL。 |
| `loss_router` | RouterExecutor 的循环步数分类和校准二分类损失。 |
| `drift_pred` | DriftEstimator 预测的平均漂移。 |
| `drift_target` | 由 recurrent/EMA-teacher state cosine distance 构造并平滑后的漂移目标。 |
| `router_expected_loop_steps` | router 概率分布对应的期望循环步数。 |
| `router_selected_loop_steps` | 本次 forward 实际使用的循环步数。 |
| `router_calibration_prob` | router 预测需要 full-calibration 的概率。 |
| `val_loss_lm` | 验证集 LM loss；有验证集时用于选择 best checkpoint。 |
| `val_perplexity` | `val_loss_lm` 对应困惑度。 |
| `tokens_seen` | 从本次 run 或 resume 后累计的训练 token 数。 |
| `tokens_per_second` | 当前运行吞吐估计。 |

注意：当前 `router_calibration_prob` 和 `drift_pred` 已经参与训练和评估，但还没有接成推理时的强制 full-calibration 闭环。它们现在是可学习信号和诊断指标，不要直接当作已经节省计算量的证据。

## EMA Attention Teacher 与四阶段训练

当前 ReCal 的训练路径明确拆成两种注意力角色：

```text
训练梯度：Input -> Student attention (front) -> R recurrent -> Decoder -> CE
监督目标：Input -> EMA Teacher attention -> stop-gradient -> state / logit / drift targets
```

`teacher_embed_tokens`、`teacher_front` 与 `teacher_state_norm` 是 Student 对应模块的 EMA 副本。它们 `requires_grad=False`，始终处于 `eval()`，不会被 CE、KD、state、drift 或 router loss 反向更新。每一次成功的 `optimizer.step()` 后才执行：

```text
theta_teacher = ema_decay * theta_teacher + (1 - ema_decay) * theta_student
```

这样 Student attention 仍会从 CE 中学习，而 R 不会追逐一个由同一 loss 实时拖动的监督目标。EMA teacher 仅用于训练；WebUI 和部署模型会关闭它，因此推理不保留第二份 attention 参数。训练 checkpoint 会保留 teacher，以便恢复训练时延续相同的监督坐标系；旧 checkpoint 缺少 teacher 权重时会自动由已加载的 Student attention 初始化。

配置中的关键项：

| 配置项 | 作用 |
|---|---|
| `ema_teacher` | 训练时启用 EMA teacher；ReCal 全训练应保持 `true`。 |
| `ema_decay` | teacher 平滑系数。20M 默认 `0.999`，150M 默认 `0.9995`。 |
| `teacher_target_layers` | 从 teacher `front` 的哪些层提取目标；不同 R step 会由浅到深对齐这些状态。 |
| `state_mse_weight` | 在 cosine state loss 外加入 normalized-MSE 的权重，避免 hidden 幅值漂移主导训练。 |
| `kd_temperature` | logit KD 温度；实现使用 `T^2 * KL(softmax(z_T/T) || softmax(z_R/T))`。 |
| `lambda_consistency` | R 输出与 Student attention 输出的功能一致性 KL 权重；使用较小值，避免把 R 压成恒等映射。 |
| `drift_accumulator_decay` | 漂移累积 EMA 的衰减系数。DriftEstimator 学习该平滑目标，router 使用其 detach 后的值生成循环/校准标签。 |

完整训练目标为：

```text
L = L_CE
  + lambda_state * L_state(cosine + normalized MSE)
  + lambda_kd * L_KD(EMA teacher logits)
  + lambda_drift * L_drift(accumulated drift)
  + lambda_consistency * L_consistency(Student-attention logits, R logits)
  + lambda_router * L_router
```

其中所有 teacher state 和 teacher logit 都在构造目标时 `detach`。因此 KD、state 和 drift 的梯度只流向 Student attention/R/decoder/controller，不会流入 EMA teacher。

建议不要直接用 3B token 从零开始联合训练。使用同一份 train/val 数据，按以下阶段分开运行；每个阶段都应以 `val_loss_lm` 选择 checkpoint，而不是按总 `loss` 选择：

| 阶段 | 命令参数 | Student attention (`front`) | R/Decoder/Router/Drift | Teacher |
|---|---|---|---|---|
| Stage 1 | `--stage stage1` | 冻结 | 正常训练 | 固定初始 teacher，不做 EMA 更新 |
| Stage 2 | `--stage stage2` | 小 LR，默认 `0.10 * LR_R` | 正常训练 | EMA 更新 |
| Stage 3 | `--stage stage3` | 正常 LR | 正常训练 | EMA 更新 |
| Stage 4 | `--stage stage4` | 联合训练 | 联合训练，并观察路由/漂移分布 | EMA 更新 |

`tie_embeddings: true` 时 `embed_tokens` 同时也是 `lm_head`；为避免冻结 decoder 输出头，Stage 1 只冻结 `front`，共享 embedding 仍属于 decoder 主训练组。这是有意的权重共享边界。

示例：先做有验证集的 Stage 1，再从其 last checkpoint 进入 Stage 2：

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py `
  --config .\ReCal-LM\configs\recal_150m.yaml `
  --data .\ReCal-LM\data\train.jsonl `
  --val-data .\ReCal-LM\data\val.jsonl `
  --tokenizer .\ReCal-LM\artifacts\tokenizer.json `
  --target-tokens 500M --seq-len 2048 --batch-size 1 --grad-accum 1 `
  --stage stage1 --device cuda --save-interval 100 --val-interval 100 `
  --output .\ReCal-LM\runs\recal_150m_stage1

.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py `
  --config .\ReCal-LM\configs\recal_150m.yaml `
  --data .\ReCal-LM\data\train.jsonl `
  --val-data .\ReCal-LM\data\val.jsonl `
  --tokenizer .\ReCal-LM\artifacts\tokenizer.json `
  --target-tokens 500M --seq-len 2048 --batch-size 1 --grad-accum 1 `
  --stage stage2 --resume .\ReCal-LM\runs\recal_150m_stage1\checkpoint_last.pt `
  --device cuda --save-interval 100 --val-interval 100 `
  --output .\ReCal-LM\runs\recal_150m_stage2
```

切换 Stage 会改变 optimizer 参数组：恢复时模型和 teacher 权重会加载；如果旧 optimizer 参数组不匹配，脚本会明确提示并从新的 optimizer 状态继续。日志新增 `loss_consistency`、`attention_lr`、`stage` 和 `ema_decay`。`loss_state`、`loss_kd`、`drift_target` 的含义也已改为相对 EMA teacher 的监督，不再是 Student 自己的 full-calibration 输出。

## 本地 WebUI 真实检测

WebUI 文件位于 `webui/`，后端会真实加载 PyTorch 模型、checkpoint 和 tokenizer。支持两类操作：

- `检测`：对输入文本计算 LM loss、PPL、漂移、router 指标和下一 token top-k。
- `生成`：用当前模型 autoregressive 采样生成文本，并返回最后一步 router/drift 指标。

启动：

```powershell
cd F:\PyTorch_venv\PyTorch\ReCal-LM
F:\PyTorch_venv\PyTorch\.venv\Scripts\python.exe .\webui\server.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

页面中的路径必须位于项目目录内。常用填写方式：

| 输入框 | 示例 |
|---|---|
| 配置路径 | `configs/recal_150m.yaml` |
| Checkpoint | `runs/recal_150m_full/checkpoint_best.pt` |
| Tokenizer | `artifacts/tokenizer.json` |
| 设备 | `auto` 或 `cuda` |
| 循环步 | `auto`、`1`、`2`、`4`、`8` |

如果不填 checkpoint，页面会加载随机初始化模型，只能用于检查代码路径，不能用于判断模型能力。真实检测请加载训练后的 `checkpoint_best.pt` 或 `checkpoint_last.pt`。

## License

MIT License。见 `LICENSE`。
