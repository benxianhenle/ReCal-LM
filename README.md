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

## 发布到 GitHub

仓库包含 `scripts/publish_github.ps1`，用于创建 GitHub 仓库并推送 `main`。
脚本不会把 token 写入 Git 配置，也不会把 token 保存到文件。

运行：

```powershell
.\scripts\publish_github.ps1
```

脚本会询问：

- 仓库名，默认 `ReCal-LM`
- 可见性，默认 `public`
- 仓库说明
- GitHub token，隐藏输入

token 权限：

- Classic token：公开仓库用 `public_repo`，私有仓库用 `repo`
- Fine-grained token：需要能访问目标仓库，并有 `Contents: Read and write`

当前本地 remote：

```text
origin https://github.com/benxianhenle/ReCal-LM.git
```

## Git 安全规则

仓库会忽略本地秘密、数据、缓存和训练产物：

- `.conf`
- `.hf_cache/`
- `data/`
- `runs/`
- `artifacts/`
- Python cache
- model checkpoint files

不要提交 API token、数据集大文件、checkpoint 或本地缓存。

## License

MIT License。见 `LICENSE`。
