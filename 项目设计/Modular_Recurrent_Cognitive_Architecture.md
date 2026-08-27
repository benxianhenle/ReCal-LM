# 模块化循环认知架构
## Modular Recurrent Cognitive Architecture

## 1. 创建目的

目标不是构建一个必须同时运行全部能力的巨大单体模型，而是建立一个**持续存在的统一认知空间（Shared Workspace）**。语言、视觉、音频、控制等能力以独立但可联合训练的 R 循环模块存在，并根据当前任务动态组合。

> **统一状态、模块计算、动态路由、局部更新、漂移监测、全局校正。**

统一的不是所有模块的网络结构，而是它们能够在同一个认知空间中读写、融合和更新信息。

## 2. 最简单闭环

```text
输入 / 当前状态
        ↓
      主 R
        ↓
当前执行模块解码器
        ↓
执行器 / Router
        ↓
选择下一轮要激活的 R 模块
        └──────────────↺
```

主 R 更新 Workspace。当前执行模块的解码器把内部状态转换成可解析的操作意图。执行器网络决定下一轮使用哪些 R 循环体、运行几次、分配多少算力，并形成闭环。

## 3. 动态组合

- 专注听内容：`Language + Audio (+ Memory)`
- 图像问答：`Vision + Language`
- 视频生成：`Text + Vision + Temporal + Audio`
- 摄像机控制：`Vision + Control + Language`

摄像机控制可以三者同时训练，也可以两两训练：
- `Vision + Control`
- `Language + Control`
- `Vision + Language`

## 4. 共享认知空间

```text
Text Encoder ─────┐
Vision Encoder ───┼──→ Shared Workspace ←→ Specialized R Modules
Audio Encoder ────┤                           │
Control Feedback ─┘                           ↓
                                       Router / Executor
                                              │
                                              └────↺
```

模块可以拥有自己的适合结构，但新模块不能完全脱离核心单独形成语义。它需要与核心语言 / 认知模块联合训练，使它学会在同一个 Workspace 中表达、读取与修改状态。

## 5. 注意力教师训练

完整注意力作为高成本教师：

\[
S_t^* = FullAttention(history + new\ input)
\]

循环体预测：

\[
\hat S_t = R(S_{t-1}, \Delta input)
\]

训练 R 逼近完整计算得到的新状态，同时训练当前解码器从状态中正确提取 token、视觉结果或控制指令。

因此 R 学习的是：

> **新信息进入以后，统一认知状态应该怎样变化。**

## 6. 特定耦合联动训练

以摄像机控制为例：

### 三模块联合
`Vision + Language + Control`

- Vision：判断画面和目标位置
- Language / Core：表示目标、计划与当前状态
- Control：生成移动、旋转、缩放、对焦等控制意图

### 两两耦合
- `Vision + Control`：视觉闭环跟踪
- `Language + Control`：指令到动作策略
- `Vision + Language`：场景理解和空间关系

之后再进行多模块联合训练。

## 7. 漂移累计与全局注意力纠正

连续局部 R 循环会产生状态漂移，因此训练漂移估计器：

\[
d_t = DriftEstimator(S_t)
\]

当：

\[
d_t > \tau
\]

触发：

\[
Full\ Global\ Attention / Calibration
\]

完整全局计算重新整合相关历史与当前状态，并将 Workspace 重新锚定。

## 8. 图像 / 视频逐层生成

生成不要求一次性完整完成，可以逐轮关注：

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

Router 根据当前误差、用户目标或模型自身判断，把下一轮计算集中到特定图层、对象或模态。

## 9. 扩展路线

1. **单一语言 R**
   - 循环状态更新
   - Attention Teacher
   - 漂移估计
   - Full Calibration

2. **Language + Vision**
   - 共享 Workspace
   - 双模块联合认知训练

3. **+ Audio**
   - 动态双模块 / 三模块组合
   - Router 学习

4. **+ Control**
   - 感知 → 思考 → 执行 → 新感知
   - 构成现实闭环

5. **更多专业 R**
   - 触觉
   - 空间
   - 机械控制
   - 长期记忆
   - 工具调用
   - 其他专业推理模块

## 10. 最终闭环

```text
S_t
 ↓
Router / Executor
 ↓
选择 {R_i, R_j, ...}
 ↓
局部 / 联合循环计算
 ↓
ΔS_t
 ↓
S_(t+1)
 ↓
Decoder / Action
 ↓
环境反馈 / 新信息
 └──────────────↺
```

最终目标不是固定一个“万能模型”，而是形成一种**能够围绕统一认知空间不断增加和组合能力的模块化思考架构**。
