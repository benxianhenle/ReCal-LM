# 代码变更与验证

此次更新将早期原型之后在真实训练环境使用的四模块训练、状态递归、诊断与 V3 实验源码同步到仓库，同时保留早期小模型接口。活动训练进程使用的模型和 objective 文件未在此次文档整理中更改。

## 主要实现

| 文件 | 变化与作用 |
|---|---|
| `recal/model/four_model.py` | 独立 A、R、D、Drift；早期 EMA teacher 与后期联合文本路径 |
| `recal/model/text_state_model.py` | 三轮共享状态递归、D(Δ) 解码、多深度 CE、辅助梯度隔离 |
| `recal/data/stateful.py` | 文档 SHA256 验证划分、Parquet token 窗口混合、文件/行/缓冲区/RNG 恢复 |
| `recal/data/dynamic_v2.py` | V2 calibration/train/validation 文档分区 |
| `recal/training/retention.py` | 历史 latest/best/阶段节点保存与原子 manifest；不代表本轮三套保留策略 |
| `recal/training/stages.py`、`text_state_schedule.py` | 历史阶段切换与辅助损失调度 |
| `recal/evaluation/state_ablation.py`、`dynamic_depth*.py` | 深度扫描、消融、真实混合路径、离线 Oracle 分析 |
| `recal/evaluation/synchronous_v3.py` | 同步三深度等权 objective、方差指标、阶段 Gate；可选 Residual Gate |
| `recal/evaluation/block_refresh_v3.py` | 后续 block refresh 实验原型，尚未实模型验收 |
| `scripts/run_v3_e1_warmstart.py` | 从 A 重置 optimizer/scheduler；独立 phase 计数；hash/基线复现；失败模型不落盘 |
| `scripts/supervise_v3_e1_warmstart.py` | 独立日志、训练进程锁配合、时间上限与信号终止 |
| `tests/test_v3_e1_warmstart.py` | 验证首步 LR、warmup 结束和末步 LR，不再继承旧调度进度 |
| `tests/test_synchronous_v3.py` | 等权 CE 数值及梯度、gate 初始化、因果性、双验证组门槛 |

`configs/four_model_3b.yaml` / `four_model_text_state_3b.yaml` 是历史基础配置，本轮在加载 A 后覆写权重与学习率；实际配置以 [manifest](evidence/2026-09-17/v3-e1-manifest.json) 为准。`scripts/run_synchronous_v3.py` 是上一轮旧 scheduler 实验；本轮使用单独的 warmstart 入口。

## 验证记录

启动本轮前：`tests/test_v3_e1_warmstart.py` 与 `tests/test_synchronous_v3.py` 共 **5 passed**，覆盖学习率、文本梯度和已有后续原型检查。实模型启动又校验 A/验证集哈希并复现 Fixed/Extra CE；逐步日志显示 warmup 学习率按新 phase 索引推进。

此次发布完整 CPU 回归 **61 passed（4.65 秒）**，结果记录在 [tests.txt](evidence/2026-09-17/tests.txt)。CPU 测试验证小规模实现性质，不等价于 H20 上所有训练阶段的收敛验证。

## 实验脚本的使用边界

仓库同步的是实际运行源码。部分历史脚本硬编码本机 `/workspace`、`/cloud/cloud-ssd1`、实验日期和 Python 环境，也依赖未上传的 checkpoint、Parquet、tokenizer 与验证 tensor；它们目前不是通用命令行工具。新机器先准备本地依赖并调整路径；新运行需新目录、新 plan 和独立预算。

旧 10B 配置、诊断与 Controller 代码保留用于复核，不表示当前正在使用那些策略。当前 E1 完成前不自动启动 E2/E3。历史仓库的大型模型、运行日志目录和语料继续忽略，发布的证据文件是固定时间点的轻量副本。
