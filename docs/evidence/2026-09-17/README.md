# 2026-09-17 真实实验快照

从本机训练报告与逐步日志直接导出。快照时间及进度以 [snapshot.json](snapshot.json) 为准，GitHub 文件不会随后台训练自动变化。

- [A 基线](a-baseline.json)：8 个 Fixed、24 个 Extra 验证批次的 CE 与状态方差。
- [上一轮 E1](previous-e1-result.json)：继承旧 LR 进度的已结束实验，不是本轮结果。
- [动态路径对比](dynamic-v2-scoreboard.csv)：固定深度、真实混合路径与 cached oracle。
- [本轮计划](v3-e1-plan.json)、[模型与实现来源](v3-e1-manifest.json)、[逐步训练指标](v3-e1-training.csv)。
- [清理汇总](cleanup-summary.json)：模型存储回收记录。

JSON 中的绝对路径为执行环境记录；权重、验证 tensor、原始语料和归档包仅存放在本地。CSV 中 batch_sha256 用于批次对照，不包含原文。训练 batch CE 不能替代验证 CE。
