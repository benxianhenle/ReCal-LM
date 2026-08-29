"""Learning-rate scheduling helpers for the training scripts.

中文：训练脚本使用的学习率调度辅助函数。"""

import math


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Return a warmup-plus-cosine learning rate for the current step.

中文：返回当前 step 对应的 warmup 加 cosine 学习率。"""

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if max_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    """Apply a scalar learning rate to every optimizer parameter group.

中文：将同一个标量学习率应用到优化器的所有参数组。"""

    for group in optimizer.param_groups:
        group["lr"] = lr
