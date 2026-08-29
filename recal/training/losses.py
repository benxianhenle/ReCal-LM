"""Loss helpers for language modeling, state matching, and distillation.

中文：用于语言建模、状态匹配和蒸馏的损失函数工具。"""

import torch
import torch.nn.functional as F


def language_modeling_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy next-token loss over flattened sequence logits.

中文：在展平后的序列 logits 上计算下一词交叉熵损失。"""

    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def cosine_state_loss(loop_state: torch.Tensor, full_state: torch.Tensor) -> torch.Tensor:
    """Measure recurrent-state mismatch against detached full-calibration states.

中文：衡量循环状态与 detach 后完整校准状态之间的不匹配。"""

    return 1.0 - F.cosine_similarity(loop_state, full_state.detach(), dim=-1).mean()


def logit_kd_loss(loop_logits: torch.Tensor, full_logits: torch.Tensor) -> torch.Tensor:
    """Compute KL distillation from recurrent logits to full-calibration logits.

中文：计算循环 logits 向完整校准 logits 对齐的 KL 蒸馏损失。"""

    return F.kl_div(
        F.log_softmax(loop_logits.float(), dim=-1),
        F.softmax(full_logits.detach().float(), dim=-1),
        reduction="batchmean",
    ) / max(loop_logits.size(1), 1)
