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


def normalized_state_mse(loop_state: torch.Tensor, target_state: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Measure state error after normalizing every hidden vector.

    Chinese: 先按每个 hidden vector 的 L2 范数归一化，再计算 MSE，避免幅值漂移主导监督。
    """

    loop_norm = loop_state / loop_state.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = target_state.detach() / target_state.detach().norm(dim=-1, keepdim=True).clamp_min(eps)
    return F.mse_loss(loop_norm, target_norm)


def state_distillation_loss(
    loop_state: torch.Tensor,
    target_state: torch.Tensor,
    mse_weight: float = 0.25,
) -> torch.Tensor:
    """Combine cosine and normalized-MSE state supervision.

    Chinese: 组合方向一致性和归一化 MSE；target 在此函数内保持 detach。
    """

    return cosine_state_loss(loop_state, target_state) + mse_weight * normalized_state_mse(loop_state, target_state)


def logit_kd_loss(loop_logits: torch.Tensor, full_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute KL distillation from recurrent logits to full-calibration logits.

中文：计算循环 logits 向完整校准 logits 对齐的 KL 蒸馏损失。"""

    temperature = max(float(temperature), 1e-4)
    return (temperature * temperature) * F.kl_div(
        F.log_softmax(loop_logits.float() / temperature, dim=-1),
        F.softmax(full_logits.detach().float() / temperature, dim=-1),
        reduction="batchmean",
    ) / max(loop_logits.size(1), 1)
