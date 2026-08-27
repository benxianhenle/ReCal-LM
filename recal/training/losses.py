import torch
import torch.nn.functional as F


def language_modeling_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def cosine_state_loss(loop_state: torch.Tensor, full_state: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(loop_state, full_state.detach(), dim=-1).mean()


def logit_kd_loss(loop_logits: torch.Tensor, full_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(loop_logits.float(), dim=-1),
        F.softmax(full_logits.detach().float(), dim=-1),
        reduction="batchmean",
    ) / max(loop_logits.size(1), 1)

