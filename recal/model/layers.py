"""Shared Transformer building blocks used by ReCal-LM and the baseline.

中文：ReCal-LM 和 baseline 共用的 Transformer 基础组件。"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Minimal configuration needed to construct decoder-only Transformer blocks.

中文：构建 decoder-only Transformer 块所需的最小配置。"""

    vocab_size: int
    context_length: int
    hidden_size: int
    num_heads: int
    ffn_dim: int
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden dimension used by attention.

中文：返回注意力中每个 head 使用的隐藏维度。"""

        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    """Root mean square normalization without mean centering.

中文：不进行均值中心化的 RMS 归一化层。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Create a learnable RMSNorm scale for the final tensor dimension.

中文：为张量最后一个维度创建可学习的 RMSNorm 缩放参数。"""

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize activations over the last dimension and apply the scale.

中文：沿最后一个维度归一化激活值，并应用缩放参数。"""

        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent even/odd channels for rotary position embeddings.

中文：为旋转位置编码旋转相邻的偶数/奇数通道。"""

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding cache for attention heads.

中文：为注意力头预计算的旋转位置编码缓存。"""

    def __init__(self, dim: int, max_position: int, base: float = 10000.0):
        """Build sine and cosine tables up to the configured context length.

中文：按配置的上下文长度构建正弦和余弦表。"""

        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors, optionally at explicit positions.

中文：将 RoPE 应用于 query/key 张量，可选择显式位置。"""

        seq_len = q.size(-2)
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1)
            sin = self.sin_cached.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1)
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused scaled-dot-product kernels.

中文：使用融合 scaled-dot-product kernel 的多头因果自注意力。"""

    def __init__(self, config: TransformerConfig):
        """Create projection layers and a RoPE cache for one attention block.

中文：为一个注意力块创建投影层和 RoPE 缓存。"""

        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run masked self-attention over a batch of hidden states.

中文：对一批隐藏状态执行带 mask 的自注意力。"""

        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, position_ids)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network used inside each Transformer block.

中文：每个 Transformer 块内部使用的 SwiGLU 前馈网络。"""

    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float):
        """Create gate, up, and down projections for the FFN path.

中文：为 FFN 路径创建 gate、up 和 down 投影。"""

        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated FFN transformation.

中文：执行带门控的 FFN 变换。"""

        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(x))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with causal attention and SwiGLU FFN paths.

中文：带因果注意力和 SwiGLU FFN 的 pre-norm decoder 块。"""

    def __init__(self, config: TransformerConfig):
        """Create the attention, feed-forward, normalization, and dropout modules.

中文：创建注意力、前馈、归一化和 dropout 模块。"""

        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_dim, config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply one residual attention step followed by one residual FFN step.

中文：先执行一次残差注意力步骤，再执行一次残差 FFN 步骤。"""

        x = x + self.resid_dropout(self.attn(self.attn_norm(x), position_ids))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


def init_weights(module: nn.Module) -> None:
    """Initialize linear and embedding weights with the project default scale.

中文：使用项目默认尺度初始化线性层和嵌入层权重。"""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def count_parameters(module: nn.Module) -> int:
    """Count all parameters in a module, including frozen parameters.

中文：统计模块内所有参数，包括被冻结的参数。"""

    return sum(p.numel() for p in module.parameters())


def estimate_transformer_flops_per_token(num_layers: int, hidden_size: int, ffn_dim: int) -> int:
    """Estimate dense matmul FLOPs per token for a stack of Transformer layers.

中文：估算一组 Transformer 层每个 token 的稠密矩阵乘法 FLOPs。"""

    attn_proj = 4 * hidden_size * hidden_size
    swiglu = 3 * hidden_size * ffn_dim
    return 2 * num_layers * (attn_proj + swiglu)
