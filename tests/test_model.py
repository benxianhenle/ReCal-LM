"""Smoke tests for ReCal-LM and baseline forward passes.

中文：ReCal-LM 和 baseline 前向过程的冒烟测试。"""

import torch

from recal.model import BaselineLM, ReCalLM


def tiny_recal_config():
    """Return a tiny ReCal config that exercises all auxiliary loss heads.

中文：返回一个能覆盖所有辅助 loss head 的极小 ReCal 配置。"""

    return {
        "model_type": "recal",
        "vocab_size": 128,
        "context_length": 32,
        "hidden_size": 64,
        "num_heads": 4,
        "ffn_dim": 128,
        "front_layers": 1,
        "recurrent_layers": 1,
        "back_layers": 1,
        "dropout": 0.0,
        "tie_embeddings": True,
        "loop_choices": [1, 2],
        "router_hidden_size": 32,
        "drift_hidden_size": 32,
        "router_drift_thresholds": [0.10],
        "drift_threshold": 0.30,
        "lambda_state": 0.1,
        "lambda_kd": 0.5,
        "lambda_drift": 0.05,
        "lambda_router": 0.01,
    }


def test_recal_forward_with_loop_losses():
    """Verify ReCal forward output shapes and scalar auxiliary losses.

中文：验证 ReCal 前向输出形状和标量辅助损失。"""

    model = ReCalLM(tiny_recal_config())
    x = torch.randint(4, 128, (2, 16))
    y = torch.randint(4, 128, (2, 16))
    out = model(x, labels=y, loop_steps=2)
    assert out["logits"].shape == (2, 16, 128)
    assert out["loss"].ndim == 0
    assert out["loss_state"].ndim == 0
    assert out["loss_kd"].ndim == 0
    assert out["loss_drift"].ndim == 0
    assert out["loss_router"].ndim == 0
    assert out["router_loop_probs"].shape == (2, 2)
    assert out["router_selected_loop_steps"] in {1, 2}
    assert out["drift_pred"].ndim == 0
    assert out["drift_target"].ndim == 0


def test_recal_full_loss_trains_all_major_modules():
    """Verify the full ReCal objective reaches every major trainable module."""

    model = ReCalLM(tiny_recal_config())
    assert all(parameter.requires_grad for parameter in model.parameters())

    x = torch.randint(4, 128, (2, 16))
    y = torch.randint(4, 128, (2, 16))
    out = model(x, labels=y, loop_steps=2)
    out["loss"].backward()

    gradient_checks = {
        "front_attention": model.front[0].attn.qkv.weight.grad,
        "recurrent_attention": model.recurrent[0].attn.qkv.weight.grad,
        "back_attention": model.back[0].attn.qkv.weight.grad,
        "state_input": model.state_input.weight.grad,
        "router_loop_head": model.router_executor.loop_head.weight.grad,
        "router_calibration_head": model.router_executor.calibration_head.weight.grad,
        "drift_estimator": model.drift_estimator.net[1].weight.grad,
        "lm_head": model.lm_head.weight.grad,
    }
    for name, gradient in gradient_checks.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_baseline_forward():
    """Verify baseline forward output shapes and scalar LM loss.

中文：验证 baseline 前向输出形状和标量语言模型损失。"""

    config = {
        "model_type": "baseline",
        "vocab_size": 128,
        "context_length": 32,
        "hidden_size": 64,
        "num_heads": 4,
        "ffn_dim": 128,
        "num_layers": 3,
        "dropout": 0.0,
        "tie_embeddings": True,
    }
    model = BaselineLM(config)
    x = torch.randint(4, 128, (2, 16))
    y = torch.randint(4, 128, (2, 16))
    out = model(x, labels=y)
    assert out["logits"].shape == (2, 16, 128)
    assert out["loss"].ndim == 0
