import torch

from recal.model import BaselineLM, ReCalLM


def tiny_recal_config():
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


def test_baseline_forward():
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
