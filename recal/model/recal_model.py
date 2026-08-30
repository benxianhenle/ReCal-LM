"""ReCal-LM model with trainable attention and an EMA attention teacher.

Chinese: ReCal-LM 模型：Student attention 正常训练，EMA attention teacher 仅提供稳定监督目标。
"""

import copy
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from recal.training.losses import logit_kd_loss, state_distillation_loss

from .layers import RMSNorm, TransformerBlock, TransformerConfig, init_weights


class RouterExecutor(nn.Module):
    """Predict recurrent loop depth and whether calibration is needed.

    Chinese: 预测循环更新深度，并判断是否需要重新校准。
    """

    def __init__(self, hidden_size: int, hidden_dim: int, num_loop_choices: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.loop_head = nn.Linear(hidden_dim, num_loop_choices, bias=False)
        self.calibration_head = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, state: torch.Tensor) -> dict:
        summary = torch.cat([state.mean(dim=1), state[:, -1]], dim=-1)
        hidden = self.net(summary)
        loop_logits = self.loop_head(hidden)
        calibration_logit = self.calibration_head(hidden).squeeze(-1)
        return {
            "loop_logits": loop_logits,
            "loop_probs": F.softmax(loop_logits, dim=-1),
            "calibration_logit": calibration_logit,
            "calibration_prob": torch.sigmoid(calibration_logit),
        }


class DriftEstimator(nn.Module):
    """Estimate the EMA-smoothed recurrent drift for each token state.

    Chinese: 预测每个 token 状态相对 EMA teacher 的平滑漂移累积值。
    """

    def __init__(self, hidden_size: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ReCalLM(nn.Module):
    """Language model with a recurrent student and detached EMA attention teacher.

    Chinese: 语言模型由可训练的 attention/R/decoder 构成；EMA teacher 只在训练时产生 stop-gradient 目标。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loop_choices = [int(item) for item in config.get("loop_choices", [1, 2, 4, 8])]
        layer_config = TransformerConfig(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            dropout=config.get("dropout", 0.0),
            tie_embeddings=config.get("tie_embeddings", True),
        )
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.drop = nn.Dropout(config.get("dropout", 0.0))
        self.front = nn.ModuleList([TransformerBlock(layer_config) for _ in range(config["front_layers"])])
        self.recurrent = nn.ModuleList([TransformerBlock(layer_config) for _ in range(config["recurrent_layers"])])
        self.back = nn.ModuleList([TransformerBlock(layer_config) for _ in range(config["back_layers"])])
        self.state_input = nn.Linear(config["hidden_size"], config["hidden_size"], bias=False)
        self.state_norm = RMSNorm(config["hidden_size"])
        self.final_norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        router_hidden = int(config.get("router_hidden_size", max(config["hidden_size"] // 2, 64)))
        drift_hidden = int(config.get("drift_hidden_size", max(config["hidden_size"] // 2, 64)))
        self.router_executor = RouterExecutor(config["hidden_size"], router_hidden, len(self.loop_choices))
        self.drift_estimator = DriftEstimator(config["hidden_size"], drift_hidden)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(init_weights)

        self.ema_teacher_enabled = bool(config.get("ema_teacher", True))
        if self.ema_teacher_enabled:
            self.teacher_embed_tokens = copy.deepcopy(self.embed_tokens)
            self.teacher_front = copy.deepcopy(self.front)
            self.teacher_state_norm = copy.deepcopy(self.state_norm)
            self._freeze_teacher()

    def teacher_modules(self) -> tuple[nn.Module, ...]:
        """Return EMA teacher modules, or no modules when teacher use is disabled.

        Chinese: 返回 EMA teacher 模块；禁用 teacher 时返回空元组。
        """

        if not self.ema_teacher_enabled:
            return ()
        return (self.teacher_embed_tokens, self.teacher_front, self.teacher_state_norm)

    def _freeze_teacher(self) -> None:
        """Keep EMA teacher parameters out of autograd and dropout.

        Chinese: EMA teacher 不接收梯度，且始终关闭 dropout。
        """

        for module in self.teacher_modules():
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        """Set student mode while keeping teacher targets deterministic.

        Chinese: 设置 student 的训练模式，但 EMA teacher 始终处于 eval 模式。
        """

        super().train(mode)
        if self.ema_teacher_enabled:
            self._freeze_teacher()
        return self

    @torch.no_grad()
    def initialize_ema_teacher(self) -> None:
        """Reset teacher values from student attention values.

        Chinese: 用当前 student attention 的权重重新初始化 EMA teacher。
        """

        if not self.ema_teacher_enabled:
            return
        self.teacher_embed_tokens.load_state_dict(self.embed_tokens.state_dict())
        self.teacher_front.load_state_dict(self.front.state_dict())
        self.teacher_state_norm.load_state_dict(self.state_norm.state_dict())
        self._freeze_teacher()

    @torch.no_grad()
    def update_ema_teacher(self, decay: float) -> None:
        """EMA-update teacher only after a successful optimizer step.

        Chinese: 仅在 optimizer.step() 后调用，以 EMA 更新 teacher，而不是通过 loss 更新。
        """

        if not self.ema_teacher_enabled:
            return
        decay = min(max(float(decay), 0.0), 1.0)
        student_modules = (self.embed_tokens, self.front, self.state_norm)
        for teacher_module, student_module in zip(self.teacher_modules(), student_modules):
            for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
                teacher_param.lerp_(student_param.detach(), 1.0 - decay)
            for teacher_buffer, student_buffer in zip(teacher_module.buffers(), student_module.buffers()):
                teacher_buffer.copy_(student_buffer)
        self._freeze_teacher()

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """Return student-only tensors for deployment.

        Chinese: 导出不含训练期 EMA teacher 的权重，避免推理参数重复。
        """

        return {key: value for key, value in self.state_dict().items() if not key.startswith("teacher_")}

    def _run_blocks(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block in blocks:
            x = block(x, position_ids)
        return x

    def student_attention(self, input_ids: torch.Tensor) -> dict:
        """Run trainable attention to construct the initial recurrent state.

        Chinese: 使用可训练的 student attention 构造循环体的初始状态。
        """

        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.drop(self.embed_tokens(input_ids))
        return {"state": self.state_norm(self._run_blocks(x, self.front, position_ids)), "position_ids": position_ids}

    def teacher_attention(self, input_ids: torch.Tensor) -> dict:
        """Produce detached multi-depth targets from EMA attention.

        Chinese: 从 EMA attention 提取多层 stop-gradient 状态目标。
        """

        if not self.ema_teacher_enabled:
            raise RuntimeError("EMA teacher is disabled; ReCal training requires ema_teacher: true.")
        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        requested = sorted({int(item) for item in self.config.get("teacher_target_layers", [len(self.teacher_front)])})
        requested = [item for item in requested if 1 <= item <= len(self.teacher_front)] or [len(self.teacher_front)]
        with torch.no_grad():
            x = self.teacher_embed_tokens(input_ids)
            targets = []
            for layer_index, block in enumerate(self.teacher_front, start=1):
                x = block(x, position_ids)
                if layer_index in requested:
                    targets.append(self.teacher_state_norm(x).detach())
        return {"state": targets[-1], "targets": targets}

    def full_calibration(self, input_ids: torch.Tensor) -> dict:
        """Run student attention, recurrent block, and decoder for CE training.

        Chinese: 执行 student attention、R 循环体和 decoder；CE 梯度经过这条路径回传。
        """

        attention = self.student_attention(input_ids)
        state = self.state_norm(self._run_blocks(attention["state"], self.recurrent, attention["position_ids"]))
        hidden = self._run_blocks(state, self.back, attention["position_ids"])
        hidden = self.final_norm(hidden)
        return {
            "logits": self.lm_head(hidden),
            "state": state,
            "attention_state": attention["state"],
            "position_ids": attention["position_ids"],
        }

    def recurrent_step(
        self,
        state: torch.Tensor,
        new_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_delta = self.state_input(self.embed_tokens(new_token_ids))
        state = self.state_norm(state + token_delta)
        return self.state_norm(self._run_blocks(state, self.recurrent, position_ids))

    def decode_from_state(self, state: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(self._run_blocks(state, self.back, position_ids)))

    def _sample_loop_steps(self) -> int:
        probs = self.config.get("loop_probs")
        if probs is None:
            return random.choice(self.loop_choices)
        return random.choices(self.loop_choices, weights=probs, k=1)[0]

    def _route_targets(self, observed_drift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create router targets from detached teacher-student drift.

        Chinese: 使用 teacher-student 漂移构造 router 监督，而不是使用手写相邻状态差。
        """

        thresholds = self.config.get("router_drift_thresholds", [0.10, 0.20, 0.35])
        loop_target = torch.zeros(observed_drift.size(0), dtype=torch.long, device=observed_drift.device)
        for threshold in thresholds[: max(len(self.loop_choices) - 1, 0)]:
            loop_target = loop_target + (observed_drift > float(threshold)).long()
        loop_target = loop_target.clamp(max=len(self.loop_choices) - 1)
        calibration_target = (observed_drift > float(self.config.get("drift_threshold", 0.30))).float()
        return loop_target, calibration_target

    def _select_loop_steps(self, route: dict) -> int:
        if self.training and random.random() < float(self.config.get("router_exploration_prob", 0.10)):
            return self._sample_loop_steps()
        if self.training:
            index = torch.multinomial(route["loop_probs"].detach().mean(dim=0), num_samples=1).item()
        else:
            index = route["loop_probs"].detach().mean(dim=0).argmax().item()
        return self.loop_choices[int(index)]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loop_steps: Optional[int] = None,
    ) -> dict:
        """Compute CE, EMA-teacher distillation, drift, consistency, and router losses.

        Chinese: 计算 CE、EMA teacher 蒸馏、漂移、功能一致性和 router 损失。
        """

        full = self.full_calibration(input_ids)
        if self.ema_teacher_enabled:
            teacher = self.teacher_attention(input_ids)
        else:
            # Deployment does not retain teacher tensors; this fallback keeps the student-only API usable.
            detached_state = full["attention_state"].detach()
            teacher = {"state": detached_state, "targets": [detached_state]}
        route = self.router_executor(full["attention_state"].detach())
        loop_values = torch.tensor(self.loop_choices, dtype=route["loop_probs"].dtype, device=input_ids.device)
        out = {
            "logits": full["logits"],
            "state": full["state"],
            "loss_lm": None,
            "loss_state": None,
            "loss_kd": None,
            "loss_drift": None,
            "loss_consistency": None,
            "loss_router": None,
            "router_loop_probs": route["loop_probs"],
            "router_expected_loop_steps": (route["loop_probs"] * loop_values).sum(dim=-1),
            "router_calibration_prob": route["calibration_prob"],
            "router_loop_target": None,
            "router_calibration_target": None,
            "drift_pred": None,
            "drift_target": None,
        }
        if labels is not None:
            out["loss_lm"] = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)), labels.reshape(-1))

        if loop_steps is None:
            loop_steps = self._select_loop_steps(route)
        loop_steps = max(0, min(int(loop_steps), input_ids.size(1) - 1))
        out["router_selected_loop_steps"] = loop_steps
        observed_drift = (1.0 - F.cosine_similarity(full["attention_state"], teacher["state"], dim=-1)).mean(dim=1).detach()

        if loop_steps > 0:
            common_len = input_ids.size(1) - loop_steps
            loop_state = full["attention_state"][:, :common_len]
            drift_accumulator = torch.zeros(loop_state.shape[:2], device=loop_state.device, dtype=loop_state.dtype)
            state_losses, kd_losses, drift_losses, consistency_losses = [], [], [], []
            drift_preds, drift_targets = [], []
            drift_decay = float(self.config.get("drift_accumulator_decay", 0.90))
            for step in range(1, loop_steps + 1):
                token_ids = input_ids[:, step : step + common_len]
                position_ids = torch.arange(step, step + common_len, device=input_ids.device)
                loop_state = self.recurrent_step(loop_state, token_ids, position_ids)
                target_index = min((step * len(teacher["targets"]) - 1) // loop_steps, len(teacher["targets"]) - 1)
                target_state = teacher["targets"][target_index][:, step : step + common_len]
                state_losses.append(
                    state_distillation_loss(loop_state, target_state, float(self.config.get("state_mse_weight", 0.25)))
                )
                instantaneous_drift = 1.0 - F.cosine_similarity(loop_state, target_state, dim=-1)
                drift_accumulator = drift_decay * drift_accumulator + (1.0 - drift_decay) * instantaneous_drift.detach()
                drift_target = drift_accumulator.clamp(0.0, 1.0)
                drift_pred = self.drift_estimator(loop_state)
                drift_losses.append(F.mse_loss(drift_pred, drift_target))
                drift_preds.append(drift_pred.detach().mean())
                drift_targets.append(drift_target.mean())

                loop_logits = self.decode_from_state(loop_state, position_ids)
                with torch.no_grad():
                    teacher_logits = self.decode_from_state(target_state, position_ids)
                    attention_logits = self.decode_from_state(
                        full["attention_state"][:, step : step + common_len], position_ids
                    )
                kd_losses.append(logit_kd_loss(loop_logits, teacher_logits, self.config.get("kd_temperature", 1.0)))
                consistency_losses.append(
                    logit_kd_loss(loop_logits, attention_logits, self.config.get("consistency_temperature", 1.0))
                )

            out["loss_state"] = torch.stack(state_losses).mean()
            out["loss_kd"] = torch.stack(kd_losses).mean()
            out["loss_drift"] = torch.stack(drift_losses).mean()
            out["loss_consistency"] = torch.stack(consistency_losses).mean()
            out["drift_pred"] = torch.stack(drift_preds).mean()
            out["drift_target"] = torch.stack(drift_targets).mean()
            observed_drift = drift_accumulator.mean(dim=1).detach()

        loop_target, calibration_target = self._route_targets(observed_drift)
        out["router_loop_target"] = loop_target
        out["router_calibration_target"] = calibration_target
        out["loss_router"] = F.cross_entropy(route["loop_logits"], loop_target)
        out["loss_router"] = out["loss_router"] + F.binary_cross_entropy_with_logits(
            route["calibration_logit"], calibration_target
        )

        if out["loss_lm"] is not None:
            total = out["loss_lm"] + self.config.get("lambda_router", 0.01) * out["loss_router"]
            for name, coefficient in (
                ("loss_state", "lambda_state"),
                ("loss_kd", "lambda_kd"),
                ("loss_drift", "lambda_drift"),
                ("loss_consistency", "lambda_consistency"),
            ):
                if out[name] is not None:
                    total = total + self.config.get(coefficient, 0.05) * out[name]
            out["loss"] = total
        else:
            out["loss"] = None
        return out
