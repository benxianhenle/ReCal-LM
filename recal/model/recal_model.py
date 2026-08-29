"""ReCal-LM model: full calibration plus recurrent hidden-state updates.

中文：ReCal-LM 模型：完整校准加循环隐藏状态更新。"""

import random
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, TransformerConfig, init_weights


class RouterExecutor(nn.Module):
    """Predicts recurrent loop depth and whether recalibration is needed.

中文：预测循环更新深度，并判断是否需要重新校准。"""

    def __init__(self, hidden_size: int, hidden_dim: int, num_loop_choices: int):
        """Create the small router MLP and its loop/calibration heads.

中文：创建小型 router MLP 及其循环深度/校准判断输出头。"""

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
        """Summarize a sequence state and produce routing probabilities.

中文：汇总序列状态并生成路由概率。"""

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
    """Predicts state drift between recurrent updates and full calibration.

中文：预测循环更新状态与完整校准状态之间的漂移。"""

    def __init__(self, hidden_size: int, hidden_dim: int):
        """Create the token-level drift predictor MLP.

中文：创建 token 级状态漂移预测 MLP。"""

        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return a bounded drift estimate for each token state.

中文：为每个 token 状态返回有界漂移估计。"""

        return self.net(state).squeeze(-1)


class ReCalLM(nn.Module):
    """Language model that trains dense calibration and recurrent reuse together.

中文：同时训练稠密校准和循环复用能力的语言模型。"""

    def __init__(self, config: dict):
        """Construct the front, recurrent, back, router, and drift modules.

中文：构建 front、recurrent、back、router 和 drift 模块。"""

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

    def _run_blocks(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run a list of Transformer blocks with shared position IDs.

中文：使用共享位置 ID 依次运行一组 Transformer 块。"""

        for block in blocks:
            x = block(x, position_ids)
        return x

    def full_calibration(self, input_ids: torch.Tensor) -> dict:
        """Run the full front/recurrent/back pass used as the calibration target.

中文：执行完整的 front/recurrent/back 前向过程，作为校准目标。"""

        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.drop(self.embed_tokens(input_ids))
        front_hidden = self._run_blocks(x, self.front, position_ids)
        state = self._run_blocks(front_hidden, self.recurrent, position_ids)
        hidden = self._run_blocks(state, self.back, position_ids)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return {"logits": logits, "state": self.state_norm(state), "hidden": hidden}

    def recurrent_step(
        self,
        state: torch.Tensor,
        new_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Advance a cached state with shifted input tokens and recurrent blocks.

中文：用移位后的输入 token 和 recurrent 块推进缓存状态。"""

        token_delta = self.state_input(self.embed_tokens(new_token_ids))
        state = self.state_norm(state + token_delta)
        state = self._run_blocks(state, self.recurrent, position_ids)
        return self.state_norm(state)

    def decode_from_state(self, state: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Decode logits from an already calibrated or recurrently updated state.

中文：从已校准或经循环更新的状态中解码 logits。"""

        hidden = self._run_blocks(state, self.back, position_ids)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    def _sample_loop_steps(self) -> int:
        """Sample an exploration loop count from configured choices.

中文：从配置的候选循环次数中采样探索值。"""

        probs = self.config.get("loop_probs")
        if probs is None:
            return random.choice(self.loop_choices)
        return random.choices(self.loop_choices, weights=probs, k=1)[0]

    def _route_targets(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build heuristic supervision targets for loop depth and recalibration.

中文：为循环深度和重新校准构造启发式监督目标。"""

        if state.size(1) < 2:
            loop_target = torch.zeros(state.size(0), dtype=torch.long, device=state.device)
            calibration_target = torch.zeros(state.size(0), device=state.device)
            return loop_target, calibration_target
        state_delta = 1.0 - F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1)
        drift_signal = state_delta.mean(dim=1).detach()
        default_thresholds = [0.10, 0.20, 0.35]
        thresholds = self.config.get("router_drift_thresholds", default_thresholds)
        thresholds = thresholds[: max(len(self.loop_choices) - 1, 0)]
        loop_target = torch.zeros(state.size(0), dtype=torch.long, device=state.device)
        for threshold in thresholds:
            loop_target = loop_target + (drift_signal > float(threshold)).long()
        loop_target = loop_target.clamp(max=len(self.loop_choices) - 1)
        calibration_threshold = float(self.config.get("drift_threshold", 0.30))
        calibration_target = (drift_signal > calibration_threshold).float()
        return loop_target, calibration_target

    def _select_loop_steps(self, route: dict) -> int:
        """Choose the recurrent loop count from router output and exploration.

中文：根据 router 输出和探索策略选择循环次数。"""

        if self.training and random.random() < float(self.config.get("router_exploration_prob", 0.10)):
            return self._sample_loop_steps()
        if self.training:
            idx = torch.multinomial(route["loop_probs"].detach().mean(dim=0), num_samples=1).item()
        else:
            idx = route["loop_probs"].detach().mean(dim=0).argmax().item()
        return self.loop_choices[int(idx)]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loop_steps: Optional[int] = None,
    ) -> dict:
        """Compute LM logits/losses and optional recurrent consistency losses.

中文：计算语言模型 logits/损失，以及可选的循环一致性损失。"""

        full = self.full_calibration(input_ids)
        logits = full["logits"]
        route = self.router_executor(full["state"].detach())
        loop_values = torch.tensor(self.loop_choices, dtype=route["loop_probs"].dtype, device=input_ids.device)
        expected_loops = (route["loop_probs"] * loop_values).sum(dim=-1)
        loop_target, calibration_target = self._route_targets(full["state"])
        loss_router = F.cross_entropy(route["loop_logits"], loop_target)
        loss_router = loss_router + F.binary_cross_entropy_with_logits(
            route["calibration_logit"],
            calibration_target,
        )
        out = {
            "logits": logits,
            "state": full["state"],
            "loss_lm": None,
            "loss_state": None,
            "loss_kd": None,
            "loss_drift": None,
            "loss_router": loss_router,
            "router_loop_probs": route["loop_probs"],
            "router_expected_loop_steps": expected_loops,
            "router_calibration_prob": route["calibration_prob"],
            "router_loop_target": loop_target,
            "router_calibration_target": calibration_target,
            "drift_pred": None,
            "drift_target": None,
        }
        if labels is not None:
            out["loss_lm"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if loop_steps is None:
            loop_steps = self._select_loop_steps(route)
        loop_steps = min(int(loop_steps), input_ids.size(1) - 1)
        out["router_selected_loop_steps"] = loop_steps

        if loop_steps > 0:
            common_len = input_ids.size(1) - loop_steps
            loop_state = full["state"][:, :common_len]
            state_losses = []
            kd_losses = []
            drift_losses = []
            drift_preds = []
            drift_targets = []
            for step in range(1, loop_steps + 1):
                token_ids = input_ids[:, step : step + common_len]
                pos = torch.arange(step, step + common_len, device=input_ids.device)
                loop_state = self.recurrent_step(loop_state, token_ids, pos)
                target_state = full["state"][:, step : step + common_len].detach()
                drift_target = 1.0 - F.cosine_similarity(loop_state, target_state, dim=-1)
                state_losses.append(drift_target.mean())
                drift_target = drift_target.detach().clamp(0.0, 1.0)
                drift_pred = self.drift_estimator(loop_state)
                drift_losses.append(F.mse_loss(drift_pred, drift_target))
                drift_preds.append(drift_pred.detach().mean())
                drift_targets.append(drift_target.mean())

                loop_logits = self.decode_from_state(loop_state, pos)
                target_logits = full["logits"][:, step : step + common_len].detach()
                kd = F.kl_div(
                    F.log_softmax(loop_logits.float(), dim=-1),
                    F.softmax(target_logits.float(), dim=-1),
                    reduction="batchmean",
                ) / common_len
                kd_losses.append(kd)
            out["loss_state"] = torch.stack(state_losses).mean()
            out["loss_kd"] = torch.stack(kd_losses).mean()
            out["loss_drift"] = torch.stack(drift_losses).mean()
            out["drift_pred"] = torch.stack(drift_preds).mean()
            out["drift_target"] = torch.stack(drift_targets).mean()

        total = None
        if out["loss_lm"] is not None:
            total = out["loss_lm"]
            if out["loss_state"] is not None:
                total = total + self.config.get("lambda_state", 0.1) * out["loss_state"]
            if out["loss_kd"] is not None:
                total = total + self.config.get("lambda_kd", 0.5) * out["loss_kd"]
            if out["loss_drift"] is not None:
                total = total + self.config.get("lambda_drift", 0.05) * out["loss_drift"]
            total = total + self.config.get("lambda_router", 0.01) * out["loss_router"]
        out["loss"] = total
        return out
