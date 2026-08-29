"""Train ReCal-LM or the matched dense baseline from a YAML config.

中文：根据 YAML 配置训练 ReCal-LM 或匹配的稠密 baseline。"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recal.data.dataset import PackedTextDataset, RandomTokenDataset, StreamingJsonlDataset
from recal.data.tokenizer import load_tokenizer
from recal.model import BaselineLM, ReCalLM
from recal.model.layers import count_parameters
from recal.training.checkpoint import load_checkpoint, save_checkpoint
from recal.training.scheduler import cosine_lr, set_optimizer_lr


def parse_count(value: str) -> int:
    """Parse human-friendly counts such as 500M or 3B into integers.

中文：将 500M、3B 等易读计数解析为整数。"""

    text = str(value).strip().replace("_", "").lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


def load_config(path: str | Path) -> dict:
    """Load a YAML model/training configuration.

中文：加载 YAML 模型/训练配置。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_model(config: dict) -> torch.nn.Module:
    """Instantiate the configured ReCal or baseline language model.

中文：实例化配置指定的 ReCal 或 baseline 语言模型。"""

    model_type = config.get("model_type")
    if model_type == "recal":
        return ReCalLM(config)
    if model_type == "baseline":
        return BaselineLM(config)
    raise ValueError(f"Unknown model_type: {model_type}")


def choose_amp_dtype(config: dict, device: torch.device):
    """Select whether autocast is enabled and which dtype it should use.

中文：选择是否启用 autocast，以及应使用的数据类型。"""

    precision = str(config.get("precision", "fp32")).lower()
    if device.type != "cuda":
        return False, torch.float32
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    if precision in {"bf16", "fp16", "float16"}:
        return True, torch.float16
    return False, torch.float32


def load_best_metadata(path: Path) -> dict:
    """Read the best-checkpoint sidecar or return an empty best-loss record.

中文：读取 best checkpoint 旁路元数据；不存在时返回空的最佳损失记录。"""

    if not path.exists():
        return {"loss": float("inf"), "step": 0, "tokens_seen": 0}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"loss": float("inf"), "step": 0, "tokens_seen": 0}


def write_best_metadata(path: Path, metrics: dict) -> None:
    """Persist the subset of metrics needed to resume best-loss tracking.

中文：持久化恢复最佳损失跟踪所需的指标子集。"""

    payload = {
        "step": metrics["step"],
        "loss": metrics["loss"],
        "loss_lm": metrics["loss_lm"],
        "tokens_seen": metrics["tokens_seen"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def mean_metric(value):
    """Convert optional tensors or numbers into Python floats for JSON logs.

中文：将可选张量或数字转换为 JSON 日志可写的 Python float。"""

    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one training run.

中文：解析单次训练运行的命令行选项。"""

    parser = argparse.ArgumentParser(description="Train ReCal-LM or its matched baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--data", default=None, help="UTF-8 text file. If omitted, uses built-in tiny text.")
    parser.add_argument("--tokenizer", default=None, help="Optional HuggingFace tokenizers JSON.")
    parser.add_argument("--output", default="runs/debug", help="Run output directory.")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override.")
    parser.add_argument("--target-tokens", type=parse_count, default=None, help="Stop after this many training tokens, e.g. 500M or 3B.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="Checkpoint path.")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--keep-interval-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--random-data", action="store_true", help="Use random token IDs for pure speed tests.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--dry-run", action="store_true", help="Build the model, print parameter count, and exit.")
    return parser.parse_args()


def main() -> None:
    """Run the complete training loop, including logging and checkpoints.

中文：运行完整训练循环，包括日志和 checkpoint。"""

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    seq_len = args.seq_len or min(128, int(config["context_length"]))
    if seq_len > int(config["context_length"]):
        raise ValueError("seq-len cannot exceed context_length in the config")

    # Model construction happens before device selection so --dry-run reports params cheaply.
    model = make_model(config)
    n_params = count_parameters(model)
    print(f"model={config.get('name', config['model_type'])} params={n_params:,}")
    if args.dry_run:
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype = choose_amp_dtype(config, device)
    model.to(device)
    if args.compile:
        model = torch.compile(model)

    tokenizer = load_tokenizer(args.tokenizer)
    if getattr(tokenizer, "vocab_size", 0) > config["vocab_size"]:
        raise ValueError("Tokenizer vocab is larger than model vocab_size")
    if args.random_data:
        dataset = RandomTokenDataset(config["vocab_size"], seq_len, size=4096)
    elif args.data and Path(args.data).suffix.lower() == ".jsonl":
        dataset = StreamingJsonlDataset(tokenizer, seq_len, args.data, repeat=True)
    else:
        dataset = PackedTextDataset(tokenizer, seq_len, text_path=args.data)
    is_iterable = isinstance(dataset, IterableDataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not is_iterable, drop_last=True)
    data_iter = iter(loader)

    lr = args.lr or float(train_cfg.get("learning_rate", 3e-4))
    betas = tuple(train_cfg.get("betas", [0.9, 0.95]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )
    start_step = 0
    tokens_seen = 0
    if args.resume:
        start_step, checkpoint = load_checkpoint(args.resume, model, optimizer, map_location=device, return_metadata=True)
        tokens_seen = int(checkpoint.get("tokens_seen", 0))

    tokens_per_step = args.batch_size * seq_len * args.grad_accum
    if args.target_tokens is not None:
        # Convert a token budget into a step cap while respecting resumed progress.
        remaining_tokens = max(0, args.target_tokens - tokens_seen)
        target_steps = start_step + math.ceil(remaining_tokens / max(tokens_per_step, 1))
        max_steps = args.steps if args.steps is not None else target_steps
        max_steps = min(max_steps, target_steps)
    else:
        max_steps = args.steps or int(train_cfg.get("max_steps", 1000))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    best_meta_path = output / "checkpoint_best.json"
    best_metadata = load_best_metadata(best_meta_path)
    best_loss = float(best_metadata.get("loss", float("inf")))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    model.train()
    run_tokens_seen = 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    current_step = start_step
    latest_metrics = None

    def save_last(step_value: int) -> None:
        """Write checkpoint_last.pt with current optimizer and run metadata.

中文：用当前优化器状态和运行元数据写入 checkpoint_last.pt。"""

        save_checkpoint(
            output / "checkpoint_last.pt",
            model,
            optimizer,
            step_value,
            config,
            tokens_seen=tokens_seen,
            seed=args.seed,
            best_loss=best_loss,
        )

    def save_best_if_needed(metrics: dict) -> None:
        """Update checkpoint_best.pt only when the logged loss improves.

中文：仅在记录的 loss 改善时更新 checkpoint_best.pt。"""

        nonlocal best_loss
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output / "checkpoint_best.pt",
                model,
                optimizer,
                metrics["step"],
                config,
                tokens_seen=tokens_seen,
                seed=args.seed,
                best_loss=best_loss,
            )
            write_best_metadata(best_meta_path, metrics)

    try:
        for step in range(start_step, max_steps):
            step_losses = []
            for _ in range(args.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(x, labels=y)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
                step_losses.append(out)
                tokens_seen += x.numel()
                run_tokens_seen += x.numel()

            # Optimizer updates are delayed until all accumulation micro-batches finish.
            lr_now = cosine_lr(step, lr, warmup_steps, max_steps)
            set_optimizer_lr(optimizer, lr_now)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            elapsed = max(time.perf_counter() - t0, 1e-6)
            last = step_losses[-1]
            lm_loss = last.get("loss_lm") if last.get("loss_lm") is not None else last["loss"]
            metrics = {
                "step": step + 1,
                "loss": float(last["loss"].detach().cpu()),
                "loss_lm": float(lm_loss.detach().cpu()),
                "loss_state": float(last["loss_state"].detach().cpu()) if last.get("loss_state") is not None else None,
                "loss_kd": float(last["loss_kd"].detach().cpu()) if last.get("loss_kd") is not None else None,
                "loss_drift": mean_metric(last.get("loss_drift")),
                "loss_router": mean_metric(last.get("loss_router")),
                "drift_pred": mean_metric(last.get("drift_pred")),
                "drift_target": mean_metric(last.get("drift_target")),
                "router_expected_loop_steps": mean_metric(last.get("router_expected_loop_steps")),
                "router_selected_loop_steps": last.get("router_selected_loop_steps"),
                "router_calibration_prob": mean_metric(last.get("router_calibration_prob")),
                "lr": lr_now,
                "tokens_seen": tokens_seen,
                "target_tokens": args.target_tokens,
                "tokens_per_second": run_tokens_seen / elapsed,
            }
            current_step = step + 1
            latest_metrics = metrics
            print(json.dumps(metrics, ensure_ascii=True))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

            if (step + 1) % args.save_interval == 0:
                save_last(step + 1)
                save_best_if_needed(metrics)
                if args.keep_interval_checkpoints:
                    # Interval snapshots are opt-in; default retention is last plus best.
                    save_checkpoint(
                        output / f"checkpoint_{step + 1}.pt",
                        model,
                        optimizer,
                        step + 1,
                        config,
                        tokens_seen=tokens_seen,
                        seed=args.seed,
                        best_loss=best_loss,
                    )
    except KeyboardInterrupt:
        if current_step > start_step:
            save_last(current_step)
            if latest_metrics is not None:
                save_best_if_needed(latest_metrics)
            print(f"Interrupted; saved checkpoint_last.pt at step {current_step}", flush=True)
        raise

    save_last(max_steps)
    if latest_metrics is not None:
        save_best_if_needed(latest_metrics)


if __name__ == "__main__":
    main()
