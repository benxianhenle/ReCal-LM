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
        "selection_metric": metrics.get("selection_metric", "loss"),
        "selection_loss": metrics.get("selection_loss", metrics["loss"]),
        "val_loss_lm": metrics.get("val_loss_lm"),
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


def unique_parameter_counts(module: torch.nn.Module) -> dict:
    """Count unique parameters in a module, split by trainability."""

    seen: set[int] = set()
    total = 0
    trainable = 0
    for parameter in module.parameters():
        ident = id(parameter)
        if ident in seen:
            continue
        seen.add(ident)
        n_values = parameter.numel()
        total += n_values
        if parameter.requires_grad:
            trainable += n_values
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def module_trainability_summary(model: torch.nn.Module) -> dict:
    """Report whole-model and major-module parameter coverage."""

    model_counts = unique_parameter_counts(model)
    module_names = [
        "embed_tokens",
        "front",
        "recurrent",
        "back",
        "state_input",
        "state_norm",
        "final_norm",
        "lm_head",
        "router_executor",
        "drift_estimator",
        "blocks",
    ]
    modules = {}
    for name in module_names:
        module = getattr(model, name, None)
        if module is not None:
            modules[name] = unique_parameter_counts(module)
    return {
        **model_counts,
        "all_parameters_trainable": model_counts["frozen"] == 0,
        "modules": modules,
    }


def make_dataset(tokenizer, vocab_size: int, seq_len: int, data_path: str | None, random_data: bool, repeat_jsonl: bool):
    """Create the project dataset matching the requested source."""

    if random_data:
        return RandomTokenDataset(vocab_size, seq_len, size=4096)
    if data_path and Path(data_path).suffix.lower() == ".jsonl":
        return StreamingJsonlDataset(tokenizer, seq_len, data_path, repeat=repeat_jsonl)
    return PackedTextDataset(tokenizer, seq_len, text_path=data_path)


def make_loader(dataset, batch_size: int):
    """Create a DataLoader with shuffle disabled for iterable streams."""

    is_iterable = isinstance(dataset, IterableDataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not is_iterable, drop_last=True)


def validation_metrics(
    model: torch.nn.Module,
    loader,
    config: dict,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype,
    max_batches: int,
    loop_steps: int | None,
) -> dict:
    """Run bounded held-out validation and return LM-first metrics."""

    was_training = model.training
    model.eval()
    losses = []
    state_losses = []
    kd_losses = []
    drift_losses = []
    router_expected_loops = []
    router_calibration_probs = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if config.get("model_type") == "recal":
                    out = model(x, labels=y, loop_steps=loop_steps)
                else:
                    out = model(x, labels=y)
            lm_loss = out.get("loss_lm") if out.get("loss_lm") is not None else out["loss"]
            losses.append(float(lm_loss.detach().cpu()))
            if out.get("loss_state") is not None:
                state_losses.append(float(out["loss_state"].detach().cpu()))
            if out.get("loss_kd") is not None:
                kd_losses.append(float(out["loss_kd"].detach().cpu()))
            if out.get("loss_drift") is not None:
                drift_losses.append(mean_metric(out.get("loss_drift")))
            if out.get("router_expected_loop_steps") is not None:
                router_expected_loops.append(mean_metric(out.get("router_expected_loop_steps")))
            if out.get("router_calibration_prob") is not None:
                router_calibration_probs.append(mean_metric(out.get("router_calibration_prob")))
    if was_training:
        model.train()
    if not losses:
        raise ValueError("Validation produced no batches; check --val-data, --seq-len, and --batch-size.")
    val_loss = sum(losses) / len(losses)
    return {
        "val_loss_lm": val_loss,
        "val_perplexity": float(torch.exp(torch.tensor(val_loss))),
        "val_loss_state": sum(state_losses) / len(state_losses) if state_losses else None,
        "val_loss_kd": sum(kd_losses) / len(kd_losses) if kd_losses else None,
        "val_loss_drift": sum(drift_losses) / len(drift_losses) if drift_losses else None,
        "val_router_expected_loop_steps": sum(router_expected_loops) / len(router_expected_loops) if router_expected_loops else None,
        "val_router_calibration_prob": sum(router_calibration_probs) / len(router_calibration_probs) if router_calibration_probs else None,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one training run.

中文：解析单次训练运行的命令行选项。"""

    parser = argparse.ArgumentParser(description="Train ReCal-LM or its matched baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--data", default=None, help="UTF-8 text file. If omitted, uses built-in tiny text.")
    parser.add_argument("--val-data", default=None, help="Optional held-out UTF-8 text or JSONL validation data.")
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
    parser.add_argument("--val-interval", type=int, default=0, help="Run validation every N steps; defaults to save interval when --val-data is set.")
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--val-loop", type=int, default=None, help="Fixed ReCal loop count during validation.")
    parser.add_argument("--keep-interval-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--random-data", action="store_true", help="Use random token IDs for pure speed tests.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--allow-frozen-params", action="store_true", help="Allow training even if some parameters are frozen.")
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
    trainability = module_trainability_summary(model)
    print(json.dumps({"trainability": trainability}, ensure_ascii=True))
    if trainability["frozen"] and not args.allow_frozen_params:
        raise RuntimeError(
            "Some model parameters are frozen. train.py is intended for full-parameter training; "
            "pass --allow-frozen-params only for an intentional partial-training run."
        )
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
    dataset = make_dataset(tokenizer, int(config["vocab_size"]), seq_len, args.data, args.random_data, repeat_jsonl=True)
    loader = make_loader(dataset, args.batch_size)
    data_iter = iter(loader)
    val_interval = args.val_interval
    val_loader = None
    if args.val_data:
        if val_interval <= 0:
            val_interval = args.save_interval
        val_dataset = make_dataset(tokenizer, int(config["vocab_size"]), seq_len, args.val_data, False, repeat_jsonl=False)
        val_loader = make_loader(val_dataset, args.batch_size)
    elif val_interval > 0:
        raise ValueError("--val-interval requires --val-data")

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
    best_loss = float(best_metadata.get("selection_loss", best_metadata.get("loss", float("inf"))))
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
        selection_metric = "val_loss_lm" if metrics.get("val_loss_lm") is not None else "loss"
        selection_loss = float(metrics[selection_metric])
        metrics["selection_metric"] = selection_metric
        metrics["selection_loss"] = selection_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            save_checkpoint(
                output / "checkpoint_best.pt",
                model,
                optimizer,
                metrics["step"],
                config,
                tokens_seen=tokens_seen,
                seed=args.seed,
                best_loss=best_loss,
                selection_metric=selection_metric,
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
            ran_validation = False
            if val_loader is not None and (step + 1) % val_interval == 0:
                metrics.update(
                    validation_metrics(
                        model,
                        val_loader,
                        config,
                        device,
                        amp_enabled,
                        amp_dtype,
                        args.val_batches,
                        args.val_loop,
                    )
                )
                ran_validation = True
            current_step = step + 1
            latest_metrics = metrics
            print(json.dumps(metrics, ensure_ascii=True))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

            if (step + 1) % args.save_interval == 0:
                save_last(step + 1)
                if val_loader is None or ran_validation:
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
            elif ran_validation:
                save_best_if_needed(metrics)
    except KeyboardInterrupt:
        if current_step > start_step:
            save_last(current_step)
            if latest_metrics is not None and (val_loader is None or latest_metrics.get("val_loss_lm") is not None):
                save_best_if_needed(latest_metrics)
            print(f"Interrupted; saved checkpoint_last.pt at step {current_step}", flush=True)
        raise

    save_last(max_steps)
    if latest_metrics is not None and (val_loader is None or latest_metrics.get("val_loss_lm") is not None):
        save_best_if_needed(latest_metrics)


if __name__ == "__main__":
    main()
