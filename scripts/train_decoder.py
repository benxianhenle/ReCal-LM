"""Train only the ReCal back-end decoder while freezing calibration blocks.

中文：冻结校准相关模块，只训练 ReCal 后端 decoder。"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recal.data.dataset import PackedTextDataset, RandomTokenDataset, StreamingJsonlDataset
from recal.data.tokenizer import load_tokenizer
from recal.model import ReCalLM
from recal.model.layers import count_parameters
from recal.training.checkpoint import load_checkpoint, save_checkpoint
from recal.training.scheduler import cosine_lr, set_optimizer_lr
from scripts.train import choose_amp_dtype, load_config, mean_metric


def parse_args() -> argparse.Namespace:
    """Parse command-line options for decoder-only training.

中文：解析 decoder-only 训练的命令行选项。"""

    parser = argparse.ArgumentParser(description="Train only the ReCal state decoder path.")
    parser.add_argument("--config", required=True, help="Path to a ReCal YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional ReCal checkpoint to initialize from.")
    parser.add_argument("--data", default=None, help="UTF-8 text or JSONL training data.")
    parser.add_argument("--tokenizer", default=None, help="Optional HuggingFace tokenizers JSON.")
    parser.add_argument("--output", default="runs/decoder-only", help="Run output directory.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--random-data", action="store_true")
    return parser.parse_args()


def set_trainable_decoder(model: ReCalLM) -> list[torch.nn.Parameter]:
    """Freeze the model except the back blocks, final norm, and LM head.

中文：冻结模型中除 back blocks、final norm 和 LM head 之外的部分。"""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.back, model.final_norm, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if model.config.get("tie_embeddings", True):
        # lm_head shares this matrix, so this is part of the output decoder.
        model.embed_tokens.weight.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def frozen_state(model: ReCalLM, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen front/recurrent blocks and return detached decoder states.

中文：运行冻结的 front/recurrent 块，并返回 detach 后的 decoder 状态。"""

    _, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device)
    with torch.no_grad():
        x = model.drop(model.embed_tokens(input_ids))
        front_hidden = model._run_blocks(x, model.front, position_ids)
        state = model._run_blocks(front_hidden, model.recurrent, position_ids)
        state = model.state_norm(state).detach()
    return state, position_ids


def main() -> None:
    """Run decoder-only training and retain best/last checkpoints.

中文：运行 decoder-only 训练，并保留 best/last checkpoint。"""

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    if config.get("model_type") != "recal":
        raise ValueError("train_decoder.py only supports ReCal configs.")
    seq_len = args.seq_len or min(128, int(config["context_length"]))
    model = ReCalLM(config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location="cpu")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype = choose_amp_dtype(config, device)
    model.to(device)
    trainable = set_trainable_decoder(model)
    model.train()
    # Keep frozen modules in eval mode so dropout does not perturb cached states.
    model.front.eval()
    model.recurrent.eval()
    model.router_executor.eval()
    model.drift_estimator.eval()

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

    train_cfg = config.get("training", {})
    lr = args.lr or float(train_cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    best_meta_path = output / "checkpoint_best.json"
    best_loss = float("inf")
    tokens_seen = 0
    t0 = time.perf_counter()

    print(
        json.dumps(
            {
                "model": config.get("name", "ReCal-LM"),
                "total_params": count_parameters(model),
                "decoder_trainable_params": sum(parameter.numel() for parameter in trainable),
                "tied_embeddings": bool(config.get("tie_embeddings", True)),
            },
            ensure_ascii=True,
        )
    )

    latest_metrics = None
    for step in range(args.steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            state, position_ids = frozen_state(model, x)
            logits = model.decode_from_state(state, position_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Only decoder parameters are trainable, but AMP/scaler flow mirrors train.py.
        scaler.scale(loss).backward()
        lr_now = cosine_lr(step, lr, int(train_cfg.get("warmup_steps", 100)), args.steps)
        set_optimizer_lr(optimizer, lr_now)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        elapsed = max(time.perf_counter() - t0, 1e-6)
        metrics = {
            "step": step + 1,
            "loss": mean_metric(loss),
            "lr": lr_now,
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_seen / elapsed,
        }
        latest_metrics = metrics
        print(json.dumps(metrics, ensure_ascii=True))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output / "checkpoint_best.pt",
                model,
                optimizer,
                step + 1,
                config,
                tokens_seen=tokens_seen,
                seed=args.seed,
                decoder_only=True,
                best_loss=best_loss,
            )
            best_meta_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                output / "checkpoint_last.pt",
                model,
                optimizer,
                step + 1,
                config,
                tokens_seen=tokens_seen,
                seed=args.seed,
                decoder_only=True,
                best_loss=best_loss,
            )

    if latest_metrics is not None:
        save_checkpoint(
            output / "checkpoint_last.pt",
            model,
            optimizer,
            latest_metrics["step"],
            config,
            tokens_seen=tokens_seen,
            seed=args.seed,
            decoder_only=True,
            best_loss=best_loss,
        )


if __name__ == "__main__":
    main()
