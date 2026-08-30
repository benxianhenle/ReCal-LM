"""Evaluate ReCal-LM or baseline checkpoints on packed validation text.

中文：在打包后的验证文本上评估 ReCal-LM 或 baseline checkpoint。"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recal.data.dataset import PackedTextDataset
from recal.data.tokenizer import load_tokenizer
from recal.model import BaselineLM, ReCalLM
from recal.training.checkpoint import load_checkpoint


def parse_args():
    """Parse evaluation command-line options.

中文：解析评估脚本的命令行选项。"""

    parser = argparse.ArgumentParser(description="Evaluate validation loss and loop drift.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--loop", type=int, default=4)
    return parser.parse_args()


def main():
    """Load a model, run bounded validation batches, and print JSON metrics.

中文：加载模型，运行有限批次验证，并输出 JSON 指标。"""

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ReCalLM(config) if config.get("model_type") == "recal" else BaselineLM(config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.to(device).eval()

    tokenizer = load_tokenizer(args.tokenizer)
    dataset = PackedTextDataset(tokenizer, args.seq_len, text_path=args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    losses = []
    state_losses = []
    kd_losses = []
    drift_losses = []
    consistency_losses = []
    drift_preds = []
    drift_targets = []
    router_expected_loops = []
    router_calibration_probs = []
    with torch.no_grad():
        # ReCal emits auxiliary drift/router metrics; baseline emits only LM loss.
        for idx, (x, y) in enumerate(loader):
            if idx >= args.batches:
                break
            x = x.to(device)
            y = y.to(device)
            out = model(x, labels=y, loop_steps=args.loop) if config.get("model_type") == "recal" else model(x, labels=y)
            lm_loss = out.get("loss_lm") if out.get("loss_lm") is not None else out["loss"]
            losses.append(float(lm_loss.cpu()))
            if out.get("loss_state") is not None:
                state_losses.append(float(out["loss_state"].cpu()))
            if out.get("loss_kd") is not None:
                kd_losses.append(float(out["loss_kd"].cpu()))
            if out.get("loss_drift") is not None:
                drift_losses.append(float(out["loss_drift"].cpu()))
            if out.get("loss_consistency") is not None:
                consistency_losses.append(float(out["loss_consistency"].cpu()))
            if out.get("drift_pred") is not None:
                drift_preds.append(float(out["drift_pred"].cpu()))
            if out.get("drift_target") is not None:
                drift_targets.append(float(out["drift_target"].cpu()))
            if out.get("router_expected_loop_steps") is not None:
                router_expected_loops.append(float(out["router_expected_loop_steps"].float().mean().cpu()))
            if out.get("router_calibration_prob") is not None:
                router_calibration_probs.append(float(out["router_calibration_prob"].float().mean().cpu()))
    result = {
        "loss": sum(losses) / len(losses),
        "perplexity": float(torch.exp(torch.tensor(sum(losses) / len(losses)))),
        "loop": args.loop if config.get("model_type") == "recal" else None,
        "state_error": sum(state_losses) / len(state_losses) if state_losses else None,
        "kd": sum(kd_losses) / len(kd_losses) if kd_losses else None,
        "drift_loss": sum(drift_losses) / len(drift_losses) if drift_losses else None,
        "consistency_loss": sum(consistency_losses) / len(consistency_losses) if consistency_losses else None,
        "drift_pred": sum(drift_preds) / len(drift_preds) if drift_preds else None,
        "drift_target": sum(drift_targets) / len(drift_targets) if drift_targets else None,
        "router_expected_loop_steps": sum(router_expected_loops) / len(router_expected_loops) if router_expected_loops else None,
        "router_calibration_prob": sum(router_calibration_probs) / len(router_calibration_probs) if router_calibration_probs else None,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
