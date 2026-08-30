"""Checkpoint save/load helpers shared by training scripts.

中文：训练脚本共享的 checkpoint 保存和加载工具。"""

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, step: int, config: dict, **metadata) -> None:
    """Save model, optional optimizer, step, config, and extra metadata.

中文：保存模型、可选优化器、步数、配置和额外元数据。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
    }
    payload.update(metadata)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu", return_metadata: bool = False):
    """Load model weights, optionally restore optimizer state, and return the step.

中文：加载模型权重，可选恢复优化器状态，并返回训练步数。"""

    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Pre-EMA checkpoints have no teacher_* tensors. Restore the student and seed teacher from it.
        incompatible = model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("teacher_")]
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("teacher_")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint is incompatible; missing={missing}, unexpected={unexpected}")
        if hasattr(model, "initialize_ema_teacher"):
            model.initialize_ema_teacher()
    if optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"Optimizer state was not restored ({exc}); continuing with a fresh optimizer state.")
    step = int(ckpt.get("step", 0))
    if return_metadata:
        return step, ckpt
    return step
