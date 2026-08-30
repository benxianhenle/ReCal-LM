"""Local WebUI server for real ReCal-LM checkpoint inspection and generation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recal.data.tokenizer import load_tokenizer
from recal.model import BaselineLM, ReCalLM
from recal.model.layers import count_parameters
from recal.training.checkpoint import load_checkpoint
from scripts.train import choose_amp_dtype, module_trainability_summary


class ModelSession:
    """Holds the currently loaded model and tokenizer."""

    def __init__(self) -> None:
        self.config_path: Path | None = None
        self.checkpoint_path: Path | None = None
        self.tokenizer_path: Path | None = None
        self.config: dict[str, Any] | None = None
        self.model: torch.nn.Module | None = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.amp_enabled = False
        self.amp_dtype = torch.float32
        self.step = 0

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.config is not None and self.tokenizer is not None

    def load(self, config_path: Path, checkpoint_path: Path | None, tokenizer_path: Path | None, device_name: str) -> dict:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        model_type = config.get("model_type")
        if model_type == "recal":
            model: torch.nn.Module = ReCalLM(config)
        elif model_type == "baseline":
            model = BaselineLM(config)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        amp_enabled, amp_dtype = choose_amp_dtype(config, device)

        step = 0
        if checkpoint_path is not None:
            step = load_checkpoint(checkpoint_path, model, map_location="cpu")
        model.to(device).eval()

        tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)
        if getattr(tokenizer, "vocab_size", 0) > int(config["vocab_size"]):
            raise ValueError("Tokenizer vocab is larger than model vocab_size")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.step = step
        return self.status()

    def status(self) -> dict:
        if not self.loaded:
            return {"loaded": False}
        assert self.model is not None
        assert self.config is not None
        return {
            "loaded": True,
            "model": self.config.get("name", self.config.get("model_type")),
            "model_type": self.config.get("model_type"),
            "config": rel(self.config_path),
            "checkpoint": rel(self.checkpoint_path),
            "tokenizer": rel(self.tokenizer_path),
            "device": str(self.device),
            "precision": self.config.get("precision", "fp32"),
            "step": self.step,
            "parameters": count_parameters(self.model),
            "trainability": module_trainability_summary(self.model),
            "context_length": self.config.get("context_length"),
            "checkpoint_loaded": self.checkpoint_path is not None,
        }

    def _encode_window(self, text: str, reserve_new_tokens: int = 0) -> list[int]:
        assert self.config is not None
        assert self.tokenizer is not None
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        max_input = max(2, int(self.config["context_length"]) - reserve_new_tokens)
        return token_ids[-max_input:]

    def inspect(self, text: str, loop_steps: int | None) -> dict:
        self._require_loaded()
        assert self.model is not None
        assert self.config is not None
        assert self.tokenizer is not None
        token_ids = self._encode_window(text)
        if len(token_ids) < 2:
            raise ValueError("Need at least two tokens for inspection.")

        x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([token_ids[1:]], dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            if self.config.get("model_type") == "recal":
                out = self.model(x, labels=y, loop_steps=loop_steps)
            else:
                out = self.model(x, labels=y)

        lm_loss = out.get("loss_lm") if out.get("loss_lm") is not None else out["loss"]
        last_logits = out["logits"][0, -1].float()
        probs = torch.softmax(last_logits, dim=-1)
        k = min(8, probs.numel())
        top_probs, top_ids = torch.topk(probs, k=k)
        return {
            "tokens": len(token_ids),
            "loss_lm": scalar(lm_loss),
            "perplexity": safe_exp(scalar(lm_loss)),
            "loss_total": scalar(out.get("loss")),
            "loss_state": scalar(out.get("loss_state")),
            "loss_kd": scalar(out.get("loss_kd")),
            "loss_drift": scalar(out.get("loss_drift")),
            "loss_router": scalar(out.get("loss_router")),
            "drift_pred": scalar(out.get("drift_pred")),
            "drift_target": scalar(out.get("drift_target")),
            "router_expected_loop_steps": scalar(out.get("router_expected_loop_steps")),
            "router_selected_loop_steps": out.get("router_selected_loop_steps"),
            "router_calibration_prob": scalar(out.get("router_calibration_prob")),
            "top_next_tokens": [
                {
                    "id": int(token_id),
                    "probability": float(prob),
                    "text": self.tokenizer.decode([int(token_id)]),
                }
                for token_id, prob in zip(top_ids.cpu(), top_probs.cpu())
            ],
        }

    def generate(self, text: str, max_new_tokens: int, temperature: float, top_k: int, loop_steps: int | None) -> dict:
        self._require_loaded()
        assert self.model is not None
        assert self.config is not None
        assert self.tokenizer is not None
        max_new_tokens = max(1, min(int(max_new_tokens), 128))
        temperature = max(float(temperature), 1e-4)
        top_k = max(0, int(top_k))
        token_ids = self._encode_window(text, reserve_new_tokens=max_new_tokens)
        generated: list[int] = []
        last_metrics: dict[str, Any] = {}

        for _ in range(max_new_tokens):
            x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                if self.config.get("model_type") == "recal":
                    out = self.model(x, loop_steps=loop_steps)
                else:
                    out = self.model(x)
            logits = out["logits"][0, -1].float() / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
                probs = torch.softmax(values, dim=-1)
                next_id = int(indices[torch.multinomial(probs, num_samples=1)].item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            token_ids.append(next_id)
            token_ids = token_ids[-int(self.config["context_length"]) :]
            last_metrics = {
                "router_expected_loop_steps": scalar(out.get("router_expected_loop_steps")),
                "router_selected_loop_steps": out.get("router_selected_loop_steps"),
                "router_calibration_prob": scalar(out.get("router_calibration_prob")),
                "drift_pred": scalar(out.get("drift_pred")),
                "drift_target": scalar(out.get("drift_target")),
            }
            if next_id == getattr(self.tokenizer, "eos_token_id", None):
                break

        return {
            "generated_token_count": len(generated),
            "text": self.tokenizer.decode(token_ids),
            "new_text": self.tokenizer.decode(generated),
            "last_step_metrics": last_metrics,
        }

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError("No model loaded. Load config/checkpoint first.")


SESSION = ModelSession()


def scalar(value) -> float | int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().cpu())
        return float(value.detach().float().mean().cpu())
    if isinstance(value, (int, float)):
        return value
    return None


def safe_exp(value: float | int | None) -> float | None:
    if value is None:
        return None
    if value > 50:
        return float("inf")
    return float(math.exp(value))


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def project_path(value: str | None, *, required: bool) -> Path | None:
    if not value:
        if required:
            raise ValueError("Missing required path.")
        return None
    raw = Path(value)
    path = raw if raw.is_absolute() else ROOT / raw
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside project root: {value}") from exc
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "ReCalWebUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(ROOT / "webui" / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            self._json(SESSION.status())
        else:
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)
            if parsed.path == "/api/load":
                result = SESSION.load(
                    project_path(payload.get("config"), required=True),
                    project_path(payload.get("checkpoint"), required=False),
                    project_path(payload.get("tokenizer"), required=False),
                    str(payload.get("device", "auto")),
                )
                self._json(result)
            elif parsed.path == "/api/inspect":
                self._json(SESSION.inspect(str(payload.get("text", "")), optional_int(payload.get("loop_steps"))))
            elif parsed.path == "/api/generate":
                self._json(
                    SESSION.generate(
                        str(payload.get("text", "")),
                        int(payload.get("max_new_tokens", 32)),
                        float(payload.get("temperature", 0.8)),
                        int(payload.get("top_k", 40)),
                        optional_int(payload.get("loop_steps")),
                    )
                )
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def optional_int(value) -> int | None:
    if value in (None, "", "auto"):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ReCal-LM WebUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ReCal-LM WebUI: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
