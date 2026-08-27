def run(statement, filename=None, sort=-1):
    raise RuntimeError("This is the ReCal-LM profile script, not the stdlib profile module.")


def runctx(statement, globals=None, locals=None, filename=None, sort=-1):
    raise RuntimeError("This is the ReCal-LM profile script, not the stdlib profile module.")


if __name__ != "profile":
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    import torch
    import yaml

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from recal.model import BaselineLM, ReCalLM
    from recal.model.layers import count_parameters, estimate_transformer_flops_per_token


def main():
    parser = argparse.ArgumentParser(description="Parameter and tiny throughput profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ReCalLM(config) if config.get("model_type") == "recal" else BaselineLM(config)
    params = count_parameters(model)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device).eval()
    x = torch.randint(4, config["vocab_size"], (args.batch_size, args.seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    if config.get("model_type") == "recal":
        full_layers = config["front_layers"] + config["recurrent_layers"] + config["back_layers"]
    else:
        full_layers = config["num_layers"]
    flops = estimate_transformer_flops_per_token(full_layers, config["hidden_size"], config["ffn_dim"])
    result = {
        "name": config.get("name"),
        "params": params,
        "rough_dense_flops_per_token": flops,
        "tokens_per_second": args.batch_size * args.seq_len * args.steps / max(elapsed, 1e-6),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
