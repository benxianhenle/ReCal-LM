import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_count(value: str) -> int:
    text = str(value).strip().replace("_", "").lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000, "b": 1_000_000_000}
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


def load_conf(path: str | None) -> dict:
    if not path:
        return {}
    conf_path = Path(path)
    if not conf_path.exists():
        return {}
    pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$')
    out = {}
    for line in conf_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def set_hf_environment(args) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    conf = load_conf(args.conf)
    token = args.hf_token or conf.get("HuggingFaceFW_Access_Token")
    if token:
        os.environ.setdefault("HF_TOKEN", token)


def row_payload(row: dict) -> dict:
    return {
        "text": row.get("text", ""),
        "id": row.get("id"),
        "url": row.get("url"),
        "dump": row.get("dump"),
        "token_count": row.get("token_count"),
        "score": row.get("score"),
        "int_score": row.get("int_score"),
    }


def write_jsonl_row(handle, row: dict) -> int:
    data = json.dumps(row_payload(row), ensure_ascii=False) + "\n"
    encoded = data.encode("utf-8")
    handle.write(encoded)
    return len(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a bounded FineWeb-Edu sample to local JSONL files.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "fineweb_edu_500m"))
    parser.add_argument("--target-bytes", type=parse_count, default=parse_count("500M"))
    parser.add_argument("--val-bytes", type=parse_count, default=parse_count("10M"))
    parser.add_argument("--target-tokens", type=parse_count, default=None)
    parser.add_argument("--val-tokens", type=parse_count, default=None)
    parser.add_argument("--cache-dir", default=str(ROOT / ".hf_cache"))
    parser.add_argument("--conf", default=str(ROOT / ".conf"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--flush-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_hf_environment(args)
    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    tmp_train = output_dir / "train.jsonl.tmp"
    tmp_val = output_dir / "val.jsonl.tmp"

    target_by_tokens = args.target_tokens is not None
    train_target = args.target_tokens if target_by_tokens else args.target_bytes
    val_target = args.val_tokens if target_by_tokens else args.val_bytes
    metric_name = "tokens" if target_by_tokens else "bytes"

    print(
        f"streaming dataset={args.dataset} config={args.config} split={args.split} "
        f"train_target_{metric_name}={train_target} val_target_{metric_name}={val_target}",
        flush=True,
    )
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    train_metric = 0
    val_metric = 0
    train_rows = 0
    val_rows = 0
    skipped_empty = 0
    started = time.perf_counter()

    with tmp_train.open("wb") as train_file, tmp_val.open("wb") as val_file:
        for row in ds:
            text = row.get("text") or ""
            if not text.strip():
                skipped_empty += 1
                continue
            row_tokens = int(row.get("token_count") or 0)
            if target_by_tokens and row_tokens <= 0:
                row_tokens = max(1, len(text.encode("utf-8")) // 4)

            if train_metric < train_target:
                written = write_jsonl_row(train_file, row)
                train_metric += row_tokens if target_by_tokens else written
                train_rows += 1
            elif val_metric < val_target:
                written = write_jsonl_row(val_file, row)
                val_metric += row_tokens if target_by_tokens else written
                val_rows += 1
            else:
                break

            rows = train_rows + val_rows
            if rows % args.flush_every == 0:
                train_file.flush()
                val_file.flush()
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"rows={rows} train_{metric_name}={train_metric} val_{metric_name}={val_metric} "
                    f"rows_per_sec={rows / elapsed:.2f}",
                    flush=True,
                )

    tmp_train.replace(train_path)
    tmp_val.replace(val_path)
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "metric": metric_name,
        "train_target": train_target,
        "val_target": val_target,
        "train_actual": train_metric,
        "val_actual": val_metric,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "skipped_empty": skipped_empty,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
