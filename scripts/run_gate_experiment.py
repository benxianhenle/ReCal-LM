"""Orchestrate the paired 3x500M ReCal-vs-baseline gate experiment.

中文：编排三组配对 500M token 的 ReCal 对 baseline gate 实验。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_BITS = ("path\\to", "path/to", "your\\", "your/")


def parse_count(value: str) -> int:
    """Parse compact count strings such as 500M into integer totals.

中文：将 500M 等紧凑计数字符串解析为整数总量。"""

    text = str(value).strip().replace("_", "").lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


def parse_seeds(value: str) -> list[int]:
    """Parse and validate the three paired random seeds for the gate.

中文：解析并校验 gate 实验所需的三个配对随机种子。"""

    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(seeds) != 3:
        raise argparse.ArgumentTypeError("Exactly three seeds are required")
    return seeds


def rel(path: Path) -> str:
    """Return a path relative to the project root when possible.

中文：尽可能返回相对于项目根目录的路径。"""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_text(cmd: list[str]) -> str:
    """Format a subprocess command for logs and plan files.

中文：将子进程命令格式化，供日志和计划文件使用。"""

    return " ".join(cmd)


def run_command(cmd: list[str], plan_only: bool, verbose: bool = True) -> str:
    """Print a command and optionally execute it.

中文：打印命令，并按需实际执行。"""

    text = command_text(cmd)
    if verbose:
        print(text, flush=True)
    if not plan_only:
        subprocess.run(cmd, check=True)
    return text


def looks_like_placeholder(value: str | None) -> bool:
    """Detect example path fragments that should not run real experiments.

中文：检测不应参与真实实验的示例路径片段。"""

    if not value:
        return False
    text = value.lower()
    return any(bit in text for bit in PLACEHOLDER_BITS)


def validate_paths(args) -> None:
    """Validate train, validation, and tokenizer paths before running jobs.

中文：在运行任务前校验训练、验证和 tokenizer 路径。"""

    warnings = []
    errors = []
    for label, value in [("train-data", args.train_data), ("val-data", args.val_data), ("tokenizer", args.tokenizer)]:
        if looks_like_placeholder(value):
            message = f"--{label} is still an example path: {value}"
            if args.plan_only:
                warnings.append(message)
            else:
                errors.append(message)
        elif value and not Path(value).exists():
            message = f"--{label} does not exist: {value}"
            if args.plan_only:
                warnings.append(message)
            else:
                errors.append(message)
    if not args.random_data and not args.train_data:
        errors.append("--train-data is required for a real experiment unless --random-data is used for debugging")
    if not args.val_data:
        errors.append("--val-data is required for a real gate decision")
    if warnings:
        print("PLAN WARNING:", flush=True)
        for warning in warnings:
            print(f"  {warning}", flush=True)
        print("  Replace example paths before running without --plan-only.", flush=True)
    if errors:
        raise SystemExit("Input path check failed:\n" + "\n".join(f"  {error}" for error in errors))


def evaluate_command(args, config: Path, checkpoint: Path, output_json: Path) -> tuple[dict | None, str]:
    """Build and optionally run the validation command for one checkpoint.

中文：为一个 checkpoint 构建并可选执行验证命令。"""

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(args.eval_batch_size),
        "--seq-len",
        str(args.eval_seq_len),
        "--batches",
        str(args.eval_batches),
        "--device",
        args.device,
    ]
    if args.val_data:
        cmd += ["--data", args.val_data]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if config.name.startswith("recal"):
        cmd += ["--loop", str(args.eval_loop)]
    text = command_text(cmd)
    if args.verbose_plan or not args.plan_only:
        print(text, flush=True)
    if args.plan_only:
        return None, text
    raw = subprocess.check_output(cmd, text=True)
    result = json.loads(raw)
    output_json.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(raw, flush=True)
    return result, text


def train_one(args, model_name: str, config: Path, seed: int, tokens: int, stage: str) -> tuple[Path, str]:
    """Train or plan one model/seed/stage run and return its last checkpoint path.

中文：训练或规划一个模型/种子/阶段组合，并返回其 last checkpoint 路径。"""

    run_dir = Path(args.output) / stage / f"seed_{seed}" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_last.pt"
    resume_checkpoint = checkpoint
    if not resume_checkpoint.exists():
        # Fall back to the newest numbered interval checkpoint if last is missing.
        numbered = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        numbered = [path for path in numbered if path.name != "checkpoint_last.pt"]
        if numbered:
            resume_checkpoint = numbered[0]
    if checkpoint.exists() and args.skip_existing:
        print(f"skip existing {checkpoint}", flush=True)
        return checkpoint, ""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(config),
        "--target-tokens",
        str(tokens),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--grad-accum",
        str(args.grad_accum),
        "--save-interval",
        str(args.save_interval),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output",
        str(run_dir),
    ]
    if resume_checkpoint.exists():
        cmd += ["--resume", str(resume_checkpoint)]
    if args.train_data:
        cmd += ["--data", args.train_data]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.random_data:
        cmd += ["--random-data"]
    text = run_command(cmd, args.plan_only, verbose=(args.verbose_plan or not args.plan_only))
    return checkpoint, text


def write_full_plan(args, seeds: list[int], recal_config: Path, baseline_config: Path) -> Path:
    """Write a PowerShell launcher for the full 3B-token follow-up runs.

中文：为后续完整 3B token 训练写出 PowerShell 启动脚本。"""

    plan_path = Path(args.output) / "start_full_3b.ps1"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$python = '{sys.executable}'",
        "$root = Split-Path -Parent $PSScriptRoot",
    ]
    for seed in seeds:
        for name, cfg in [("recal", recal_config), ("baseline", baseline_config)]:
            run_dir = Path(args.output) / "full_3b" / f"seed_{seed}" / name
            cmd = [
                "& $python",
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(cfg),
                "--target-tokens",
                str(args.full_tokens),
                "--batch-size",
                str(args.batch_size),
                "--seq-len",
                str(args.seq_len),
                "--grad-accum",
                str(args.grad_accum),
                "--save-interval",
                str(args.save_interval),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output",
                str(run_dir),
            ]
            if args.train_data:
                cmd += ["--data", args.train_data]
            if args.tokenizer:
                cmd += ["--tokenizer", args.tokenizer]
            lines.append(" ".join(f"'{part}'" if " " in part else part for part in cmd))
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan_path


def parse_args():
    """Parse gate orchestration options.

中文：解析 gate 编排脚本选项。"""

    parser = argparse.ArgumentParser(description="Run the 3x500M ReCal/Baseline gate, then optionally start 3B runs.")
    parser.add_argument("--recal-config", default=str(ROOT / "configs" / "recal_150m.yaml"))
    parser.add_argument("--baseline-config", default=str(ROOT / "configs" / "baseline_150m.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "gate_500m"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("2026082201,2026082202,2026082203"))
    parser.add_argument("--pilot-tokens", type=parse_count, default=500_000_000)
    parser.add_argument("--full-tokens", type=parse_count, default=3_000_000_000)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-loop", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verbose-plan", action="store_true", help="Print every planned train/eval command.")
    parser.add_argument("--auto-full", action="store_true", help="Actually start 3B runs if the gate passes.")
    parser.add_argument("--random-data", action="store_true", help="Debug only; do not use for real gate decisions.")
    return parser.parse_args()


def main() -> None:
    """Run or plan the pilot gate, write summary JSON, and handle promotion.

中文：运行或规划 pilot gate，写出汇总 JSON，并处理晋级逻辑。"""

    args = parse_args()
    validate_paths(args)
    if args.plan_only:
        print(
            "PLAN ONLY: no training or evaluation is being run; commands below are a dry-run plan.",
            flush=True,
        )
    recal_config = Path(args.recal_config)
    baseline_config = Path(args.baseline_config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results = []
    plan_commands = []
    planned_train_jobs = 0
    planned_eval_jobs = 0
    for seed in args.seeds:
        # Each seed trains and evaluates both models so paired wins are meaningful.
        seed_result = {"seed": seed}
        for model_name, config in [("recal", recal_config), ("baseline", baseline_config)]:
            checkpoint, train_text = train_one(args, model_name, config, seed, args.pilot_tokens, "pilot_500m")
            if train_text:
                plan_commands.append(train_text)
            planned_train_jobs += 1
            eval_json = output / "pilot_500m" / f"seed_{seed}" / model_name / "eval.json"
            if args.plan_only:
                _, eval_text = evaluate_command(args, config, checkpoint, eval_json)
                plan_commands.append(eval_text)
                planned_eval_jobs += 1
            elif eval_json.exists() and args.skip_existing:
                seed_result[model_name] = json.loads(eval_json.read_text(encoding="utf-8"))
            else:
                seed_result[model_name], _ = evaluate_command(args, config, checkpoint, eval_json)
        if not args.plan_only and seed_result["recal"] and seed_result["baseline"]:
            seed_result["recal_wins"] = seed_result["recal"]["loss"] < seed_result["baseline"]["loss"]
        if not args.plan_only:
            results.append(seed_result)

    summary = {"pilot_tokens_per_run": args.pilot_tokens, "seeds": args.seeds}
    if not args.plan_only:
        # Promotion requires both lower average loss and at least two paired wins.
        summary["results"] = results
        recal_losses = [item["recal"]["loss"] for item in results]
        baseline_losses = [item["baseline"]["loss"] for item in results]
        wins = sum(1 for item in results if item["recal_wins"])
        recal_avg = sum(recal_losses) / len(recal_losses)
        baseline_avg = sum(baseline_losses) / len(baseline_losses)
        gate_pass = recal_avg < baseline_avg and wins >= 2
        summary.update(
            {
                "recal_average_val_loss": recal_avg,
                "baseline_average_val_loss": baseline_avg,
                "paired_recal_wins": wins,
                "gate_pass": gate_pass,
                "gate_rule": "ReCal average validation loss < Baseline average and ReCal wins at least 2 of 3 paired seeds.",
            }
        )
    else:
        plan_path = output / "pilot_500m_plan.ps1"
        plan_path.write_text("\n".join(plan_commands) + "\n", encoding="utf-8")
        summary.update(
            {
                "mode": "plan_only",
                "planned_pilot_train_jobs": planned_train_jobs,
                "planned_eval_jobs": planned_eval_jobs,
                "plan_path": str(plan_path),
                "results": "not_run_in_plan_only_mode",
            }
        )
        summary["gate_rule"] = "Plan only: run all six 500M-token pilot jobs, evaluate, then require ReCal avg loss lower and at least 2/3 paired wins."

    summary_path = output / "gate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)

    if summary.get("gate_pass"):
        plan_path = write_full_plan(args, args.seeds, recal_config, baseline_config)
        print(f"full 3B plan written: {plan_path}", flush=True)
        if args.auto_full:
            for seed in args.seeds:
                train_one(args, "recal", recal_config, seed, args.full_tokens, "full_3b")
                train_one(args, "baseline", baseline_config, seed, args.full_tokens, "full_3b")


if __name__ == "__main__":
    main()
