# ReCal-LM demo

This folder contains a runnable PyTorch demo for the ReCal-LM P0/P1 architecture.
The document requirements in `.md` are implemented as project requirements:

- ReCal model: `E -> A -> R -> B -> LMHead`
- Baseline model: `E -> Transformer -> LMHead`
- RMSNorm, RoPE, SwiGLU, tied embedding/head weights
- ReCal training losses: LM loss, state distillation, logit distillation
- Single-R Router/Executor head: predicts loop-depth budget and calibration trigger probability
- DriftEstimator head: predicts recurrent-state drift against the full-calibration teacher state
- Loop-depth sampling from `N in {1, 2, 4, 8}`

The Router/Executor is intentionally limited to the current single recurrent
module. It does not expose a multi-R module interface yet. The 150M configs keep
hidden size 768 and 18 physical transformer layers. To keep the parameter count
near 150M with SwiGLU, the actual SwiGLU inner dimension is 2048, which is the
parameter-equivalent form of a 3072 dense FFN.

## Quick smoke test

Run from `F:\PyTorch_venv\PyTorch` with the requested virtual environment:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_20m.yaml --steps 2 --batch-size 1 --seq-len 64 --output .\ReCal-LM\runs\smoke-recal
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\baseline_150m.yaml --dry-run
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --dry-run
```

## 150M local demo

On a small GPU, start with short sequences and tiny steps:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\recal-150m-demo
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\baseline_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\baseline-150m-demo
```

If CUDA memory is tight, add `--device cpu` only for correctness checks, or reduce
`--seq-len` to 32.

## 3x500M gate before 3B

The promotion rule is encoded in `configs/experiment_gate_500m.yaml` and
`scripts/run_gate_experiment.py`:

- train ReCal three times and Baseline three times;
- each pilot run targets 500M training tokens;
- evaluate every checkpoint on the same validation data;
- promote only if ReCal has lower average validation loss and wins at least two
  of the three paired seeds.

Generate the full command plan without starting long runs:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --plan-only --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --output .\ReCal-LM\runs\gate_500m
```

`path\to\train.txt` and `path\to\val.txt` are placeholders. In `--plan-only`
mode, the script prints warnings and does not train. In real training mode,
placeholder or missing paths stop immediately before any long job starts.

Run the actual 6 pilot jobs. Existing `checkpoint_last.pt` files are resumed by
default; add `--skip-existing` to skip runs that already have a checkpoint:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --grad-accum 1 --output .\ReCal-LM\runs\gate_500m
```

If the gate passes, the script writes `start_full_3b.ps1`. To start 3B runs
automatically after a passing gate, add `--auto-full`.

Training overwrites `checkpoint_last.pt` at each save interval by default to
avoid filling the disk with many 150M checkpoints. Add
`--keep-interval-checkpoints` to `scripts/train.py` only when you intentionally
want every numbered checkpoint.

## Train a tokenizer

Prepare one or more UTF-8 text files, then run:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train_tokenizer.py --input path\to\text.txt --output .\ReCal-LM\artifacts\tokenizer.json --vocab-size 32000
```

Use it in training with `--tokenizer .\ReCal-LM\artifacts\tokenizer.json`.

## Fetch a tiny FineWeb-Edu sample

`.conf` contains a sample Hugging Face rows API URL. This command writes a small
JSONL text sample for local testing:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\prepare_data.py --conf .\ReCal-LM\.conf --output .\ReCal-LM\data\fineweb_edu_sample.jsonl
```

## Download 500MB FineWeb-Edu training data

Stream from the public `HuggingFaceFW/fineweb-edu` `sample-10BT` config into
local JSONL files:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\download_fineweb_edu.py --target-bytes 500M --val-bytes 10M --output-dir .\ReCal-LM\data\fineweb_edu_500m
```

The output files are:

- `.\ReCal-LM\data\fineweb_edu_500m\train.jsonl`
- `.\ReCal-LM\data\fineweb_edu_500m\val.jsonl`
- `.\ReCal-LM\data\fineweb_edu_500m\manifest.json`

Training reads `.jsonl` files in streaming mode, so the 500MB sample is not
loaded into memory all at once.
