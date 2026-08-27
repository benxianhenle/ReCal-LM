# ReCal-LM

ReCal-LM is a runnable PyTorch demo for a recurrent-calibration language model.
It implements the first language-only stage of a Modular Recurrent Cognitive
Architecture: a shared latent state is calibrated by full attention, updated by a
recurrent path, monitored for drift, and decoded back into language logits.

The project is intentionally small enough to run local smoke tests, while still
keeping a matched ReCal/Baseline experiment structure for larger comparisons.

## Current Architecture

Implemented:

- ReCal model: `E -> A/front -> R/recurrent -> B/back -> LMHead`
- Baseline model: `E -> Transformer -> LMHead`
- RMSNorm, RoPE, SwiGLU, tied embedding/head weights
- Full-calibration teacher path
- Recurrent state update path
- LM loss, state distillation, logit distillation
- Single-R Router/Executor head
- DriftEstimator head
- Loop-depth choices from `N in {1, 2, 4, 8}`
- Matched ReCal/Baseline configs and gate experiment script

Not implemented yet:

- Multiple specialized `R_i` modules
- Vision, audio, control, memory, or tool-use modules
- Cross-module routing
- Runtime full global recalibration triggered by drift threshold

The Router/Executor is currently limited to the single recurrent module. It
predicts loop-depth budget and calibration-trigger probability, but it does not
select among multiple modules yet. The DriftEstimator predicts recurrent-state
drift against the full-calibration teacher state.

## Model Sizes

The 150M ReCal config keeps hidden size 768 and 18 physical transformer layers:

- `front_layers: 7`
- `recurrent_layers: 4`
- `back_layers: 7`
- `ffn_dim: 2048`
- Router/Drift hidden size: 384

With the Router/Executor and DriftEstimator heads included, the verified ReCal
parameter counts are:

- `configs/recal_20m.yaml`: `23,317,056`
- `configs/recal_150m.yaml`: `153,632,256`

## Quick Validation

Run from `F:\PyTorch_venv\PyTorch` with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m compileall .\ReCal-LM\recal .\ReCal-LM\scripts .\ReCal-LM\tests
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_20m.yaml --dry-run
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --dry-run
```

Tiny one-step CPU smoke run:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_20m.yaml --steps 1 --batch-size 1 --seq-len 16 --random-data --device cpu --output .\ReCal-LM\runs\smoke-router-drift --save-interval 1
```

The ReCal training metrics include:

- `loss_lm`
- `loss_state`
- `loss_kd`
- `loss_drift`
- `loss_router`
- `drift_pred`
- `drift_target`
- `router_expected_loop_steps`
- `router_selected_loop_steps`
- `router_calibration_prob`

Evaluate a checkpoint:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\evaluate.py --config .\ReCal-LM\configs\recal_20m.yaml --checkpoint .\ReCal-LM\runs\smoke-router-drift\checkpoint_last.pt --batch-size 1 --seq-len 16 --batches 1 --loop 2 --device cpu
```

## 150M Local Demo

On a small GPU, start with short sequences and tiny steps:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\recal_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\recal-150m-demo
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train.py --config .\ReCal-LM\configs\baseline_150m.yaml --steps 5 --batch-size 1 --seq-len 64 --grad-accum 4 --output .\ReCal-LM\runs\baseline-150m-demo
```

If CUDA memory is tight, reduce `--seq-len` to 32 or use `--device cpu` for
correctness checks only.

## 3x500M Gate Before 3B

The promotion rule is encoded in `configs/experiment_gate_500m.yaml` and
`scripts/run_gate_experiment.py`:

- Train ReCal three times and Baseline three times.
- Each pilot run targets 500M training tokens.
- Evaluate every checkpoint on the same validation data.
- Promote only if ReCal has lower average validation loss and wins at least two
  of the three paired seeds.

Generate the command plan without starting long runs:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --plan-only --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --output .\ReCal-LM\runs\gate_500m
```

`path\to\train.txt` and `path\to\val.txt` are placeholders. In `--plan-only`
mode, the script prints warnings and does not train. In real training mode,
placeholder or missing paths stop immediately before any long job starts.

Run the actual six pilot jobs:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\run_gate_experiment.py --train-data path\to\train.txt --val-data path\to\val.txt --tokenizer .\ReCal-LM\artifacts\tokenizer.json --pilot-tokens 500M --full-tokens 3B --seq-len 2048 --eval-seq-len 2048 --grad-accum 1 --output .\ReCal-LM\runs\gate_500m
```

Existing `checkpoint_last.pt` files are resumed by default. Add
`--skip-existing` to skip runs that already have a checkpoint. If the gate
passes, the script writes `start_full_3b.ps1`. Add `--auto-full` only when you
intentionally want to start the 3B runs automatically.

## Tokenizer And Data

Train a tokenizer:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\train_tokenizer.py --input path\to\text.txt --output .\ReCal-LM\artifacts\tokenizer.json --vocab-size 32000
```

Fetch a tiny FineWeb-Edu sample using `.conf`:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\prepare_data.py --conf .\ReCal-LM\.conf --output .\ReCal-LM\data\fineweb_edu_sample.jsonl
```

Download a 500MB FineWeb-Edu local sample:

```powershell
.\.venv\Scripts\python.exe .\ReCal-LM\scripts\download_fineweb_edu.py --target-bytes 500M --val-bytes 10M --output-dir .\ReCal-LM\data\fineweb_edu_500m
```

Training reads `.jsonl` files in streaming mode, so the 500MB sample is not
loaded into memory all at once.

## Publishing To GitHub

This repository includes `scripts/publish_github.ps1` for creating a GitHub repo
and pushing `main` without storing a token in Git config:

```powershell
.\scripts\publish_github.ps1
```

The script prompts for:

- Repository name, default `ReCal-LM`
- Visibility, default `public`
- Description
- GitHub token, entered as hidden input

For a classic token, use `public_repo` for public repositories or `repo` for
private repositories. For a fine-grained token, make sure it can access the
target repository and has `Contents: Read and write`.

## Git Hygiene

The repository intentionally ignores local secrets, datasets, caches, and
training artifacts:

- `.conf`
- `.hf_cache/`
- `data/`
- `runs/`
- `artifacts/`
- Python cache directories
- model checkpoint files

Do not commit API tokens or generated checkpoints.

## License

MIT License. See `LICENSE`.
