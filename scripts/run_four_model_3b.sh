#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
PY=/workspace/ai-training/.venv/bin/python
resume=()
manifest=runs/four-model-r3b-10b/checkpoints/manifest.json
if [[ -f "$manifest" ]]; then
    latest=$("$PY" -c 'import json; from pathlib import Path; p=Path("runs/four-model-r3b-10b/checkpoints"); m=json.loads((p/"manifest.json").read_text()); print(p / m["roles"]["latest"])')
    resume=(--resume "$latest")
fi
exec "$PY" -u scripts/train_four_models.py \
    --config configs/four_model_3b.yaml \
    --data artifacts/four-model-v1/data_manifest.json \
    --val-data artifacts/four-model-v1/data_manifest.json \
    --tokenizer artifacts/four-model-v1/tokenizer.json \
    --output runs/four-model-r3b-10b \
    --target-tokens 10B --weights-only "${resume[@]}" "$@"
