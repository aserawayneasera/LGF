#!/usr/bin/env bash
set -e

python - <<'PY'
from train_lgf import evaluate_checkpoint
evaluate_checkpoint(
    ckpt_path="${CKPT:-/path/to/BEST_ckpt.pth}",
    dataset="custom",
    val_img="${VAL_IMG}",
    val_ann="${VAL_ANN}",
    use_ema=True,
    batch_size=16
)
PY
