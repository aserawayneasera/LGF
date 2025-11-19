# Evaluation

Fast path to score a checkpoint:

```bash
python - <<'PY'
from train_lgf import evaluate_checkpoint
evaluate_checkpoint(
    ckpt_path="/path/to/BEST_ckpt.pth",
    dataset="custom",
    val_img="/path/to/val/images",
    val_ann="/path/to/val/annotations.json",
    use_ema=True,
    batch_size=16
)
PY
```

The evaluator prints COCO metrics and optional speed numbers.
