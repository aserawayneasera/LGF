# Local-Global Fusion (LGF) for Small-Object Detection

LGF mixes local detail with global context at C3 in RetinaNet. A depthwise local path and a global context path feed a lightweight gate. The fused output goes into FPN. The goal is stronger AP on small objects in clear and adverse conditions with tight compute and latency.

## Highlights

- Plug-in module at C3 for RetinaNet ResNet‑50 + FPN
- Depthwise 3×3 local branch, GAP + 1×1 global branch, GroupNorm
- Per-pixel, per-channel gating with sigmoid, residual connection
- +1.2 AP_small on COCO Non‑weather, +1.2 AP_small on COCO Weather, mAP near baseline
- Gains on ACDC with fog, night, rain, snow
- +9% params, +3.6% FLOPs, +6% latency at 640px

## Install

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Data

You can point to any COCO-style dataset via flags. Examples below assume COCO and ACDC already exist locally.

Folder structure expectations:
- Images in a folder
- COCO JSON annotations

## Quick start

Train a baseline RetinaNet on a custom COCO-style dataset:

```bash
python run.py --mode baseline --dataset custom   --train-img /path/to/train/images   --train-ann /path/to/train/annotations.json   --val-img   /path/to/val/images   --val-ann   /path/to/val/annotations.json   --insert-level C3 --epochs 80 --batch-size 4 --accum-steps 4
```

Train LGF with gated spatial mixing at C3:

```bash
python run.py --mode lgf_gated_spatial --dataset custom   --train-img /path/to/train/images   --train-ann /path/to/train/annotations.json   --val-img   /path/to/val/images   --val-ann   /path/to/val/annotations.json   --insert-level C3 --epochs 80 --batch-size 4 --accum-steps 4
```

Evaluate a saved checkpoint on a dataset:

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

Reproduce the paper’s 3-seed grid for one dataset:

```bash
python run.py --run-grid --grid-datasets coco_nw   --grid-modes baseline lgf_gated_spatial cbam se   --grid-levels C3 --grid-seeds 42 1337 2025
```

## Results summary

COCO Non‑weather (validation)

| Method     | AP_small | AP_medium | AP_large | AP50  | AP75  | mAP  |
|------------|---------:|----------:|---------:|------:|------:|-----:|
| Baseline   | 0.218    | 0.385     | 0.444    | 0.657 | 0.294 | 0.330 |
| CBAM       | 0.215    | 0.404     | 0.445    | 0.653 | 0.308 | 0.335 |
| SE         | 0.213    | 0.388     | 0.436    | 0.652 | 0.287 | 0.323 |
| LGF (C3)   | 0.230    | 0.390     | 0.440    | 0.663 | 0.297 | 0.331 |

COCO Weather (validation)

| Method     | AP_small | AP_medium | AP_large | AP50  | AP75  | mAP  |
|------------|---------:|----------:|---------:|------:|------:|-----:|
| Baseline   | 0.177    | 0.328     | 0.390    | 0.605 | 0.210 | 0.280 |
| CBAM       | 0.167    | 0.340     | 0.406    | 0.599 | 0.228 | 0.287 |
| SE         | 0.166    | 0.330     | 0.401    | 0.602 | 0.217 | 0.281 |
| LGF (C3)   | 0.189    | 0.331     | 0.396    | 0.605 | 0.227 | 0.283 |

ACDC (validation)

| Method     | AP_small | AP_medium | AP_large | AP50  | AP75  | mAP  |
|------------|---------:|----------:|---------:|------:|------:|-----:|
| Baseline   | 0.076    | 0.331     | 0.536    | 0.549 | 0.340 | 0.322 |
| CBAM       | 0.074    | 0.326     | 0.531    | 0.543 | 0.332 | 0.318 |
| SE         | 0.075    | 0.329     | 0.523    | 0.543 | 0.331 | 0.317 |
| LGF (C3)   | 0.083    | 0.344     | 0.541    | 0.561 | 0.347 | 0.330 |

Complexity at 640px

| Model   | Params (M) | FLOPs (G) | Latency (ms/img) | Images/s |
|---------|-----------:|----------:|-----------------:|---------:|
| Baseline| 32.25      | 81.85     | 10.98            | 91.05    |
| CBAM    | 32.38      | 81.86     | 11.39            | 87.79    |
| SE      | 32.38      | 81.86     | 11.05            | 90.50    |
| LGF     | 35.16      | 84.81     | 11.64            | 85.93    |

## Method overview

- Insert LGF at C3 before FPN.
- Local branch, depthwise 3×3 then 1×1, GroupNorm, ReLU.
- Global branch, GAP, 1×1, GroupNorm, SiLU, broadcast to H×W.
- Gating, per-pixel and per-channel with sigmoid. Fuse local and global. Residual add to input.
- Sigmoid gating ranked first in AP_small in ablations. Sum and softmax trailed closely.
- Fusion drives most of the gain. Learned gates refine the mix.

## Repo layout

```
.
├── train_lgf.py            # main training script
├── run.py                  # thin CLI wrapper
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── docs/
│   ├── installation.md
│   ├── training.md
│   ├── evaluation.md
│   ├── design.md
│   └── results.md
├── scripts/
│   ├── train.sh
│   └── eval.sh
└── .github/workflows/ci.yml
```

## Citation

If you use LGF in research, cite the preprint in this repo and the code.

```bibtex
@misc{lgf2025,
  title  = {Local-Global Fusion for Small-Object Detection},
  author = {Asera, Wayne},
  year   = {2025},
  note   = {Code and documentation}
}
```

## License

MIT
