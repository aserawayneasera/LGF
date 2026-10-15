# Validation performed

Date: 2026-09-14. All checks below used synthetic inputs on CPU. No COCO/ACDC training run or LGF comparison result was produced in this workspace.

## Environment

Python 3.12.14; torch 2.10.0+cpu; torchvision 0.25.0+cpu; Albumentations 2.0.8; NumPy 2.3.5; pycocotools 2.0.11; opencv-python-headless 5.0.0.93. CUDA was unavailable.

## Automated checks

The 25 tests cover:

- Nonuniform grid symmetry, monotonicity, endpoints, and the uniform limit.
- Normalized two-hot targets, exact in-range reconstruction, explicit clipping, and invalid configurations.
- Cross-entropy and entropy derivatives, including the softmax chain rule.
- Expected-offset decoding against known bounding boxes.
- Feature perturbation norms, zero-gradient handling, and gradients into features and distribution weights.
- The frozen-backbone phase, no-positive-target behavior, unchanged classification loss, and inference without the refinement branch.
- Baseline and LGF bridge equivalence to their original model definitions, using identical weights and nonempty predictions. This validates the bridge; it is not a comparison of trained detectors.
- Dataset category mismatch and exact cross-split duplicate detection.
- Native COCO evaluation and exported category IDs/box coordinates using a deliberately perfect synthetic detector fixture.
- Real ResNet-50 training on tiny synthetic images, interrupted epoch-commit recovery, selected checkpoint creation, EMA reload, and evaluation seed metadata.
- Matching-seed comparison statistics and rejection of incompatible evaluation settings, selection rules, seeds, or UGS configurations.

Reproduce with:

```bash
python -m unittest discover -s tests -v
```

The independent CLI smoke check ran one optimizer step in each training phase through the full ResNet-50 detector at 64×64. It verified finite losses, active UR directions, nonempty decoded predictions/NMS, and identical predictions after checkpoint reload. Only synthetic smoke inference lowers the score threshold to zero; ordinary training/evaluation retains 0.05. The smoke report is `validation/synthetic_smoke.json`.

The grid shell script passed `bash -n run_grid.sh`. The environment check passed on CPU. The archived original definition cell matches its source SHA-256, and the four original uploads and four previously delivered storage-only notebooks retain their recorded hashes. These checks are recorded in `validation/source_integrity.json`.

## Limits

The real prepared datasets, server filesystem, COCO-pretrained download, full 800/1333 GPU training, CUDA kernels, AMP path, and actual convergence were not exercised locally. The software checks do not validate the unspecified UGS hyperparameters or reproduce the authors' benchmark results. Run the real-data smoke check on atlantis before the first full experiment. Synthetic fixture AP values are test assertions, not research results.
