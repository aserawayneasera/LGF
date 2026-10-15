# Test evaluation update, 15 September 2026

The original Python source files remain unchanged, preserving model behavior and training source hashes. Use the sibling `paper1_test_evaluation_v1/evaluate_test.py` for original RetinaNet/LGF/SE/CBAM and UGS test inference. It supports checkpoint freezing, interrupted-export restart, official COCO submission files, ACDC packaging with official ID metadata, and local evaluation of genuinely unused labelled holdouts.

Read `../paper1_test_evaluation_v1/README_TEST_EVALUATION.md`. Official test downloads do not include ground-truth boxes. Standard all-class server scores are not the paper's six-class metrics. Existing `run.py evaluate` remains the original validation-evaluation command.

---

# Paper 1: UGS comparison code

This package trains a separately labelled **UGS RetinaNet reimplementation** on your existing six-class datasets. It includes training, EMA checkpoint selection, evaluation, restart support, and comparison tables. Your original notebooks and LGF implementation are not edited.

**Scientific status:** this is an independent implementation with documented assumptions, not author-released UGS code or a verified reproduction of its published results. Read [METHOD_NOTES.md](METHOD_NOTES.md) before interpreting the results. The supplied UGS hyperparameters are starting values, not verified author settings.

## 1. Install on atlantis

Copy `paper1_ugs_comparison_v1.zip` to `/tmp/asera/NEW`, then run:

```bash
cd /tmp/asera/NEW
unzip paper1_ugs_comparison_v1.zip
cd paper1_ugs_comparison_v1
sha256sum -c CHECKSUMS.sha256

/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python -m venv --system-site-packages /tmp/asera/paper1_ugs_env
export P1_UGS_PY=/tmp/asera/paper1_ugs_env/bin/python
"$P1_UGS_PY" -m pip install -r requirements-extra.txt
"$P1_UGS_PY" run.py check --device cuda
```

This creates a separate environment that reuses your installed CUDA PyTorch. The runner requires **torch 2.10.x and torchvision 0.25.x**. The install command does not request another PyTorch build. `check --device cuda` stops if CUDA is unavailable. Requirements include Albumentations 2.0.8, matching the interface used by this runner.

The code targets one Linux machine and one GPU. It does not implement distributed training. Set `CUDA_VISIBLE_DEVICES` before launching if you need a particular GPU.

## 2. Use the existing prepared datasets

The default dataset root is `/tmp/asera/paper1_original/DATASET`. If you already copied the datasets with the earlier storage-only notebook package, use those copies. Otherwise:

```bash
mkdir -p /tmp/asera/paper1_original/DATASET/COCO/non_weather-mini
mkdir -p /tmp/asera/paper1_original/DATASET/COCO/weather-mini
mkdir -p /tmp/asera/paper1_original/DATASET/ACDC-1/ACDC-1-NEW

rsync -a /nas.dbms/asera/PROJECTS/DATASET/COCO/non_weather-mini/ /tmp/asera/paper1_original/DATASET/COCO/non_weather-mini/
rsync -a /nas.dbms/asera/PROJECTS/DATASET/COCO/weather-mini/ /tmp/asera/paper1_original/DATASET/COCO/weather-mini/
rsync -a /nas.dbms/asera/PROJECTS/DATASET/ACDC-1/ACDC-1-NEW/ /tmp/asera/paper1_original/DATASET/ACDC-1/ACDC-1-NEW/
```

These commands preserve the existing prepared images and annotation files. They do not regenerate the weather dataset.

| CLI dataset | Directory below the dataset root | Annotation filenames |
| --- | --- | --- |
| `coco_nw` | `COCO/non_weather-mini` | `annotations/mini_train2017_non_weather-2400k6c.json`, `annotations/mini_val2017_non_weather-500k6c.json` |
| `coco_weather` | `COCO/weather-mini` | `annotations/mini_train2017_weather-2400k6c.json`, `annotations/mini_val2017_weather-500k6c.json` |
| `acdc` | `ACDC-1/ACDC-1-NEW` | `annotations/mini_train.json`, `annotations/mini_val.json` |

The corresponding image directories are defined in `data.py`. Use `--data-root` only if you intentionally place the same prepared directory tree elsewhere. The audit checks the six classes, train/validation category mapping, file availability, and exact image-byte overlap. It hashes the inputs for later comparison. These checks do not establish the lineage of transformed weather images.

## 3. Check before committing training time

```bash
"$P1_UGS_PY" run.py smoke --synthetic --device cuda --num-workers 0
"$P1_UGS_PY" run.py audit --dataset coco_nw --device cuda
"$P1_UGS_PY" run.py smoke --dataset coco_nw --device cuda
```

The synthetic check runs the full ResNet-50 model at a small image size without downloading weights. It lowers the score threshold only for synthetic inference so that box decoding and NMS receive nonempty predictions. The real-data check uses your actual training images, the normal 800/1333 transform, and COCO-pretrained weights. Both exercise the frozen-backbone and unfrozen-backbone phases, inference, and checkpoint reload. They do not save experimental results. The real-data check downloads the pretrained weights on first use if they are absent from `TORCH_HOME`.

Optionally run the included CPU tests with `"$P1_UGS_PY" -m unittest discover -s tests -v`. See `VALIDATION.md` for the checks already performed locally.

## 4. Run the first UGS experiment

Run inside your usual persistent terminal session, such as tmux:

```bash
"$P1_UGS_PY" -u run.py train --method ugs_reimpl --dataset coco_nw --seed 42 --device cuda
```

Defaults: 80 epochs, batch size 4, accumulation 4, SGD learning rate 0.005, EMA 0.9999, and FP32. The effective batch size is 16. Localization uses the configuration in `configs/ugs_reimpl.json`.

The new runner preserves the original model construction, augmentation definitions, anchors, optimizer parameters, and actual freeze/warmup/scheduler behavior. Its operational differences, including precision and explicit augmentation seeding, are listed in `METHOD_NOTES.md`. A run through this runner should not be described as a bitwise reproduction of the old notebook training.

Inspect the first run's losses, `diagnostics`, and validation progression before starting the full grid. `clipped_coordinates / regression_coordinates` shows whether the configured localization range is inadequate. Nonzero perturbation directions and finite losses verify that UR is active; they do not establish good detection performance. Do not tune on the final comparison set and present that same set as an untouched test.

Complete the other two seeds after the first run:

```bash
for seed in 1337 2025; do
  "$P1_UGS_PY" -u run.py train --method ugs_reimpl --dataset coco_nw --seed "$seed" --device cuda || break
done
"$P1_UGS_PY" run.py summarize --selection best_ap_small
```

For a dataset with no existing runs, the grid script runs all three seeds:

```bash
bash run_grid.sh ugs_reimpl coco_weather
bash run_grid.sh ugs_reimpl acdc
```

`bash run_grid.sh ugs_reimpl all` is an alternative for a completely fresh run directory. It starts nine full runs. Do not use it to continue the first completed run: existing directories are deliberately protected against accidental overwrite.

## 5. Checkpoints and continuation

Default run directory:

```text
/tmp/asera/paper1_original/ugs_comparison/runs/coco_nw/ugs_reimpl/seed_42/
```

| File | Meaning |
| --- | --- |
| `protocol.json`, `data_audit.json` | Settings, software versions, source hashes, dataset hashes |
| `history.jsonl` | Training losses, diagnostics, timing, validation metrics |
| `last.pth` | Latest complete epoch, model/EMA/optimizer/scheduler/scaler/RNG state |
| `best_ap_small.pth`, `best_map.pth` | Separately selected EMA checkpoints |
| `best_ap_small.metrics.json`, `best_map.metrics.json` | Metrics and selected checkpoint identity |
| `evaluation/epoch_*/predictions.json` | Actual COCO-format detections |
| `completed.json` | Written only after all planned epochs finish |

Resume a stopped run with the same arguments plus its `last.pth`:

```bash
"$P1_UGS_PY" -u run.py train --method ugs_reimpl --dataset coco_nw --seed 42 --device cuda \
  --resume /tmp/asera/paper1_original/ugs_comparison/runs/coco_nw/ugs_reimpl/seed_42/last.pth
```

Resume verifies the saved protocol, including data, source code, environment, and total planned epochs. It resumes from a completed epoch; an interrupted partial epoch is repeated. Multiworker augmentation continuation is not guaranteed to be bitwise identical. If a run failed before its first checkpoint, use a new `--run-root` after resolving the cause.

All dataset reads and model writes use the requested temporary locations by default. `/tmp` can be cleared by the server; copy finished runs to your chosen durable location before that happens. No automatic NAS checkpoint mirroring is enabled. Pretrained weights are cached at `/tmp/asera/paper1_original/torch_cache` unless you already set `TORCH_HOME`.

## 6. Compare with LGF

Choose one checkpoint-selection rule for every compared method. Your uploaded trainer saves best mAP, while its aggregation code selects the highest logged APsmall. A best-mAP checkpoint cannot automatically reproduce the peak-APsmall table row.

If the exact LGF checkpoints and selection history are available, evaluate them with this runner. Replace the checkpoint path below with the actual file and supply its actual seed and original level label. This example is appropriate only after verifying that the checkpoint was selected by mAP:

```bash
"$P1_UGS_PY" run.py evaluate --original-checkpoint \
  --checkpoint /absolute/path/to/verified_lgf_seed42.pth \
  --dataset coco_nw --level C3 --seed 42 --device cuda \
  --selection-note best_map \
  --output /tmp/asera/paper1_original/ugs_comparison/evaluated_lgf/coco_nw/seed_42

"$P1_UGS_PY" compare.py \
  --reports \
  /tmp/asera/paper1_original/ugs_comparison/evaluated_lgf/coco_nw/seed_42/metrics.json \
  /tmp/asera/paper1_original/ugs_comparison/runs/coco_nw/ugs_reimpl/seed_42/best_map.metrics.json \
  --output /tmp/asera/paper1_original/ugs_comparison/comparisons/coco_nw_seed42
```

`--selection-note` is your provenance declaration, not automatic proof. Only declare `best_ap_small` when the saved weights genuinely correspond to that selection. The original checkpoint must contain its configuration and EMA weights. Dataset, level, and seed are checked against the original experiment name when it follows the uploaded naming format.

For the final three-seed comparison, supply all six report paths to `compare.py`: three LGF and three UGS reports. It requires matching seeds, validation-content hashes, evaluation settings, and selection rules, and writes per-run scores plus mean and sample standard deviation. Scores use the 0–1 AP scale. No score is synthesized or copied from the UGS paper. Matching evaluation does not establish identical training histories.

If suitable original checkpoints are unavailable, the package can train your unchanged baseline and LGF model definitions through the same new runner:

```bash
bash run_grid.sh baseline coco_nw
bash run_grid.sh lgf_original coco_nw
"$P1_UGS_PY" run.py summarize --selection best_ap_small
```

These are additional experiments; the original notebooks remain intact. The original `C3` placement mapping is preserved exactly. This package does not resolve the separate placement question.

## Files to inspect

`ugs.py` contains the new localization implementation. `core_bridge.py` imports a checksum-verified, unmodified copy of the uploaded notebook's definition cell. `run.py` provides the separate training loop. `evaluation.py` invokes the original COCO evaluator and captures the same detections. `METHOD_NOTES.md` documents method assumptions and comparison limits. No pretrained or trained checkpoint is included in the download.
