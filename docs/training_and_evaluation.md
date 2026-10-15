# Training, checkpoint selection, and evaluation

This repository assembles the recovered research packages used for the revision. The root launcher starts each package in a separate Python process. Model construction and archived training code retain their existing definitions. The package-specific README files contain the original server commands; the commands below work from the repository root with explicit data and output locations.

The retained manuscript results and a fresh run through a revised runner are separate records. Running `baseline` or `lgf_original` through the revised runner creates a new experiment. Those outputs do not replace the original RetinaNet/LGF table rows.

## 1. Runtime and package identity

Use the matching environment from the original experiment when resuming a run. The model adapters require the Torch/Torchvision version families below. Install a matching CUDA build for the target GPU through the official PyTorch installation instructions, then install the supplied extra requirements. Do not install a CPU build into an existing CUDA experiment environment.

| Component | Required or recorded identity |
| --- | --- |
| Original/UGS revised runner | `paper1_ugs_comparison_v1`, internal version `1.0.0` |
| Detector comparison runner | `paper1_reviewer34_v1`, repair version `1.0.1`; prepared-label schema `1.0.0` |
| Shared test exporter | `paper1_test_evaluation_v1`; manifest schema `paper1-test-data-1.0` |
| SE/CBAM extension | `paper1_se_cbam_extension_nas_v1_0_2`, internal version `1.0.2` |
| Torch / Torchvision | `torch 2.10.x`, `torchvision 0.25.x`, checked by the revised runners |
| Albumentations | `2.0.8` |
| Ultralytics | `8.3.253`, checked by the detector runner |
| Ultralytics extras | `ultralytics-thop==2.0.18`, `polars==1.35.2` |
| COCO evaluator | Detector extras pin `pycocotools==2.0.11`; original extras accept `>=2.0.7,<3` |
| Original-runner extras | TensorBoard `>=2.16,<3`, matplotlib `>=3.7`, tqdm `>=4.66` |
| Image I/O | NumPy, OpenCV, and Pillow, also used by the official-test toolkit |

```bash
python -m pip install -r experiments/paper1_ugs_comparison_v1/requirements-extra.txt
python -m pip install -r experiments/paper1_reviewer34_v1/requirements-reviewer34.txt
```

The recovered CPU validation records used Python `3.12.14`, Torch `2.10.0+cpu`, Torchvision `0.25.0+cpu`, Albumentations `2.0.8`, NumPy `2.3.5`, pycocotools `2.0.11`, and opencv-python-headless `5.0.0.93`. These are historical software checks, not the measured GPU environment for the manuscript. Each real run writes its own environment into `protocol.json`. SE/CBAM matched training compares the actual interpreter and package versions against its reference record, allowing a GPU name difference while recording the difference.

Use one GPU per process. The original runner uses `--device cuda` or `cpu`; the detector and test exporters use `--device 0` or `cpu`. Set `CUDA_VISIBLE_DEVICES` before launch to select the physical GPU. The packages do not implement distributed training.

## 2. Data preparation and directory contract

The prepared datasets and trained weights are not included. Set paths to your existing data and durable output directory:

```bash
export LGF_REPO="$PWD"
export LGF_DATA_ROOT=/absolute/path/DATASET
export LGF_RUN_ROOT=/absolute/path/revision_runs
export LGF_TEST_ROOT=/absolute/path/TEST_DATASET
export TORCH_HOME=/absolute/path/torch_cache
```

The revised original/UGS runner accepts this tree:

| Dataset flag | Paths below `LGF_DATA_ROOT` |
| --- | --- |
| `coco_nw` | `COCO/non_weather-mini/images/train2017_non_weather-2400k6c/`; `images/val2017_non_weather-500k6c/`; `annotations/mini_train2017_non_weather-2400k6c.json`; `annotations/mini_val2017_non_weather-500k6c.json` |
| `coco_weather` | `COCO/weather-mini/images/train2017_weather-2400k6c/`; `images/val2017_weather-500k6c/`; `annotations/mini_train2017_weather-2400k6c.json`; `annotations/mini_val2017_weather-500k6c.json` |
| `acdc` | `ACDC-1/ACDC-1-NEW/images/train/`; `images/val/`; `annotations/mini_train.json`; `annotations/mini_val.json` |

Directory names are historical names. The audits count the actual annotation records and hash the actual images. Record the resulting counts instead of inferring counts from a folder name. The manuscript's expanded synthetic-weather condition analysis uses the same 500 source scenes in fog, rain, and snow, totaling 1,500 images. This expanded analysis is distinct from the original 500-image weather validation subset used by training and checkpoint selection.

The retained preparation notebooks are in `legacy/notebooks/`:

1. `1a.visualize_create_coco_6c_train&val_dataset.ipynb`, six-class source selection and prepared splits.
2. `1b.create_weather_aug_coco_dataset.ipynb`, fog/rain/snow preparation and visual checks.
3. `2.train_evaluate_controls.ipynb`, original training, aggregation, evaluation, and historical control cells.
4. `3.parameters.ipynb`, original complexity/timing notebook.

Review the notebook paths before running a preparation cell. Preserve existing splits for reproducing the submitted experiments. The preparation scripts/notebooks also serve as source evidence for split-before-weather construction. Exact-byte duplicate checks in the revised trainer do not detect every transformed copy of a source scene; use the source-ID leakage audit documented in the weather workflow for this separate question.

## 3. Entry points

```bash
python run.py list
python run.py original -- --help
python run.py detectors -- --help
python run.py official-export -- --help
python run.py se-cbam -- --help
```

| Root workflow | Preserved entry point | Actions |
| --- | --- | --- |
| `original`, alias `ugs` | `paper1_ugs_comparison_v1/run.py` | `check`, `audit`, `smoke`, `train`, `evaluate`, `summarize` |
| `detectors` | `paper1_reviewer34_v1/reviewer34_compare.py` | `check`, `prepare`, `smoke`, `train`, `evaluate`, `compare` |
| `official-export` | `paper1_test_evaluation_v1/evaluate_test.py` | `export`, `score`, `compare`, `package-acdc` |
| `se-cbam` | `paper1_se_cbam_extension_nas_v1_0_2/run_se_cbam.py` | Native original-runner actions with explicit SE/CBAM construction |
| `legacy-train` | `legacy/train_lgf.py` | Historical CLI, including original fusion variants and grid options |

The `--` separator passes the remaining arguments to the original CLI. For auxiliary tools such as `test_data.py`, `compare.py`, or `train_se_cbam_grid.py`, use the direct commands below. The root launcher preserves the caller's working directory for relative input/output paths. The examples use absolute data/checkpoint/output paths for clarity.

The historical `run_grid.sh` and older package README commands preserve `/tmp/asera` defaults. The SE/CBAM extension preserves `/nas.dbms/asera` defaults. Use explicit path flags for a different machine. Set `TORCH_HOME` explicitly before launching any package; the SE/CBAM extension rewrites the old `/tmp/asera` cache prefix to its NAS prefix by design.

## 4. Original RetinaNet/LGF definitions and UGS adaptation

Run checks before starting full training:

```bash
python run.py original -- check --device cuda
python run.py original -- smoke --method ugs_reimpl --synthetic --device cuda --num-workers 0
python run.py original -- audit --dataset coco_nw --data-root "$LGF_DATA_ROOT" --device cuda
python run.py original -- smoke --method ugs_reimpl --dataset coco_nw --data-root "$LGF_DATA_ROOT" --device cuda
```

Train the UGS reimplementation for one seed:

```bash
python run.py original -- train \
  --method ugs_reimpl --dataset coco_nw --level C3 --seed 42 \
  --data-root "$LGF_DATA_ROOT" --run-root "$LGF_RUN_ROOT/original" --device cuda
```

Repeat using datasets `coco_nw`, `coco_weather`, and `acdc`, and seeds `42`, `1337`, and `2025`. The same entry point accepts `--method baseline` or `lgf_original` for additional definition-preserving controls. The extension described below adds `se` and `cbam` without changing the original package's recursive source hash.

Defaults are 80 epochs, batch 4, accumulation 4, SGD learning rate 0.005, weight decay 1e-4, EMA 0.9999, and FP32. The model's real-image transform is 800-pixel shorter side with a 1333-pixel longer-side cap. `--amp` enables the optional CUDA AMP route; choose precision explicitly for new comparisons. The historical notebook training used AMP on CUDA. The revised loop also explicitly seeds Albumentations per run/worker. A new run is not a bitwise replay of historical training.

The actual preserved schedule trains heads and attention first, unfreezes at epoch index 5, and resets the optimizer/scheduler at unfreezing. Milestones `[40, 60]` apply after that reset. The original multiplicative warmup halves the current learning rate at the first epoch and multiplies by one at the second; the resulting rate persists until the reset. See `experiments/paper1_ugs_comparison_v1/METHOD_NOTES.md` for this implementation detail and the independent UGS choices.

UGS is an independent RetinaNet adaptation with distributional localization CE, uncertainty minimization, and first-order feature refinement. It is not author-released UGS code or a reproduction of the paper's benchmark results. `configs/ugs_reimpl.json` records the actual support-grid and loss parameters. Retain the label “UGS (our RetinaNet reimplementation)” and preserve any tuning disclosure.

Resume the latest complete epoch with identical settings and the same source/environment:

```bash
python run.py original -- train \
  --method ugs_reimpl --dataset coco_nw --level C3 --seed 42 \
  --data-root "$LGF_DATA_ROOT" --run-root "$LGF_RUN_ROOT/original" --device cuda \
  --resume "$LGF_RUN_ROOT/original/coco_nw/ugs_reimpl/seed_42/last.pth"
```

The original training loader requires a nonzero number of batches divisible by the accumulation count. A partial epoch is repeated on restart. Multiworker augmentation continuation does not guarantee identical draws to an uninterrupted run.

## 5. YOLO11m and RT-DETR-L

The supplied comparison trains standard Ultralytics YOLO11m and RT-DETR-L with HGNetv2. The runner does not add LGF to these architectures. The attention-augmented YOLO11 paper discussed by the reviewers is a separate related work.

```bash
python run.py detectors -- check --device 0
python run.py detectors -- smoke --model yolo11m --device 0
python run.py detectors -- smoke --model rtdetr-l --device 0
python run.py detectors -- prepare --dataset coco_nw \
  --data-root "$LGF_DATA_ROOT" --work-root "$LGF_RUN_ROOT/detectors" --device 0

python run.py detectors -- train --model yolo11m --dataset coco_nw --seed 42 \
  --data-root "$LGF_DATA_ROOT" --work-root "$LGF_RUN_ROOT/detectors" --device 0

python run.py detectors -- train --model rtdetr-l --dataset coco_nw --seed 42 \
  --data-root "$LGF_DATA_ROOT" --work-root "$LGF_RUN_ROOT/detectors" --device 0
```

Repeat across the three datasets and seeds. Training defaults are 80 epochs, 1280x1280 inputs, batch 4, nominal accumulation target 16, FP32, two warmup epochs, and disabled early stopping. YOLO11m uses SGD with learning rate 0.005; RT-DETR-L uses AdamW with learning rate 0.0001. Both use weight decay 1e-4 and COCO pretraining. Common validation uses 800/1333 resizing. These are explicit fine-tuning recipes, not identical training recipes to historical RetinaNet.

Preparation writes separate YOLO-format labels and links to existing images. The original COCO annotations remain the evaluation source. The runner rejects invalid/undecodable images, dimension mismatches, duplicate label rows, out-of-bounds training boxes, and exact train/validation image duplicates. These checks do not prove source-scene independence.

Add `--resume` to the same detector training command to resume an interrupted run. The `1.0.1` repair stabilizes inference settings at the Ultralytics trainer teardown/reload boundary and records reload diagnostics. Runs started under an older script retain their original source requirement; do not rewrite their saved signatures to accept a new script.

## 6. SE/CBAM extension

The NAS `1.0.2` extension calls the unchanged original/UGS runner and installs its SE/CBAM selection adapter in memory. It directly uses the archived `SEBlock` and `CBAMBlock`. The archived CBAM comparator uses softmax pooling with beta 20 and a 5x5 spatial kernel; describe this variant as CBAM-style rather than asserting canonical CBAM.

```bash
python run.py se-cbam -- check --device cuda
python run.py se-cbam -- smoke --method se --synthetic --device cuda --num-workers 0
python run.py se-cbam -- smoke --method cbam --synthetic --device cuda --num-workers 0

python run.py se-cbam -- train --method se --dataset coco_nw --level C3 --seed 42 \
  --data-root "$LGF_DATA_ROOT" --run-root "$LGF_RUN_ROOT/original" --device cuda
```

For settings matched to existing revised-runner LGF/baseline records, use the grid instead of manually copying options. This route requires the existing `protocol.json` for every requested dataset/seed. Historical notebook logs are not compatible reference protocols.

```bash
python experiments/paper1_se_cbam_extension_nas_v1_0_2/train_se_cbam_grid.py \
  --package "$LGF_REPO/experiments/paper1_ugs_comparison_v1" \
  --reference-root "$LGF_RUN_ROOT/original" \
  --data-root "$LGF_DATA_ROOT" --run-root "$LGF_RUN_ROOT/original" --check
```

Remove `--check` to execute the planned SE/CBAM x three datasets x three seeds grid sequentially. Add `--methods se --datasets coco_nw --seeds 42` for one initial run. Add `--reference-method baseline` to inherit baseline records. The preview verifies reference files and source/settings signatures; actual execution also checks all reference environments before training, then data and model protocols for each run. Rerunning resumes incomplete matching runs and skips completed matching runs.

Keep the extension as a sibling of `paper1_ugs_comparison_v1`. Adding extension files inside the original package changes its recursive Python source fingerprint. Preserve the exact extension version for an active run; `1.0.2` does not resume an extension `1.0.1` fingerprint.

## 7. Checkpoint selection and local validation

| File or rule | Meaning |
| --- | --- |
| `last.pth` / native detector last checkpoint | Latest committed training state, used for restart |
| `best_ap_small.pth` or `.pt` | EMA checkpoint selected by validation APsmall |
| `best_map.pth` or `.pt` | EMA checkpoint selected by overall validation AP |
| `protocol.json`, `data_audit.json`, `history.jsonl` | Settings, source/software/data identity, training and validation history |
| `best_*.metrics.json` | Selected checkpoint identity and measured validation metrics |
| `completed.json` | Planned epochs completed and protocol identity |
| `evaluation/epoch_*/predictions.json` | COCO-format validation detections from the revised original runner |

The legacy trainer saves its best-mAP checkpoint, while historical aggregation selects the highest logged APsmall. A saved best-mAP file does not establish peak-APsmall checkpoint availability. Verify the checkpoint against the history before claiming a selection rule. The manuscript's frozen gate controls use best-APsmall selection; condition and balanced-failure analyses use best-mAP selection. Their numeric differences are not interchangeable.

Re-evaluate a native checkpoint:

```bash
python run.py original -- evaluate --dataset coco_nw --seed 42 --level C3 \
  --data-root "$LGF_DATA_ROOT" --device cuda \
  --checkpoint "$LGF_RUN_ROOT/original/coco_nw/ugs_reimpl/seed_42/best_ap_small.pth" \
  --output "$LGF_RUN_ROOT/validation_rerun/coco_nw/ugs_reimpl/seed_42/best_ap_small"
```

For an archived notebook checkpoint, add `--original-checkpoint` and declare the verified rule with `--selection-note best_map` or `best_ap_small`. The supplied seed and level should match the saved experiment name. For native SE/CBAM checkpoint evaluation, use `python run.py se-cbam -- evaluate ...`; this installs the required construction adapter.

```bash
python run.py original -- summarize --selection best_ap_small \
  --run-root "$LGF_RUN_ROOT/original" --output "$LGF_RUN_ROOT/summary_best_ap_small"
```

`experiments/paper1_ugs_comparison_v1/compare.py` compares supplied validation `metrics.json` reports with matching seeds, data hashes, selection rules, and evaluation signatures. The detector runner's `compare` action checks its reports in the same way. Pass the complete seed sets. Summary scores are mean and sample standard deviation, not a significance test. Paired bootstrap uncertainty is a separate workflow documented elsewhere in this repository.

All AP reports use the 0-1 scale unless an output explicitly labels AP points. Validation remains validation even when a directory or notebook calls the operation “test.” Expanded condition analyses reuse selected validation checkpoints and are not independent held-out test estimates.

## 8. Official hidden-label tests and labelled holdouts

Use the shared preparation tool from the repository root:

```bash
python experiments/paper1_test_evaluation_v1/test_data.py download \
  --dataset coco --test-root "$LGF_TEST_ROOT"

python experiments/paper1_test_evaluation_v1/test_data.py prepare \
  --dataset coco_nw --test-root "$LGF_TEST_ROOT" --data-root "$LGF_DATA_ROOT"

python experiments/paper1_test_evaluation_v1/test_data.py weather \
  --condition mixed --weather-seed 42 --test-root "$LGF_TEST_ROOT" --data-root "$LGF_DATA_ROOT"
```

COCO preparation selects the official 20,288 test-dev images from the test2017 archive. The mixed weather command assigns one reproducible fog/rain/snow variant per scene before inference. `--condition fog`, `rain`, or `snow` produces full condition-specific manifests. These custom weather outputs are separate from standard unmodified-image benchmark scores and from the expanded 1,500-image validation workflow.

For ACDC, reuse an official RGB archive:

```bash
python experiments/paper1_test_evaluation_v1/test_data.py download \
  --dataset acdc --archive /absolute/path/rgb_anon_trainvaltest.zip --test-root "$LGF_TEST_ROOT"

python experiments/paper1_test_evaluation_v1/test_data.py prepare \
  --dataset acdc --test-root "$LGF_TEST_ROOT" --data-root "$LGF_DATA_ROOT"
```

Preparation inventories 2,000 adverse-condition images, 500 each for fog/night/rain/snow. A new authenticated download accepts `--url-file` containing the provider's current signed URL. Private URLs stay out of reports. Official image-ID/category metadata are required for ACDC submission packaging; filename sorting is not an official ID mapping.

Freeze a checkpoint selected on validation before export:

```bash
python run.py official-export -- export \
  --method lgf_original --train-dataset coco_nw --seed 42 --level C3 \
  --checkpoint /absolute/path/verified_original_lgf_seed42.pth \
  --selection best_map --selection-note 'Verified best validation mAP in the saved history' \
  --manifest "$LGF_TEST_ROOT/coco_nw_test.json" --data-root "$LGF_DATA_ROOT" \
  --device 0 --batch-size 4 --runtime-root "$LGF_RUN_ROOT/test_runtime" \
  --output "$LGF_RUN_ROOT/official_exports/coco_nw/lgf_original/seed_42"
```

The exporter supports `baseline`, `lgf_original`, `se`, `cbam`, `ugs_reimpl`, `yolo11m`, and `rtdetr-l`. Use the actual training dataset, seed, and frozen selection. Clear-to-weather transfer needs `--transfer` and the true clear-trained checkpoint. All exports use FP32 and EMA, with common 800/1333 geometry, confidence floor 0.05, and up to 100 detections per image. Native normalization and postprocessing remain model-specific.

For SE/CBAM checkpoints trained through the new extension, use its adapter explicitly:

```bash
python experiments/paper1_se_cbam_extension_nas_v1_0_2/evaluate_test_se_cbam.py \
  --package "$LGF_REPO/experiments/paper1_ugs_comparison_v1" \
  --eval-dir "$LGF_REPO/experiments/paper1_test_evaluation_v1" \
  export --method se --train-dataset coco_nw --seed 42 --level C3 \
  --checkpoint "$LGF_RUN_ROOT/original/coco_nw/se/seed_42/best_map.pth" \
  --selection best_map --selection-note 'Completed SE runner, best validation mAP' \
  --manifest "$LGF_TEST_ROOT/coco_nw_test.json" --data-root "$LGF_DATA_ROOT" \
  --device 0 --runtime-root "$LGF_RUN_ROOT/test_runtime" \
  --output "$LGF_RUN_ROOT/official_exports/coco_nw/se/seed_42"
```

The exporter copies selected weights into the output and commits prediction batches atomically. Add `--resume` to the identical command for interrupted exports. Changed data, source, checkpoint, or selection prevents reuse. `--limit 2` creates a software-only trial with no scientific score or server submission ZIP.

Official COCO/ACDC image downloads have hidden box labels. A successful prediction export is not measured AP. The standard server aggregates cover 80 COCO classes or eight ACDC classes; they are not the paper's six-class APsmall. Do not infer six-class small-object AP from an all-class aggregate or overall per-class AP. Official submission is a separate user action.

With genuine unused labelled holdout data, prepare and score separately:

```bash
python experiments/paper1_test_evaluation_v1/test_data.py prepare-labelled \
  --dataset coco_nw --images /absolute/path/unused_images \
  --annotations /absolute/path/unused_coco_annotations.json \
  --data-root "$LGF_DATA_ROOT" --output "$LGF_TEST_ROOT/coco_nw_labelled_test.json"
```

Export on the labelled manifest into a fresh output, then run:

```bash
python run.py official-export -- score --export /absolute/path/labelled_export
```

The scorer preserves all supplied evaluation images, original box geometry and area/crowd metadata, IoU thresholds 0.50:0.05:0.95, and maxDets 1/10/100. An evaluation-only copy normalizes annotation IDs to positive unique integers for COCOeval's unmatched sentinel. Undefined size groups return the evaluator's `-1` sentinel. Repartitioning scenes already used by a trained model does not establish an unused test set.

## 9. Source integrity and verification

`paper1_ugs_comparison_v1/core_bridge.py` verifies `vendor/original_core.py` before importing definitions. The recorded original notebook identity is `e6c4aa3fd6a95019eb11ee0a35a068e968b0959d30c40f47bfdc82fdd80408f1`; the extracted source hash is `64a662586fd3c5b71a850ca7db19079d13366dfabae7b764e02de1f49567ae95`. The loader suppresses notebook launch behavior and preserves the original definition cell.

The test-weather module records source notebook hash `03c78c152feb7d8278878951f5d808191011728a4f2849da7eee581fc6aae9bc` and generated module hash `eeed06e594b19f961e0cf1b167c8ff8de93af8a4a8c83206b2ededc1196e2482`. The full legacy notebook files have different bytes from those recorded originals. Their training definition cell and all retained weather-function definitions were checked for equal Python syntax trees during assembly; a filename comment differs in the training cell. Do not describe the legacy notebook files as byte-identical to the recorded source uploads.

Run package tests from their own directory to avoid collisions between same-named modules such as `data`, `utils`, and `models`:

```bash
(cd experiments/paper1_ugs_comparison_v1 && python -m unittest discover -s tests -v)
(cd experiments/paper1_reviewer34_v1 && python -m unittest discover -s tests -v)
(cd experiments/paper1_test_evaluation_v1 && python -m unittest discover -s tests -v)
(cd experiments/paper1_se_cbam_extension_nas_v1_0_2 && python environment_tests.py)
(cd experiments/paper1_se_cbam_extension_nas_v1_0_2 && python nas_path_tests.py)
```

The SE/CBAM native model self-test needs the full matching environment:

```bash
python experiments/paper1_se_cbam_extension_nas_v1_0_2/self_test.py \
  --package "$LGF_REPO/experiments/paper1_ugs_comparison_v1" \
  --eval-dir "$LGF_REPO/experiments/paper1_test_evaluation_v1" \
  --device cpu --output "$LGF_RUN_ROOT/se_cbam_self_test.json"
```

During assembly on 13 October 2026, the original/UGS suite passed all 25 tests and the detector suite passed all 29 tests, including synthetic YOLO training interruption/resume and the original LGF adapter. The SE/CBAM self-test passed all 13 tests without stubs or skips, including real ResNet attention gradients in both training phases and strict checkpoint-adapter reload. Its environment suite (6 tests) and NAS-path suite (9 tests) also passed. The shared test-export suite passed 23 workflow/regression tests, and its opt-in test subsequently passed all seven model adapters at 800/1333 geometry. All Python files in these four copied packages parsed. Consult the repository validation report for the complete release checks. Synthetic fixture scores and software smoke outputs are not research results.

Missing Torch, Torchvision, OpenCV, Albumentations, TensorBoard, pycocotools, tqdm, or Ultralytics causes import/dependency errors before training. Use the recorded compatible environment rather than bypassing version checks. GPU full-size training, CUDA/AMP behavior, convergence, and authenticated benchmark submission require validation on your research machine. Model source files are deliberately retained to protect existing checkpoint/resume identities. Release adapter fixes are described in [maintenance_fixes.md](maintenance_fixes.md), with the recovered versions preserved under `provenance/archived_source/`.
