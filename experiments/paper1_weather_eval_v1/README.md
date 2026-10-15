# Paper 1: weather-condition evaluation

Version 1.0.0, 21 September 2026.

This add-on reads the saved models from `paper1_recovery_1_0_1/paper1_all_codes_with_test_evaluation`. It imports the existing `paper1_test_evaluation_v1/models.py` adapter. It does not replace any training, architecture, data-generation, or test-export code. It never rewrites source annotations or images. Run after the selected training runs finish.

## What this evaluates

The defaults use the existing labelled **validation** data. They do not turn validation data into an independent test set.

- COCO-Weather: conditions present in the selected annotation JSON, usually fog, rain and snow.
- ACDC: conditions present in the selected annotation JSON, drawn from fog, night, rain and snow.
- `ALL`: a separate pooled COCO evaluation over all selected images. This is not an average of the weather rows.

One fixed checkpoint processes the full evaluation split. The predictions are then scored separately for each condition. There is no training, checkpoint selection per condition, threshold tuning, condition classifier, or synthetic-weather generation.

## Where the labels come from

The retrieved `1b.create_weather_aug_coco_dataset.ipynb` writes `weather` into each COCO image record and uses suffixes such as `_fog.jpg`. The evaluator checks metadata and the documented suffix against each other. It does not assume the directory name `weather-mini` guarantees three conditions in the JSON.

For ACDC, the original layout includes condition and split, for example:

```text
rgb_anon/fog/val/GOPR0122/GOPR0122_frame_000001_rgb_anon.png
```

Converted copies often have flat filenames. The previous project audit documents the following exact Roboflow conversion:

```text
GOPR0122_frame_000001_rgb_anon_png.rf.<32 hexadecimal characters>.jpg
```

The add-on joins that complete frame identity to original paths. It does not infer conditions from a sequence number alone or from appearance. Metadata/path conflicts and unmatched images stop evaluation. Source frame IDs, labels and evidence are written into `image_conditions.csv`.

The script automatically checks for the previously downloaded archive:

```text
/tmp/asera/acdc_direct/rgb_anon_trainvaltest.zip
```

It reads ZIP filenames, not image payloads, and does not extract the archive. A different source is supplied with `--acdc-source`. Accepted sources are the original RGB ZIP, an extracted original directory, the official combined detection JSON, or the official `gt_detection_trainval.zip`. A test-only extraction will not cover validation frames.

The archive is used only to recover condition metadata. **Inference uses the existing converted evaluation images and their existing ground-truth boxes.** Original full-resolution images are not substituted for the converted images.

## Save the add-on separately

Save `paper1_weather_eval.py` as:

```text
/nas.dbms/asera/paper1_weather_eval.py
```

No files need to be copied into the recovery package.

Set your paths:

```bash
P1PY=/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python
WEATHER=/nas.dbms/asera/paper1_weather_eval.py
PKG=/tmp/asera/NEW/paper1_recovery_1_0_1/paper1_all_codes_with_test_evaluation
OUT=/tmp/asera/paper1_original/weather_evaluation_v1
```

Use the same Python environment which already runs the original/UGS code. Override `P1PY` with that executable when its location differs. The launcher preserves virtual-environment symlinks rather than resolving them to the base interpreter.

## 1. Audit the two validation datasets

```bash
"$P1PY" -u "$WEATHER" audit \
  --datasets coco_weather acdc \
  --output-root "$OUT"
```

The audit requires no model loading and no GPU. It prints image and small-object counts for each condition, plus unknown labels, conflicts and missing files. It writes:

```text
audits/coco_weather/condition_audit.json
audits/coco_weather/image_conditions.csv
audits/coco_weather/condition_image_ids.json
audits/acdc/condition_audit.json
audits/acdc/image_conditions.csv
audits/acdc/condition_image_ids.json
```

Proceed with the intended condition comparison when `unknown=0`, `conflicts=0`, and `missing files=0`. Verify each intended condition has nonzero support. If a dataset contains only fog, the tool reports only fog; it does not manufacture rain/snow rows or change the selected dataset.

To point to another original ACDC copy:

```bash
"$P1PY" -u "$WEATHER" audit \
  --datasets acdc \
  --acdc-source /your/existing/rgb_anon_trainvaltest.zip \
  --output-root "$OUT"
```

Pass the same `--acdc-source` to evaluation. A supplied metadata CSV uses exact `file_name,condition` columns and is accepted through `--condition-map` for one dataset at a time. Fill it only from verified provenance, never from random assignment or model predictions.

## 2. Check model paths without inference

```bash
"$P1PY" -u "$WEATHER" evaluate \
  --datasets coco_weather acdc \
  --methods baseline lgf_original \
  --seeds 42 1337 2025 \
  --selection best_map \
  --package-root "$PKG" \
  --legacy-python "$P1PY" \
  --output-root "$OUT" \
  --check
```

`--check` audits labels and checks filesystem paths/job construction. It does not load checkpoint tensors, verify training completion, import GPU dependencies, or test GPU memory. Any missing job stops the grid before inference.

The checkpoint sources follow the previously supplied test-grid launcher:

| Method | Default checkpoint source |
|---|---|
| baseline, SE, CBAM | Original notebook, `/tmp/asera/paper1_original/NEW` |
| lgf_original, ugs_reimpl | `/tmp/asera/paper1_original/ugs_comparison/runs` |
| yolo11m, rtdetr-l | `/tmp/asera/paper1_original/reviewer34/runs` |

For a baseline trained through the UGS runner, add `--baseline-source ugs`. For LGF trained through the original notebook, add `--lgf-source notebook`. These switches refer to the training runner, not a change of detector architecture.

Original notebook checkpoint names have the form:

```text
BEST_acdc_C3_baseline_s42_epoch_<epoch>_map_<score>.pth
```

The original notebook keeps improving validation-mAP checkpoints. The last improving epoch from a single completed run is selected, including rounded score ties. Decreasing scores or duplicate epochs cause an ambiguity error. Set `--original-root` to the correct single-run storage directory, or use `--checkpoint` for one method/dataset/seed.

## 3. Run baseline and LGF, both datasets, all three seeds

```bash
"$P1PY" -u "$WEATHER" evaluate \
  --datasets coco_weather acdc \
  --methods baseline lgf_original \
  --seeds 42 1337 2025 \
  --selection best_map \
  --package-root "$PKG" \
  --legacy-python "$P1PY" \
  --device 0 \
  --batch-size 4 \
  --output-root "$OUT"
```

Jobs run sequentially. Each model runs inference once per evaluation split. The saved detections support all weather rows. Repeating the identical command skips completed runs after checking data, checkpoint, adapter and prediction fingerprints. An interrupted incomplete inference run restarts that job. Changed data/settings require a new output root. Use a different output root if changing batch size.

For a first single-model run, replace the method/dataset/seed arguments with:

```bash
--datasets coco_weather --methods lgf_original --seeds 42
```

An explicit checkpoint example:

```bash
"$P1PY" -u "$WEATHER" evaluate \
  --datasets acdc --methods lgf_original --seeds 42 \
  --checkpoint /tmp/asera/paper1_original/ugs_comparison/runs/acdc/lgf_original/seed_42/best_map.pth \
  --selection best_map --package-root "$PKG" --legacy-python "$P1PY" \
  --device 0 --output-root "$OUT"
```

## Additional saved methods

The original controls use the same legacy environment:

```bash
"$P1PY" -u "$WEATHER" evaluate \
  --datasets coco_weather acdc --methods se cbam ugs_reimpl \
  --seeds 42 1337 2025 --selection best_map \
  --package-root "$PKG" --legacy-python "$P1PY" \
  --device 0 --output-root "$OUT"
```

Recent detectors use their existing separate environment:

```bash
P34PY=/tmp/asera/paper1_reviewer34_env/bin/python

"$P1PY" -u "$WEATHER" evaluate \
  --datasets coco_weather acdc --methods yolo11m rtdetr-l \
  --seeds 42 1337 2025 --selection best_map \
  --package-root "$PKG" --legacy-python "$P1PY" --recent-python "$P34PY" \
  --device 0 --output-root "$OUT"
```

All methods retain the model loading, class mapping, original EMA policy and inference implementation in your existing adapter. The adapter checks original RetinaNet settings of min/max input size 800/1333, confidence 0.05, NMS 0.5 and 100 detections per image. No AMP or new preprocessing is added.

## Checkpoint-selection rule

`best_map` is the default because it matches the supplied independent-test grid and original notebook checkpoint naming. Keep the same predeclared selection rule across methods and all weather conditions.

`--selection best_ap_small` is supported for the UGS/recent runners when those matching checkpoint files exist. Original notebook `BEST_*_map_*.pth` files are not relabelled as APsmall-selected checkpoints. No checkpoint is selected using an individual weather score. Do not mix mAP-selected rows with APsmall-selected rows and describe them as one common selection protocol.

## Output and interpretation

Main outputs:

```text
weather_evaluation_v1/
  audits/...
  jobs/...
  runs/coco_weather_to_coco_weather/best_map/<method>/seed_<seed>/
    run_protocol.json
    predictions_coco.json
    evaluated_image_ids.json
    weather_metrics.json
    weather_metrics.csv
    evaluation.log
  runs/acdc_to_acdc/best_map/<method>/seed_<seed>/...
  summary/
    weather_per_seed.csv
    weather_mean_std.csv
    lgf_minus_baseline.csv
```

`weather_per_seed.csv` contains all 12 COCO AP/AR metrics plus image, object and small-object support. `weather_mean_std.csv` uses sample standard deviation across the available training seeds. Check `n_seeds`; an incomplete grid is not presented as a three-seed result. Undefined metrics are `NA`, not zero and not averaged as COCO's sentinel value -1.

`lgf_minus_baseline.csv` contains paired mean differences and sample SD across matching seeds. Its differences use **AP points**, `(LGF - baseline) * 100`. Ordinary AP columns use 0-1. Seed SD is not a bootstrap confidence interval or a significance test. No bootstrap is implemented here.

The evaluator keeps every image in each condition, including images with zero predictions and images without small objects. It recomputes COCO AP on the full condition subset using `COCOeval.params.imgIds`. It does not average per-image AP. The support count for small objects uses non-crowd annotation area in the evaluator's inclusive [0, 1024] range. The reported number of categories with small-object ground truth helps interpret sparsely populated subsets.

Rebuild summary files without inference:

```bash
"$P1PY" "$WEATHER" summarize --output-root "$OUT"
```

## Separate clear-to-weather evaluation

This is an additional transfer analysis, not the same as the matched-condition breakdown. Use an original-COCO-trained checkpoint unchanged:

```bash
"$P1PY" -u "$WEATHER" evaluate \
  --datasets coco_weather --train-dataset coco_nw \
  --methods baseline lgf_original --seeds 42 1337 2025 \
  --selection best_map --package-root "$PKG" --legacy-python "$P1PY" \
  --device 0 --output-root /tmp/asera/paper1_original/weather_transfer_v1
```

The script records the training dataset and maps classes by name. It does not certify historical independence or prove that the original COCO training data were all naturally clear. Existing `coco_nw` is your code's dataset label. Check source-scene independence and describe the training data accurately before making unseen-weather claims.

## Validation versus independent testing

The default annotation paths are:

```text
/tmp/asera/paper1_original/DATASET/COCO/weather-mini/annotations/mini_val2017_weather-500k6c.json
/tmp/asera/paper1_original/DATASET/ACDC-1/ACDC-1-NEW/annotations/mini_val.json
```

These rows are labelled `validation`. The audit reports original ACDC split labels when recoverable and checks recognizable current train/evaluation frame overlap, but it does not audit every historical training/selection source. Do not describe the outputs as independent test evidence merely because the audit finds no overlap.

A separately established labelled holdout is supplied through `--annotations`, `--images` and `--split heldout_labelled` for one dataset. This is a reporting label, not automatic certification of independence. Official hidden-label test image-info JSONs do not contain ground truth and cannot be scored locally. Keep using `evaluate_test.py` for official test exports. That existing script already has condition-specific scoring for a genuinely labelled test manifest; its test-exclusion checks should not be bypassed to pass validation images as tests.

If a condition is absent from the current COCO annotation JSON, first document the absence. Existing additional weather variants require a separate, explicitly documented evaluation annotation view with verified source correspondence. Do not replace the original JSON, randomly relabel images, or claim the earlier pooled score included absent conditions.

## Dependencies and validation status

Use the working environments from the recovery package. Inference needs the same Torch/Torchvision, OpenCV, NumPy, Pillow and pycocotools dependencies as the existing evaluation code. The original adapter enforces Torch 2.10.x and Torchvision 0.25.x. Recent detectors also require the existing Ultralytics environment. This add-on does not install or update any packages.

Twenty synthetic software checks passed in the authoring environment. They cover label recovery, archive indexing, refusal of unknown/conflicting labels, no source edits, hidden-label rejection, checkpoint discovery, APsmall/mAP selection mismatches, summary calculations and CLI dry-run construction. Two numerical COCO-metric tests were skipped because pycocotools was unavailable in that environment. No real saved checkpoint or GPU inference was tested here. The user's current dataset JSONs, archive and checkpoint files were not mounted in the authoring environment. See `validation_tests.txt` and `validation_tests.xml`.

To run all tests in an environment with pycocotools and pytest already installed:

```bash
cd /path/to/paper1_weather_eval_v1
"$P1PY" -m pytest -q tests
```

## Implementation evidence

The source interfaces were checked against retrieved project files, not inferred only from conversation history:

- `models.py`: `load_model(args, class_names)`, `rgb_tensors(records, image_root, device)`, EMA and metadata checks.
- `test_data.py`: original dataset path definitions and condition-bearing ACDC paths.
- `run_paper1_test_grid.py`: notebook/UGS/recent checkpoint layout and selection.
- `1b.create_weather_aug_coco_dataset.ipynb`: explicit image weather field and suffix convention.
- `00_audit_existing_datasets_revised.py`: exact original-frame join for flattened ACDC names.
- `evaluate_test.py`: labelled-test condition scoring versus hidden-label export.

Source hashes for the directly materialized interface files are recorded in `source_contract.json`. The new evaluator imports the user's local adapter at runtime, rather than bundling an older copy of the detector implementation.

External metric reference: COCO API, `PythonAPI/pycocotools/cocoeval.py`, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py . ACDC dataset reference: https://arxiv.org/abs/2104.13395 .
