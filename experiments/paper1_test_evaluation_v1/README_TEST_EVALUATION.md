# Paper 1: test downloads and evaluation

Prepared 15 September 2026. This update adds test inference for the original RetinaNet, LGF, SE, CBAM, the supplied UGS reimplementation, YOLO11m, and RT-DETR-L. The original training notebooks and Python training/model files are unchanged. A separate folder preserves their source hashes and existing restart checks.

## Read this before choosing the evaluation route

| Dataset | Download/preparation | Available result |
| --- | --- | --- |
| COCO test-dev2017 | Download the official test images and image-info ZIP. Evaluate all 20,288 test-dev images. | COCO-format JSON and submission ZIP. Test labels are private, so local AP is unavailable. |
| Synthetic COCO weather | Apply the original uploaded weather functions to the same frozen test-dev images. | A separately labelled custom-weather submission. This is not an official weather dataset download. |
| ACDC adverse test | Use the RGB archive from your ACDC account. Extract only the 2,000 adverse test images. | Predictions by filename, followed by official submission ZIPs when official image/category metadata are supplied. |
| A separately supplied labelled holdout | Verify unused images against the original splits. | Local six-class AP, APsmall, AP50, AP75, medium/large AP, and AR. |

Official image-info files do not contain bounding-box ground truth. The code never passes an unlabelled test set to COCOeval and never reports fabricated zero AP for missing labels.

Your detectors predict six classes: person, bicycle, car, motorcycle, bus, and truck. Standard COCO scores cover 80 classes; ACDC object detection covers eight. A standard server aggregate is not the paper's six-class AP. Obtain explicit scores for your six classes before forming the paper's test comparison table. Six-class APsmall needs per-class small-object scores or a custom six-class evaluation from the benchmark provider. Overall per-class AP alone does not provide APsmall.

The official COCO test includes images without your target classes. Keep all official test images. Do not select a favourable subset by looking at predictions. These test scores also have a different image distribution from the original six-class validation subset.

No server submission happens automatically. The code prepares reviewable files and records their hashes. Uploading benchmark results is a separate user action.

## 1. Install the combined update

Copy `paper1_all_codes_with_test_evaluation.zip` to `/tmp/asera/NEW` and extract:

```bash
cd /tmp/asera/NEW
unzip -o paper1_all_codes_with_test_evaluation.zip
cd paper1_test_evaluation_v1
sha256sum -c CHECKSUMS.sha256

export P1PY=/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python
export P34PY=/tmp/asera/paper1_reviewer34_env/bin/python
export P1_TEST_ROOT=/tmp/asera/paper1_original/TEST_DATASET
export P1_TEST_RUNS=/tmp/asera/paper1_original/test_evaluations
```

Use `P1PY` for original-model/UGS exports, or `P34PY` for all methods if the separate newer-detector environment is installed. The previous packages contain installation commands. Requirements remain torch 2.10.x, torchvision 0.25.x, NumPy, Pillow, OpenCV, Albumentations 2.0.8, and pycocotools. Newer models additionally require the existing pinned Ultralytics 8.3.253 environment.

Every dataset, downloaded archive, derived image, frozen checkpoint copy, and result defaults to `/tmp/asera`. Transfer finished results to your chosen durable destination before temporary storage is cleared.

The ZIP has four sibling folders:

```text
paper1_original_tmp/
paper1_ugs_comparison_v1/
paper1_reviewer34_v1/
paper1_test_evaluation_v1/
```

Keep this layout. The test runner loads model definitions from the adjacent packages. Paths also have explicit `--lgf-package` and `--recent-package` overrides.

## 2. Download and prepare COCO test-dev

The image archive contains all 40,670 COCO test2017 images. Official metadata selects the 20,288 test-dev images. The supplied commands do not use the test-challenge partition for reporting.

```bash
"$P1PY" test_data.py download --dataset coco --test-root "$P1_TEST_ROOT"

"$P1PY" test_data.py prepare --dataset coco_nw --test-root "$P1_TEST_ROOT"
```

The download resumes a matching `.part` file using the server ETag and byte ranges. Re-running after completion verifies/reuses the files. Extraction rejects traversal paths and checks ZIP CRCs. The metadata archive and test-dev JSON have pinned SHA-256 values from the official release. The image archive's observed hash is recorded after download/extraction.

The official archive is approximately 6 GB, in addition to extracted images and generated weather images. A metadata-only connectivity check downloads about 1.1 MB:

```bash
"$P1PY" test_data.py download --dataset coco --metadata-only --test-root "$P1_TEST_ROOT"
```

If `test2017.zip` already exists, add `--archive /absolute/path/test2017.zip` to the full download command. Existing training/validation images are required during preparation to check image hashes and original source IDs.

Output: `/tmp/asera/paper1_original/TEST_DATASET/coco_nw_test.json`.

## 3. Prepare synthetic weather test images

```bash
"$P1PY" test_data.py weather --condition mixed --weather-seed 42 --test-root "$P1_TEST_ROOT"
```

Output: `coco_weather_mixed_test.json` in the test root. Each original test image gets one reproducibly assigned fog, rain, or snow transformation. The seed is derived from the fixed weather seed and original image ID. Assignment is fixed before model inference and shared by every method. Original image IDs and box geometry remain unchanged.

The functions `add_fog`, `add_rain`, `add_snow`, and their helpers are copied without changes from your uploaded weather notebook. All copies of those functions in the notebook have identical syntax trees. The new test image seed assignment is documented in the manifest. This does not regenerate or replace any existing training/validation weather images.

For full condition-specific versions, run the same command with `--condition fog`, `rain`, or `snow`. Each condition produces a separate manifest and submission. The mixed set has disjoint condition groups; per-condition full sets use the same base images.

When submitting weather predictions, describe the synthetic input transformations. Do not label weather scores as standard unmodified-image COCO benchmark results. Availability of custom weather/six-class server reporting needs confirmation with the benchmark provider.

## 4. Download or reuse ACDC test images

ACDC issues downloads through a signed-in account at [the official download page](https://acdc.vision.ee.ethz.ch/download). The public site does not offer a permanent unauthenticated archive URL. Do not use an invented URL or an unrelated dataset mirror.

You previously downloaded `rgb_anon_trainvaltest.zip`. If that archive is available on the server, reuse the full archive:

```bash
export P1_ACDC_ARCHIVE=/absolute/path/to/rgb_anon_trainvaltest.zip

"$P1PY" test_data.py download --dataset acdc --archive "$P1_ACDC_ARCHIVE" \
  --test-root "$P1_TEST_ROOT"
```

For a new download, sign in to ACDC and obtain the RGB image download. Save the issued HTTPS link in `/tmp/asera/acdc_download_url.txt`, then run:

```bash
"$P1PY" test_data.py download --dataset acdc \
  --url-file /tmp/asera/acdc_download_url.txt --test-root "$P1_TEST_ROOT"
```

Add `--md5 VALUE_FROM_ACDC` to either command when you have the checksum displayed on the official page. The URL stays outside logs and saved reports. If a link expires, download the archive through your account and use `--archive`. A partial file from a different signed URL is not appended blindly.

Extracting a full RGB archive writes only `rgb_anon/{fog,night,rain,snow}/test/*_rgb_anon.png`. It excludes training, validation, and normal-reference images.

```bash
"$P1PY" test_data.py prepare --dataset acdc --test-root "$P1_TEST_ROOT"
```

Preparation requires exactly 500 test images per adverse condition, 2,000 total. Output: `acdc_test.json` in the test root. For an already extracted archive, add `--images /path/to/directory/containing/rgb_anon`.

This inventory uses filenames as internal IDs. Internal IDs are never presented as official ACDC server IDs. The public submission instructions specify COCO-format JSON but do not establish the filename-to-server-ID mapping. This package therefore requires official test image metadata before building an ACDC submission. See step 6.

## 5. Export a selected model

Identify the selected checkpoint for the correct training dataset and seed. Original notebook files save best mAP, while the submitted tables select peak APsmall. Verify which checkpoint you have. Use the same selection rule for every compared method, fixed on validation before test inference.

Example for original LGF trained on unmodified COCO, seed 42:

```bash
export P1_TEST_CKPT=/absolute/path/to/verified_original_lgf_coco_nw_seed42.pth

"$P1PY" evaluate_test.py export \
  --method lgf_original --train-dataset coco_nw --seed 42 --level C3 \
  --checkpoint "$P1_TEST_CKPT" --selection best_map \
  --selection-note "Verified best validation mAP in the original training log" \
  --manifest "$P1_TEST_ROOT/coco_nw_test.json" \
  --device 0 --batch-size 4 \
  --output "$P1_TEST_RUNS/coco_nw/lgf_original/seed_42"
```

Use these values for each method:

| Model | `--method` | Python environment |
| --- | --- | --- |
| Original RetinaNet | `baseline` | `P1PY` or `P34PY` |
| Original proposed LGF | `lgf_original` | `P1PY` or `P34PY` |
| Original SE control | `se` | `P1PY` or `P34PY` |
| Original CBAM control | `cbam` | `P1PY` or `P34PY` |
| Supplied UGS adaptation | `ugs_reimpl` | `P1PY` or `P34PY` |
| YOLO11m | `yolo11m` | `P34PY` |
| RT-DETR-L | `rtdetr-l` | `P34PY` |

For YOLO11m, for example:

```bash
"$P34PY" evaluate_test.py export \
  --method yolo11m --train-dataset coco_nw --seed 42 \
  --checkpoint /tmp/asera/paper1_original/reviewer34/runs/coco_nw/yolo11m/seed_42/best_map.pt \
  --selection best_map --manifest "$P1_TEST_ROOT/coco_nw_test.json" \
  --device 0 --batch-size 4 --output "$P1_TEST_RUNS/coco_nw/yolo11m/seed_42"
```

For weather-trained models, use `--train-dataset coco_weather`, their actual weather checkpoint, and `--manifest "$P1_TEST_ROOT/coco_weather_mixed_test.json"`. For ACDC-trained models, use `--train-dataset acdc`, their ACDC checkpoint, and `--manifest "$P1_TEST_ROOT/acdc_test.json"`. Change the output folder and seed accordingly. Do not pass a COCO-trained checkpoint as an ACDC-trained model.

A deliberate clear-to-weather or COCO-to-ACDC transfer experiment uses the actual training dataset plus `--transfer`. The report records this as a separate experiment.

All methods use FP32, EMA weights, batch 4 by default, shorter side 800, longer-side cap 1333, confidence floor 0.05, and at most 100 detections per image. The original LGF transform and original C3 mapping remain unchanged. Newer models retain their native normalization and postprocessing. This does not make their training recipes identical.

The runner copies the selected checkpoint into the result folder before inference. It writes atomic batches and progress messages. Re-run the same command with `--resume` after interruption. Completed batches are reused. Data, source, checkpoint, or selection changes block reuse.

For a software-only trial, add `--limit 2` and use a separate output folder. Partial exports do not produce a server ZIP or research metrics.

Long runs work inside tmux:

```bash
tmux new -s paper1_test
```

Inside tmux, set the variables above, change to the test toolkit directory, and run the export command. Detach with Ctrl+B, then D. Reattach with `tmux attach -t paper1_test`.

## 6. Submit the exported files

COCO exports contain:

```text
export_report.json
predictions_by_filename.json
predictions_coco.json
submission.zip
frozen_checkpoint.pth or frozen_checkpoint.pt
progress.json
```

The ZIP contains one results JSON using official COCO image/category IDs. Use the COCO test-dev object-detection submission interface. Record the returned result and its submission ID alongside the export. Respect the server's current submission limits. Do not tune checkpoints, thresholds, or weather generation using test feedback.

ACDC initially produces predictions by filename. Obtain an official test image-info JSON containing `images` with `id`, `file_name`, `width`, and `height`. Supply official category definitions in the same file or in a separate original ACDC category JSON. Bounding-box test labels are not required for submission packaging.

Look for suitable metadata among your downloaded ACDC files:

```bash
"$P1PY" test_data.py find-acdc-info \
  --manifest "$P1_TEST_ROOT/acdc_test.json" \
  --metadata-root /absolute/path/to/downloaded_ACDC_metadata
```

This prints candidates, not proof of official origin. If no matching file exists, obtain the image-ID mapping from the ACDC provider. The code does not infer IDs by sorting filenames, reuse training IDs, or assign guessed category numbers.

Once official metadata are available:

```bash
"$P1PY" evaluate_test.py package-acdc \
  --export "$P1_TEST_RUNS/acdc/lgf_original/seed_42" \
  --image-info /absolute/path/to/official_acdc_test_image_info.json \
  --categories-json /absolute/path/to/original_acdc_category_definitions.json \
  --metadata-provenance "Official ACDC download, package name and date"
```

Omit `--categories-json` if the image-info file already contains the official categories. The command verifies every filename and image dimension, remaps labels by class name, and creates `all.zip`, `fog.zip`, `night.zip`, `rain.zip`, and `snow.zip`. Each ZIP contains one COCO-format JSON and respects the documented 100 MB detection-submission limit.

In ACDC's submission form, choose Object Detection and the matching condition task. State the actual training data, including use of ACDC labels and COCO pretraining. Files are created locally; the script does not submit or publish them.

Server packaging was validated against the public format and synthetic ID-mapping cases. No authenticated ACDC metadata or live submission was available in this workspace. Successful acceptance and six-class APsmall availability therefore remain unverified.

## 7. Local six-class scoring, only with labelled holdout data

This route is separate from the hidden-label official tests. Use images excluded from fine-tuning, validation selection, prior development, and relevant pretraining. Repartitioning data already used by a trained checkpoint does not make those images unseen.

```bash
"$P1PY" test_data.py prepare-labelled --dataset coco_nw \
  --images /absolute/path/to/unused_images \
  --annotations /absolute/path/to/their_real_coco_annotations.json \
  --output "$P1_TEST_ROOT/coco_nw_labelled_test.json"
```

Pass this manifest to the same export command with a fresh output directory. Then:

```bash
"$P1PY" evaluate_test.py score --export "$P1_TEST_RUNS/coco_nw_labelled/lgf_original/seed_42"
```

`test_metrics.json` reports six-class metrics on the 0–1 AP scale. The scorer retains all supplied evaluation images, original ground-truth boxes, area/crowd fields, IoUs 0.50:0.05:0.95, and maxDets 1/10/100. Annotation record IDs are normalized to positive unique IDs in an evaluation-only copy because COCOeval uses zero as its unmatched sentinel. Source annotation files remain unchanged. Missing object-size groups return the COCO sentinel -1. Condition-specific scores are included when the manifest has verified condition assignments.

Combine matching local test reports:

```bash
"$P1PY" evaluate_test.py compare --reports \
  /absolute/path/lgf_seed42/test_metrics.json \
  /absolute/path/yolo_seed42/test_metrics.json \
  --output "$P1_TEST_RUNS/comparisons/coco_nw_labelled_seed42"
```

Pass all six reports for two methods across seeds 42, 1337, and 2025. The comparison checks common test/evaluation signatures, selection rules, seed sets, and training configuration consistency within each method. Export reports and official all-class server aggregates are rejected as local six-class metrics. Mean/std do not constitute a significance test.

## Evaluation independence and interpretation

The preparation audit checks original COCO clear/weather or ACDC training and validation files using both byte hashes and parsed source-image/frame IDs. Add `--exclude-json /path/to/earlier_experiment_images.json` during preparation for earlier development sets. Repeat for multiple files. The tool checks those additional IDs without claiming a complete history audit.

This does not prove geographical independence or complete pretraining provenance. The official ACDC split provides a geographical separation, but your original files must be checked against that release. The script preserves the submitted training setup and does not silently correct dataset counts, remap C3, or retrain any detector.

Keep your original validation tables labelled as validation results. Add a separate final test table only after obtaining comparable scores for the same six classes. Never compare original LGF validation AP against another method's test AP. Keep the original RetinaNet versus RetinaNet+LGF experiment for the module's contribution; newer-detector comparisons assess complete systems.

## Sources

- [COCO official download page](https://cocodataset.org/#download), [official page source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm).
- [COCO result format](https://cocodataset.org/#format-results), [test guidelines](https://cocodataset.org/#guidelines).
- [ACDC download](https://acdc.vision.ee.ethz.ch/download), [submission instructions](https://acdc.vision.ee.ethz.ch/submit).
- [ACDC paper, dataset splits and evaluation server](https://arxiv.org/html/2104.13395v5).

日本語：公式テスト画像には正解ラベルが含まれていません。この更新では画像を準備し、保存済みモデルの予測結果を公式サーバーへ提出するためのファイルを作ります。ACDCのダウンロードにはログインが必要です。提出ファイルの作成には公式の画像ID情報も必要です。学習コードは変更していません。論文用の比較では、全手法の同じ6クラスの評価結果を使ってください。
