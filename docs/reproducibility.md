# Reproducibility and source provenance

## Source recovery

The baseline GitHub snapshot is `aserawayneasera/LGF`, commit `0125590291f52469eb6427697cc2d319bcdce6fd` from 18 February 2026. Its files are retained under `legacy/`. The revision packages were recovered from saved source archives, rather than recreated from paper equations. `provenance/source_archives.json` records archive names and SHA-256 values. `provenance/source_files.json` maps redistributed files to recovered sources and records maintenance changes.

The frozen definition cell in each `vendor/original_core.py` has SHA-256:

```text
64a662586fd3c5b71a850ca7db19079d13366dfabae7b764e02de1f49567ae95
```

Loaders verify this identity before reconstruction. Historical notebooks include storage-only edits. Full notebook hashes therefore differ from the recorded source notebook identity, while the recovered definition-cell behavior is retained. Use the frozen core as the checkpoint schema authority.

This release adds a top-level dispatcher, annotation preflight checks, updated documentation, release metadata, and tests. The shared official-export SE/CBAM adapter also receives a documented correction. The scientific LGF, UGS, gate intervention, weather, and placement definitions retain their archived bytes.

## What reproduction requires

Exact numerical reproduction needs the historical annotation snapshots, selected checkpoint weights, prediction files, run protocols, software versions, and selection histories. Source alone does not recover these inputs. In particular, peak logged APsmall rows and best-mAP checkpoint files do not necessarily refer to the same epoch.

Keep these files together for each run:

1. Training and validation annotations, category mapping, and image/source-scene manifest.
2. `protocol.json`, `data_audit.json`, full training history, and source/environment hashes.
3. Selected EMA checkpoint and its `best_ap_small` or `best_map` identity.
4. Actual COCO predictions and metric report, including hashes and evaluation settings.
5. Condition assignment evidence, output manifests, seeds, and analysis settings.

The source release excludes those large artifacts. It supports generating them and rerunning the archived analyses once you supply the inputs.

## Result families

| Family | Selection and interpretation |
|---|---|
| Retained original AP/AR tables | Original logs/selection family, preserve reported values |
| Revised training runner | Same frozen model definitions, operational changes including FP32 and augmentation seeding, a new run |
| New detector table and bootstrap | APsmall-selected validation checkpoints and actual archived predictions |
| Gate interventions | APsmall-selected LGF, trained branches and weights frozen |
| Weather condition tables | Best-mAP checkpoint reused across all conditions in each dataset |
| Failure diagnostics | Best-mAP predictions, confidence 0.25 and correct-class IoU 0.50 for primary examples |
| Clear-to-weather transfer | Fixed clear-trained checkpoint, no retraining; identify source selection history |
| C4 pooled weather | Select on original 500-image fog subset, score pooled 1,500 weather images |
| Official test exports | Hidden ground truth, prediction files only unless independent labels/server scores are supplied |

The fixed-gate interventions do not train a separate fixed-fusion network. Unscaled summation changes residual magnitude. Gate maps display learned weights and do not prove causal weather adaptation. Detector comparisons describe complete models with different native training recipes.

## Annotation lineage

Historical paper overview counts and later detection-diagnostic counts differ. Preserve the actual archived annotations for each workflow and record their hashes. Do not force regenerated counts to match a transcribed manuscript table by editing annotations.

The recovered `balanced_failure_analysis_v1.zip` resolves the diagnostic support: its `image_counts.csv` contains exactly 500 unique synthetic-weather image IDs, all with fog filenames. Its manifest names `mini_val2017_weather-500k6c.json`. The latest manuscript describes 1,500 images in that diagnostic subsection, while the saved diagnostic export uses 500 fog images. This is a documented manuscript/export support mismatch. The separate condition-AP and pooled C4 weather workflows score 1,500 images. Do not relabel the saved 500-image diagnostic counts as a pooled 1,500-image analysis. Regenerating full-weather failure diagnostics requires saved full-weather predictions and their corresponding 1,500-image annotations, and produces a separately identified result. Aggregate evidence is recorded in `provenance/diagnostic_support.json` without distributing raw predictions or per-image records.

The saved dataset-preparation notebook records a weather-off pass. Its weather-generation functions are available, but the precise historical weather-on invocation is not recovered from notebook outputs. Preserve your already prepared weather dataset for published-result reruns. Regeneration is a separately recorded data preparation run.

COCO source-scene independence requires grouping weather variants by source ID. Byte overlap alone does not establish scene independence. ACDC condition groups contain different scenes and class mixtures. Condition AP differences do not isolate weather as the only changing variable.

## Bootstrap inputs

The archived bootstrap implementation requires strictly positive integer ground-truth annotation IDs. COCO matching uses zero as an unmatched sentinel. Run `tools/audit_inputs.py --require-positive-ann-ids` first. If an input has ID zero, preserve the original file and create an explicitly documented positive-ID annotation copy. Annotation IDs are separate from image IDs, prediction records, and category IDs. Do not silently alter archived inputs or claim unchanged numerical reproduction.

The paper's 10,000-resample intervals describe selected-validation detector rankings. They do not establish significance for the retained original RetinaNet-to-LGF improvement or an independent test set. Paired sampling preserves the same images for both detectors; read the implementation for its exact seed/image sampling and grouping behavior. Additional source-scene cluster sampling is not silently substituted in this release.

## Placement and cost

The archived model code retains historical run-label routing. The later semantic-remapping package `paper1_placement_checks_v2` is excluded from this release's manuscript reproduction path. Inspect runtime routes for any new experiments and report both saved labels and tensor routes.

The reported C3/C4 parameters are 35.155M/43.825M. Reference-image convolution/linear GMACs are 222.735/222.719. The separate legacy cost comparison reports +2.91M parameters, +2.96G FLOPs, and 10.98 to 11.64 ms/image at batch 16. These are recorded reference values, not outputs measured in this release build. Similar GMACs do not imply equal detector latency.

## External API references

- Torchvision RetinaNet factory and detection input contract: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html
- Torchvision RetinaNet implementation: https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/retinanet.html
- Ultralytics prediction result tensor fields: https://docs.ultralytics.com/reference/engine/results/
- Ultralytics RT-DETR implementation documentation: https://docs.ultralytics.com/models/rtdetr/

These current references describe public interfaces. The recovered source and pinned package versions define this research release's implementation.
