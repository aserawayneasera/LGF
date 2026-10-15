> Repository integration: use [diagnostics.md](../../docs/diagnostics.md) for portable commands and input provenance. The instructions below describe the recovered historical server workflow. Scientific Python/model sources retain their original bytes. Shell launchers use repository-relative paths and documented environment overrides.

# Paper 1 balanced detection failures, v1.0.1

This extension addresses Reviewer Comment 15 using saved validation predictions.
It reads the existing selected-checkpoint reports. It performs no training,
loads no model weights, and needs no GPU. It leaves the training packages intact.

The analysis runs locally on your server, where your real prediction exports,
labelled annotations and images are stored. The bundled validation uses synthetic
fixtures. The bundle contains no new research counts or real failure figures.

## Run on your current server

Save `paper1_balanced_failures_v1.zip` under `/nas.dbms/asera` and extract:

```bash
cd /nas.dbms/asera
unzip paper1_balanced_failures_v1.zip
cd paper1_balanced_failures_v1
sha256sum -c CHECKSUMS.sha256
```

Your working Paper 1 environment already uses NumPy and pycocotools. The figures
also need Pillow. Check the inputs before starting:

```bash
bash /nas.dbms/asera/paper1_balanced_failures_v1/run_failure_analysis.sh --check
```

Run the full five-model, three-dataset, three-seed analysis and render every model
on the selected paired cases:

```bash
bash /nas.dbms/asera/paper1_balanced_failures_v1/run_failure_analysis.sh \
  --render-all-models \
  --example-size any
```

The launcher uses `P1_FAILURE_PY`, then `P1_UGS_PY`, then
`/nas.dbms/asera/paper1_ugs_env/bin/python`. Set the first variable if your working
Python has another path:

```bash
export P1_FAILURE_PY=/nas.dbms/asera/paper1_ugs_env/bin/python
```

The defaults are:

| Item | Default |
| --- | --- |
| Legacy run root | `/nas.dbms/asera/paper1_original/ugs_comparison/runs` |
| Recent run root | `/nas.dbms/asera/paper1_original/reviewer34/runs` |
| Datasets | `coco_nw coco_weather acdc` |
| Models | `baseline lgf_original ugs_reimpl yolo11m rtdetr-l` |
| Seeds | `42 1337 2025` |
| Checkpoint selection | `best_map` |
| Output | `/nas.dbms/asera/paper1_original/balanced_failure_analysis_v1` |
| Count score thresholds | `0.05 0.25 0.50` |
| Count IoUs | `0.50 0.75` |
| Main image/log operating point | score `0.25`, IoU `0.50` |
| Example seed | `42` |
| Examples | up to two distinct images per outcome category per dataset |

The code remaps old `/tmp/asera` references to `/nas.dbms/asera`. It verifies the
saved prediction hashes. It also finds the exact selected epoch's prediction
file under `evaluation/epoch_NNN`, `shared_evaluation/epoch_NNN` or the archived
`shared_evaluations/epoch_NNN` layout when needed.
It never searches for the most favourable epoch.

For a smaller first run:

```bash
bash /nas.dbms/asera/paper1_balanced_failures_v1/run_failure_analysis.sh \
  --datasets coco_nw \
  --methods baseline lgf_original \
  --seeds 42 \
  --example-size any \
  --output /nas.dbms/asera/paper1_original/balanced_failure_pilot_v1
```

Use `--no-render` for counts and detection logs without loading image pixels.
Without `--render-all-models`, figures show ground truth, RetinaNet and LGF.
Use `--example-size small` to focus figures on small objects. Counts always
include all four size groups. An unavailable example category is recorded with
zero examples; the script does not substitute a favourable category or seed.

## Models and checkpoint provenance

The default analysis uses the same `best_map.metrics.json` files from the current
UGS and reviewer34 runners. Each report supplies its actual selected prediction
file and epoch. The code checks dataset, method, seed, selection, predictions,
annotation hash and recorded evaluation content. All models need matching seed
sets and the same evaluated annotations and audited image bytes.

Use the exact baseline/LGF reports supporting your manuscript tables. If the
manuscript uses older original-checkpoint evaluations, set their report and audit
paths explicitly with `--manifest`. Do not replace them silently with another
training run. Original report method names `original_none`, `original_lgf`,
`original_se` and `original_cbam` map to their corresponding analysis labels.

Example explicit input file, with real paths replacing these sample paths:

```json
{
  "datasets": {
    "coco_nw": {
      "audit": "/nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_nw/baseline/seed_42/data_audit.json"
    }
  },
  "runs": [
    {
      "dataset": "coco_nw", "method": "baseline", "seed": 42,
      "report": "/nas.dbms/asera/your_original_baseline_evaluation/metrics.json"
    },
    {
      "dataset": "coco_nw", "method": "lgf_original", "seed": 42,
      "report": "/nas.dbms/asera/your_original_lgf_evaluation/metrics.json"
    }
  ]
}
```

```bash
bash /nas.dbms/asera/paper1_balanced_failures_v1/run_failure_analysis.sh \
  --manifest /nas.dbms/asera/your_failure_inputs.json \
  --example-size any \
  --output /nas.dbms/asera/paper1_original/original_checkpoint_failures_v1
```

An explicit dataset entry also accepts `annotations` and `images` for relocated
files. The annotations must match the saved audit hash. An explicit run accepts
its own `audit` to override a dataset entry. The report must declare the chosen
checkpoint selection and retain the prediction hash. A missing or incompatible
audit stops the analysis before output generation.

For SE and CBAM, add their names once the selected prediction reports exist:

```bash
bash /nas.dbms/asera/paper1_balanced_failures_v1/run_failure_analysis.sh \
  --methods baseline lgf_original ugs_reimpl yolo11m rtdetr-l se cbam \
  --render-all-models \
  --example-size any \
  --output /nas.dbms/asera/paper1_original/balanced_failure_analysis_se_cbam_v1
```

If SE/CBAM use another run root, add `--se-cbam-root /absolute/path/to/runs`.
Missing selected reports are reported explicitly. No new training starts.

## What the counts mean

Matching uses the official `pycocotools.COCOeval` bbox matcher, including class,
score ordering, one-to-one ordinary-object matching and crowd ignore rules.
The script normalizes annotation IDs in an evaluation-only copy because zero is
COCOeval's unmatched sentinel. Original annotation IDs appear in the event logs.

| Measure | Definition at the declared score and IoU |
| --- | --- |
| TP | Correct class and matched ground-truth object at the required IoU |
| FP | Retained detection without a nonignored match |
| FN | Nonignored ground-truth object without a retained matched detection |
| Ignored | A detection ignored by COCO, such as a crowd match |
| Evaluated detections | TP + FP |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2TP / (2TP + FP + FN) |
| Miss rate | FN / (TP + FN) |
| FP per image | FP / number of evaluated images |

Zero denominators produce a blank CSV cell. TN is undefined for this analysis.
Object detection has no fixed finite set of negative background boxes. Therefore
classification accuracy, specificity and false-positive rate based on TN are
not reported. A wrong-class detection counts as an FP in the predicted class and
leaves the true object as an FN if no other correct detection matches it.

Crowd annotations are not ordinary missed objects. Background-only images stay
in the image set and contribute their false detections.

The official evaluator uses the declared cap per image per category. These
exports already have their detector's native postprocessing and global cap;
the script does not reconstruct detections discarded during inference.

### Size definitions

All sizes use original image coordinates. TP and FN use the ground-truth
annotation's `area`, which sometimes differs from bbox area. FP has no matched
ground-truth object, so its size uses the predicted bbox width times height.

The main diagnostic partition is exclusive:

| Group | Area |
| --- | --- |
| Small | Area < 1,024 px² |
| Medium | 1,024 px² <= area < 9,216 px² |
| Large | Area >= 9,216 px² |

For TP, FP and FN, small + medium + large equals all in this partition.
The size groups assign matched detections by their target's size and unmatched
detections by their own box size. Their per-size precision is a diagnostic of
those assignments, not COCO's size-specific AP metric.

`coco_area_counts_per_seed.csv` separately preserves the official COCO area
matcher. COCO's size ranges share the exact 1,024 and 9,216 endpoints and apply
area-specific ignore rules to out-of-range matches. Those official-area counts
do not necessarily add to the overall count. They are labelled separately to
prevent confusion with the exclusive partition.

Scores from different architectures are not calibrated probabilities on a shared
scale. The common score sweep reports descriptive operating points. It does not
establish equal calibration or equal FP budgets. Existing AP metrics remain
separate from threshold-dependent precision, recall and F1. The script refuses a
requested threshold below the saved export's score floor.

## Balanced examples for Comment 15

The image selection covers four outcome categories:

1. LGF detects an annotated object which RetinaNet misses.
2. RetinaNet detects an annotated object which LGF misses.
3. Both detectors miss the same annotated object.
4. LGF adds an unmatched detection without a paired RetinaNet false positive.

LGF-only false positives are defined by a same-class, one-to-one pairing between
the two detectors' FP boxes at the declared `--fp-pair-iou`, default 0.50. This is
an operational definition of additional false detections at a chosen threshold,
not proof about the internal cause of an error.

The examples use a fixed inference seed and a reproducible sampling seed. The
sampler chooses distinct image IDs within each category. Categories can share
an image because a scene can contain several different object outcomes. The
target crop and white outline identify the specific event.

Each image shows the same scene and crop for ground truth and both detectors.
Use `--render-all-models` to add the remaining models to those same cases. The
code verifies the displayed image hash against the saved evaluation audit.

| Colour | Meaning |
| --- | --- |
| Cyan | Ground truth |
| Green | True positive |
| Red | False positive |
| Orange | Missed annotated object |
| Grey | Ignored/crowd |
| White outline | Selected target object or detection |

Review the displayed cases before publication. Describe visible evidence such
as low contrast, partial occlusion, dense objects, class confusion or a duplicate
box only when the image supports the explanation. An unmatched prediction is an
FP relative to the supplied labels; visible annotation omissions should be
described separately. Do not infer a gating explanation from these boxes alone.

`figure_review_template.csv` provides caption and annotation-review fields.
Use one compact paired figure in the manuscript and place additional cases and
size tables in the supplement to respect the seven-page main-paper limit.
Update the reviewer response and exact locations after the real figures and
tables have been inspected and inserted. The current response's statement about
missing balanced detections should not be replaced before that evidence exists.

## Outputs

| File | Contents |
| --- | --- |
| `counts_per_seed.csv` | TP, FP, FN, precision, recall, F1 and miss rate for every model, seed, size, threshold and IoU |
| `counts_seed_summary.csv` | Mean and sample SD across the available matching seeds |
| `coco_area_counts_per_seed.csv` | Counts using the native COCO area matcher |
| `source_metrics_per_seed.csv` | Saved AP metrics and selected epochs, copied from the source reports |
| `image_counts.csv` | Per-image counts for every model and size at the main operating point |
| `detection_outcomes.jsonl` | Actual retained prediction boxes, scores, classes, TP/FP/ignored labels and matched annotations |
| `missed_objects.jsonl` | Actual missed annotated boxes for each model |
| `paired_objects.csv` | Matched object-level RetinaNet/LGF outcomes |
| `paired_false_positives.csv` | Shared and additional FP boxes, including paired prediction indices |
| `paired_counts_per_seed.csv` | Recovery, loss, shared-miss and additional-FP counts by size |
| `paired_counts_seed_summary.csv` | Seed means and sample SD for paired outcomes |
| `all_example_candidates.json` | Every eligible object/FP event, including its focus box |
| `selected_examples.json` | Reproducibly selected cases, image hashes and crop coordinates |
| `example_category_inventory.csv` | Available and selected examples per category, including empty categories |
| `figure_review_template.csv` | Fields for manual image review and final captions |
| `figures/` | Ground-truth and detector overlays with target crops |
| `analysis_manifest.json` | Source hashes, checkpoint identities, definitions, thresholds and software versions |

Counts across different training seeds describe repeated predictions of the same
labelled objects. The summary reports means and sample SD; it does not treat
three predictions of one object as three independent ground-truth objects.
Undefined ratios are excluded from ratio means; each metric includes its number
of defined seed values. SD is
not a significance test. These selected validation checkpoints do not provide
independent held-out test estimates.

The official hidden-label COCO/ACDC test exports alone do not support TP, FP, FN,
matched failures or local size-specific error rates. Image-info JSON supplies
image identities, not annotated objects. Use the labelled validation exports
for this analysis.

## Verification and dependencies

Dependencies: NumPy, pycocotools and Pillow for rendering. No PyTorch upgrade is
needed. The package was tested with pycocotools 2.0.11. For this analysis, use your
existing working environment so evaluator-version changes stay visible.

```bash
"$P1_FAILURE_PY" -m unittest discover \
  -s /nas.dbms/asera/paper1_balanced_failures_v1/tests -v
```

The synthetic tests cover annotation ID zero, duplicate detections, wrong
classes, crowds, empty images/predictions, score sweeps checked against fresh
COCO matching, area boundaries, GT versus predicted-box size, paired FP matching,
automatic selected-report discovery, hashes, hidden labels and figure generation.

Primary evaluator reference:
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
