# Reviewer diagnostics

Run commands from the repository root. The Python algorithms and model source
in these recovered packages retain their original bytes. The gate-map and
failure-analysis shell launchers now locate their scripts relative to their
own directories. Raw research predictions, weights, images, and rendered
figures are external inputs or generated outputs.

## Environment and paths

Use the environment documented in the root installation instructions. Gate
controls and map inference require the supported torch/torchvision environment.
Bootstrap and failure analysis run on CPU without torch. Failure figures require
Pillow; gate maps require NumPy, Pillow, and matplotlib. COCO matching requires
pycocotools.

Define your existing data and run locations. Replace these three example paths:

```bash
export P1_DATA_ROOT=/absolute/path/to/DATASET
export P1_UGS_RUN_ROOT=/absolute/path/to/ugs_comparison/runs
export P1_RECENT_RUN_ROOT=/absolute/path/to/reviewer34/runs
export P1_RESULTS_ROOT="$PWD/results"
```

The legacy LGF checkpoint layout is
`<run-root>/<dataset>/lgf_original/seed_<seed>/<selection>.pth`.
The corresponding selected prediction report usually ends in
`<selection>.metrics.json`. Datasets are `coco_nw`, `coco_weather`, and `acdc`.
Training seeds are `42`, `1337`, and `2025`.

Use the exact checkpoint and prediction reports behind your manuscript tables.
The recovered failure-analysis output used 500 fog images for COCO-weather.
This differs from the latest manuscript's 1,500-image diagnostic description.
Use the saved 500-image inputs to repeat those exact counts, or generate a
separately identified full-weather analysis from 1,500-image exports. See
`provenance/diagnostic_support.json` and the reproducibility document.
The retained defaults differ between scripts: controls and gate maps use
`best_ap_small`, while failure analysis uses `best_map`. An original notebook
checkpoint selected by mAP must use `best_map`. Set the rule explicitly for each
analysis and do not substitute a recent training run for a reported original run.

## Frozen-checkpoint gate controls

Package: `experiments/lgf_original_controls_v1`.
`controls.py` verifies input inventory, model source, EMA weights, and data
provenance. Each intervention uses the same saved checkpoint. No new training
starts.

Check, evaluate, then summarize the three-dataset, three-seed grid:

```bash
python experiments/lgf_original_controls_v1/controls.py check \
  --run-root "$P1_UGS_RUN_ROOT" --data-root "$P1_DATA_ROOT" \
  --datasets coco_nw coco_weather acdc --seeds 42 1337 2025 \
  --level C3 --selection best_ap_small --suite paper --device cuda

python experiments/lgf_original_controls_v1/controls.py run \
  --run-root "$P1_UGS_RUN_ROOT" --data-root "$P1_DATA_ROOT" \
  --output-root "$P1_RESULTS_ROOT/controls_paper" \
  --datasets coco_nw coco_weather acdc --seeds 42 1337 2025 \
  --level C3 --selection best_ap_small --suite paper --device cuda \
  --batch-size 4 --shuffle-seed 123

python experiments/lgf_original_controls_v1/controls.py summarize \
  --output-root "$P1_RESULTS_ROOT/controls_paper"
```

The default source package is the bundled `source_runner/`. Override with
`--lgf-package /absolute/path/to/source` only when the preserved original core
hash matches. To evaluate an original notebook checkpoint, supply
`--checkpoint /absolute/path/to/BEST_checkpoint.pth`, one dataset, one seed,
the matching level, and `--selection best_map`. The checkpoint must contain
compatible metadata and EMA weights.

The default paper suite retains the residual `X + gamma * Y`:

| Intervention | Fusion update Y |
| --- | --- |
| native | wL L + wG G |
| uniform | 0.5 L + 0.5 G |
| shuffle | Both native gate maps receive the same batch/channel permutation |
| swap | wG L + wL G |
| alpha_0.25 | 0.25 L + G |
| alpha_0.5 | 0.5 L + G |
| alpha_2.0 | 2 L + G |
| no_gate | L + G |

The sigmoid weights are independent. Their sum is not constrained to one.
Shuffling acts across batch times channels and retains spatial positions.
With batch size one, shuffling acts across channels only. Keep batch composition
fixed when comparing shuffle outcomes. These frozen interventions measure
sensitivity to changed fusion. They do not estimate a retrained ungated model.

The optional `--suite notebook` reproduces the recovered notebook's historical
normalization and `no_module` intervention. Those equations differ from the
paper suite. Use another output directory and label the suite in your report.

Outputs include `controls_per_seed.csv`, `controls_mean_std.csv`,
`summary_status.json`, and an input/source/environment manifest. AP uses the
0 to 1 scale. Delta columns use `100 * (control AP - native AP)` in percentage
points. Seed standard deviations use `ddof=1`.

Repeat the identical run command to resume. Changed input bytes or settings
require a new output directory. Optional `--benchmark` measurements use the
first validation batch, synchronized CUDA execution, five warmup passes, and
twenty measured passes. Amortized batch time is distinct from batch-one latency.

## Paired synthetic-weather gate maps

Package: `experiments/paper1_gate_maps_four_conditions_v2`.
One COCO-weather LGF checkpoint processes clear, fog, rain, and snow versions
of the same source scenes. Selected small ground-truth boxes must agree across
conditions. Plot scales stay fixed at 0 to 1.

The included `coco_weather_image_conditions.csv` records the historical audited
dataset filenames and IDs. Replace `--weather-index` with a freshly audited
condition index when using another dataset inventory. The script refuses
inconsistent annotation/index filenames or incomplete four-condition scenes.

First check the pairing without loading weights:

```bash
python experiments/paper1_gate_maps_four_conditions_v2/paper1_gate_visualization_conditions_v2.py \
  --checkpoint "$P1_UGS_RUN_ROOT/coco_weather/lgf_original/seed_42/best_ap_small.pth" \
  --package-root experiments/lgf_original_controls_v1/source_runner \
  --data-root "$P1_DATA_ROOT" \
  --weather-index experiments/paper1_gate_maps_four_conditions_v2/coco_weather_image_conditions.csv \
  --output "$P1_RESULTS_ROOT/gates_coco_weather" \
  --level C3 --num-scenes 4 --seed 42 --device cuda:0 --plan-only
```

Repeat the same command without `--plan-only` for inference and figures. To use
the complete pooled weather annotations, add
`--weather-ann /absolute/path/to/mini_val2017_weather_fog_rain_snow.json`.
Override relocated paths with `--clear-ann`, `--weather-ann`, `--clear-images`,
and `--weather-images`. Supply repeated `--scene-key` flags to select known
source scenes instead of deterministic sampling.

If weather annotations contain only fog, the audited rain/snow siblings inherit
unchanged boxes from the annotated scene for visualization. The script records
the inherited source ID. These display boxes are not an AP evaluation manifest.
The images must already exist and have matching dimensions. This script does
not generate weather images.

The portable wrapper runs the check and figure commands together:

```bash
bash experiments/paper1_gate_maps_four_conditions_v2/run_gate_maps_conditions_v2.sh
```

The wrapper reads `P1_DATA_ROOT` and `P1_UGS_RUN_ROOT`. Optional overrides are
`P1_UGS_PY` for the Python executable, `P1_LGF_PACKAGE`, `P1_GATE_CHECKPOINT`,
`P1_GATE_OUTPUT`, `P1_WEATHER_INDEX`, and `P1_WEATHER_ANN`. Its default output is
`results/gate_visualization/reviewer4_paired_seed42_v2`.

Outputs: full panels, small-object crops, `target_gate_stats.csv`, and
`manifest.json`. Gate means describe learned fusion weights before residual
gamma scaling. Gate maps alone do not measure a performance contribution.

## Optional real-weather ACDC gate maps

Package: `experiments/paper1_acdc_gate_maps_by_condition_v1`.
ACDC uses fog, rain, snow, and night. Different conditions show different real
scenes; these are not paired weather versions of the same scene. Only frames
labelled as the original ACDC validation split enter the figure selection.

```bash
python experiments/paper1_acdc_gate_maps_by_condition_v1/paper1_acdc_gate_visualization.py \
  --checkpoint "$P1_UGS_RUN_ROOT/acdc/lgf_original/seed_42/best_ap_small.pth" \
  --package-root experiments/lgf_original_controls_v1/source_runner \
  --data-root "$P1_DATA_ROOT" \
  --condition-index experiments/paper1_acdc_gate_maps_by_condition_v1/acdc_image_conditions.csv \
  --output "$P1_RESULTS_ROOT/gates_acdc" \
  --level C3 --device cuda:0 --selection-seed 42 --per-condition 2 --plan-only
```

Repeat without `--plan-only` to render. The portable wrapper is
`run_gate_maps_acdc.sh`; optional overrides use the same gate variables plus
`P1_ACDC_INDEX`. Its default output is
`results/gate_visualization/acdc_by_condition_seed42`. The historical
`acdc_image_conditions.csv` must match your annotation filenames and IDs.

## Paired bootstrap

Package: `experiments/paper1_paired_bootstrap`.
The preserved v1.0.1 program recomputes APsmall from saved predictions. Every
replicate shares image draws between detectors and across seed pairs, then
resamples matched training-seed pairs with replacement. The interval is a
percentile confidence interval for the mean paired APsmall change.

Example for baseline versus LGF on COCO-NW:

```bash
python experiments/paper1_paired_bootstrap/paper1_paired_bootstrap.py \
  --annotations "$P1_DATA_ROOT/COCO/non_weather-mini/annotations/mini_val2017_non_weather-500k6c.json" \
  --pair 42 "$P1_UGS_RUN_ROOT/coco_nw/baseline/seed_42/best_map.metrics.json" "$P1_UGS_RUN_ROOT/coco_nw/lgf_original/seed_42/best_map.metrics.json" \
  --pair 1337 "$P1_UGS_RUN_ROOT/coco_nw/baseline/seed_1337/best_map.metrics.json" "$P1_UGS_RUN_ROOT/coco_nw/lgf_original/seed_1337/best_map.metrics.json" \
  --pair 2025 "$P1_UGS_RUN_ROOT/coco_nw/baseline/seed_2025/best_map.metrics.json" "$P1_UGS_RUN_ROOT/coco_nw/lgf_original/seed_2025/best_map.metrics.json" \
  --baseline-label RetinaNet --proposed-label LGF \
  --replicates 10000 --confidence 0.95 --random-seed 20260919 \
  --workers 8 --iou-backend auto \
  --output "$P1_RESULTS_ROOT/bootstrap_coco_nw_baseline_vs_lgf"
```

Repeat separately for each dataset and comparator. For YOLO11m or RT-DETR-L,
use their matching selected reports from `P1_RECENT_RUN_ROOT` as the baseline
side and set the label accordingly. The sign always means proposed minus
baseline. Reports are preferred over direct prediction files because reports
record hashes, seeds, checkpoint selection, and full-data APsmall.

Input annotations must contain strictly positive annotation IDs. The preserved
bootstrap uses the historical COCO evaluator directly, whose zero ID represents
an unmatched detection. An annotation document containing ID zero requires a
separately versioned correction and revalidation. Do not edit historical
annotations or prediction reports in place.

The script verifies full-data APsmall against the selected report and refuses
mixed declared datasets or selection rules. All reports must evaluate the same
annotation file. Direct prediction files lack checkpoint provenance and need
an independently audited manifest. Labelled validation annotations are required;
hidden test image-info files are insufficient.

`--iou-backend auto` probes the compiled bbox IoU helper and selects the NumPy
bbox fallback for the documented ndarray-conversion failure. COCOeval performs
matching and accumulation under either backend. `--workers 1` works across
platforms; multiple workers require fork support, with a serial fallback.

The output directory must be new. Outputs include a JSON/CSV summary and all
replicate deltas. Raw AP differences use the 0 to 1 scale; the summary also
reports percentage points. A confidence interval crossing zero does not
establish a positive gain. This validation bootstrap is not a held-out test
estimate or an inference-latency measurement.

## Balanced detection failures

Package: `experiments/paper1_balanced_failures_v1`.
This CPU workflow loads selected prediction reports, checks hashes and data
audits, then counts TP, FP, FN, ignored detections, precision, recall, F1, miss
rate, and FP per image. TN is undefined for detection with an unrestricted set
of background boxes.

Check then run the original five-model analysis:

```bash
python experiments/paper1_balanced_failures_v1/paper1_failure_analysis.py \
  --ugs-root "$P1_UGS_RUN_ROOT" --recent-root "$P1_RECENT_RUN_ROOT" \
  --datasets coco_nw coco_weather acdc \
  --methods baseline lgf_original ugs_reimpl yolo11m rtdetr-l \
  --seeds 42 1337 2025 --selection best_map \
  --output "$P1_RESULTS_ROOT/balanced_failures" --check

python experiments/paper1_balanced_failures_v1/paper1_failure_analysis.py \
  --ugs-root "$P1_UGS_RUN_ROOT" --recent-root "$P1_RECENT_RUN_ROOT" \
  --datasets coco_nw coco_weather acdc \
  --methods baseline lgf_original ugs_reimpl yolo11m rtdetr-l \
  --seeds 42 1337 2025 --selection best_map \
  --confidences 0.05 0.25 0.50 --ious 0.50 0.75 \
  --primary-confidence 0.25 --primary-iou 0.50 \
  --example-seed 42 --example-random-seed 20260930 \
  --example-size any --render-all-models \
  --output "$P1_RESULTS_ROOT/balanced_failures"
```

Use `--no-render` for counts/logs alone. Include `se cbam` in `--methods` only
when their matching reports exist, and set `--se-cbam-root` if needed. Reports
using `original_none`, `original_lgf`, `original_se`, and `original_cbam` receive
the corresponding method labels automatically.

If baseline/LGF tables use original-checkpoint reports stored elsewhere, create
an explicit JSON manifest rather than relying on run-root discovery:

```json
{
  "datasets": {
    "coco_nw": {
      "audit": "/absolute/path/to/data_audit.json",
      "annotations": "/absolute/path/to/validation_annotations.json",
      "images": "/absolute/path/to/validation_images"
    }
  },
  "runs": [
    {"dataset": "coco_nw", "method": "baseline", "seed": 42,
     "report": "/absolute/path/to/baseline/best_map.metrics.json"},
    {"dataset": "coco_nw", "method": "lgf_original", "seed": 42,
     "report": "/absolute/path/to/lgf/best_map.metrics.json"}
  ]
}
```

```bash
python experiments/paper1_balanced_failures_v1/paper1_failure_analysis.py \
  --manifest /absolute/path/to/failure_inputs.json --selection best_map \
  --example-size any --output "$P1_RESULTS_ROOT/original_checkpoint_failures"
```

Add every intended seed/model to `runs`. Relocated annotation and image bytes
must match the audit. Use repeated `--remap OLD=NEW` flags for embedded absolute
paths. The historical default remaps `/tmp/asera` to `/nas.dbms/asera`; a new
remap extends this list. Prediction discovery verifies the selected epoch and
hash. The script does not choose a more favourable epoch or another seed.

Matching follows official COCO bbox rules, including class, score ordering,
crowd handling, and maximum detections. Zero annotation IDs receive a normalized
evaluation-only ID while logs retain original IDs. The requested score floor
must stay at or above the export's recorded floor.

The exclusive diagnostic size partition is small below 1,024 px², medium from
1,024 to below 9,216 px², and large at or above 9,216 px². TP/FN use GT area;
FP uses predicted box area. A separate output preserves official COCO area
matching with its overlapping exact endpoints and ignore rules. Size-specific
precision is a threshold diagnostic, distinct from COCO APsmall.

Selected cases cover LGF-only recovery, RetinaNet-only recovery, shared misses,
and additional unmatched LGF boxes. Fixed seeds and deterministic image
sampling prevent best-case selection. The same scenes/crops appear across
models. Image hashes are checked before rendering.

## Software verification

```bash
python -m unittest discover -s experiments/lgf_original_controls_v1/tests -v
python -m unittest discover -s experiments/paper1_gate_maps_four_conditions_v2/tests -v
python -m unittest discover -s experiments/paper1_paired_bootstrap/tests -v
python -m unittest discover -s experiments/paper1_balanced_failures_v1/tests -v
```

Controls tests require torch and torchvision. Bootstrap COCO accumulation tests
require pycocotools; without this dependency those tests explicitly skip.
All fixtures are synthetic. Passing software checks does not reproduce paper
metrics without the original trained checkpoints and audited dataset bytes.
