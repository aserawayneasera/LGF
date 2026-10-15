# Weather evaluation and placement experiments

The runnable packages in this guide preserve the recovered revision source. Model
definitions, checkpoint loading, input transforms, and saved run labels are
unchanged. The published placement comparison uses the saved labels `C3` and
`C4`. The recovered runner also supports exploratory `C5` and combined-placement
runs; these configurations do not belong in the reported C3/C4 comparison.

| Package under `experiments/` | Role | Default checkpoint selection |
| --- | --- | --- |
| `paper1_weather_eval_v1` | Audit condition labels and evaluate saved detectors by condition | `best_map` |
| `paper1_coco_three_weather_v1` | Build evaluation manifests for existing fog/rain/snow variants | Launcher uses `best_map` |
| `paper1_ugs_multilevel_v1` | Recovered placement training and validation runner | Saves `best_map` and `best_ap_small` |
| `paper1_placement_profile_v1` | Parameters, defined MAC/FLOP counts, and detector timings | `best_ap_small` |
| `c4_weather_evaluation_v1` | Strict saved-checkpoint evaluation for C3/C4 on all three weather conditions | `best_ap_small` |

Package READMEs retain the original server-specific instructions as historical
documentation. The commands below use this repository's layout. Generated
manifests, predictions, profiler reports, logs, and checkpoint binaries are
external inputs or generated outputs, rather than distributed source files.

## Paths and environments

Start from the repository root and replace the absolute paths below. `P1PY` is
the original RetinaNet/UGS training environment. `P34PY` is the recent-detector
environment used for YOLO11m and RT-DETR-L. Exporting these variables passes them
to the recovered shell launchers.

```bash
export REPO="$PWD"
export P1PY=/absolute/path/to/original-environment/bin/python
export P34PY=/absolute/path/to/recent-detector-environment/bin/python
export DATA=/absolute/path/to/DATASET
export RUNS_BASE=/absolute/path/to/ugs_comparison
export UGS_RUNS="$RUNS_BASE/runs"
export RECENT_RUNS=/absolute/path/to/reviewer34/runs
export ORIGINAL_ROOT=/absolute/path/to/original-notebook-checkpoints
export WEATHER="$REPO/experiments/paper1_weather_eval_v1/paper1_weather_eval.py"
export PKG="$REPO/experiments"
export SOURCE_OUT=/absolute/path/to/new-weather-condition-audit
export OUT3=/absolute/path/to/new-three-weather-evaluation
export TORCH_HOME=/absolute/path/to/torch-cache
```

The original runtime checks require Torch 2.10.x and Torchvision 0.25.x. Retain
the environment used to produce the checkpoints. Inference also requires NumPy,
OpenCV, Pillow, Albumentations, and pycocotools. Profiling adds
`fvcore==0.1.5.post20221221`, listed in
`experiments/paper1_placement_profile_v1/requirements-profile.txt`. Preparation
and the audit/preflight commands do not load detector weights. Checkpoint
inference and numerical COCO evaluation require their runtime dependencies.

The expected dataset layout follows the retained original code:

```text
DATASET/
  COCO/non_weather-mini/annotations/mini_{train,val}2017_non_weather-{2400,500}k6c.json
  COCO/non_weather-mini/images/{train,val}2017_non_weather-{2400,500}k6c/
  COCO/weather-mini/annotations/mini_{train,val}2017_weather-{2400,500}k6c.json
  COCO/weather-mini/images/{train,val}2017_weather-{2400,500}k6c/
  ACDC-1/ACDC-1-NEW/annotations/mini_train.json
  ACDC-1/ACDC-1-NEW/annotations/mini_val.json
  ACDC-1/ACDC-1-NEW/images/train/
  ACDC-1/ACDC-1-NEW/images/val/
```

The braces above describe the naming pattern, rather than shell commands. See
the data guide for the complete dataset preparation workflow. Explicit
`--annotations` and `--images` flags override one evaluation dataset at a time.

## 1. Audit the existing weather annotations

```bash
"$P1PY" "$WEATHER" audit \
  --datasets coco_weather \
  --data-root "$DATA" \
  --output-root "$SOURCE_OUT"
```

The resulting `audits/coco_weather/condition_audit.json` records the original
annotations, their hash, image root, weather labels, and support counts. A
folder containing all weather variants does not establish which variants the
annotation JSON contains. The audit reports only annotated conditions.

For flattened ACDC filenames, provide original condition provenance:

```bash
"$P1PY" "$WEATHER" audit \
  --datasets acdc \
  --data-root "$DATA" \
  --acdc-source /absolute/path/to/rgb_anon_trainvaltest.zip \
  --output-root "$SOURCE_OUT"
```

`--acdc-source` accepts the original RGB archive, extracted original directory,
or supported official detection metadata. A verified CSV containing exact
`file_name,condition` columns is supplied through `--condition-map` for one
dataset. Unknown labels, contradictory metadata, missing images, and recognised
train/evaluation source overlap stop evaluation.

## 2. Evaluate all existing COCO weather variants

This preparation reads the audit from step 1. Its default requires 500 source
scenes, each with an existing geometry-preserving fog, rain, and snow image.
The combined manifest contains 1,500 image variants, with 500 images per
condition. The same source scenes occur in all three conditions. Source images
and original annotations retain their contents.

The existing launcher's alignment assertion applies to the original
geometry-preserving weather transformations. For a different collection,
verify source alignment and adjust the preparer's explicit inputs before
running. Equal dimensions alone do not establish box alignment.

Set checkpoint origins explicitly. The example below uses the later matched
weather comparison saved under the UGS runner's directories. To evaluate
retained notebook checkpoints, choose `BASELINE_SOURCE=notebook` or
`LGF_SOURCE=notebook` and supply the relevant `ORIGINAL_ROOT`.

```bash
export BASELINE_SOURCE=ugs
export LGF_SOURCE=ugs
export GPU_DEVICE=0
export EVAL_BATCH=4

bash "$REPO/experiments/paper1_coco_three_weather_v1/run_coco_three_weather.sh" prepare
bash "$REPO/experiments/paper1_coco_three_weather_v1/run_coco_three_weather.sh" check \
  baseline lgf_original ugs_reimpl yolo11m rtdetr-l
bash "$REPO/experiments/paper1_coco_three_weather_v1/run_coco_three_weather.sh" run \
  baseline lgf_original ugs_reimpl yolo11m rtdetr-l
bash "$REPO/experiments/paper1_coco_three_weather_v1/run_coco_three_weather.sh" summarize
```

The launcher uses seeds 42, 1337, and 2025, fixed `best_map` selection, and saved
run label C3. Each detector performs inference once over all 1,500 variants.
The evaluator then scores fog, rain, snow, and `ALL`. `ALL` pools detections and
ground truth before COCO scoring. `ALL` is not an average of condition APs.

Additional `se cbam` arguments select original notebook controls when those
checkpoints exist. Changing settings or data requires a new output root.
Verified identical reruns reuse complete results.

## 3. Evaluate ACDC conditions with one checkpoint per seed

```bash
"$P1PY" "$WEATHER" evaluate \
  --datasets acdc --methods baseline lgf_original ugs_reimpl yolo11m rtdetr-l \
  --seeds 42 1337 2025 --selection best_map --level C3 \
  --data-root "$DATA" --package-root "$PKG" \
  --original-root "$ORIGINAL_ROOT" --ugs-root "$UGS_RUNS" --recent-root "$RECENT_RUNS" \
  --baseline-source ugs --lgf-source ugs \
  --legacy-python "$P1PY" --recent-python "$P34PY" \
  --acdc-source /absolute/path/to/rgb_anon_trainvaltest.zip \
  --device 0 --batch-size 4 --output-root "$SOURCE_OUT" --check
```

Run the same command after removing `--check` to perform inference. `--check`
verifies paths, label provenance, and job construction. Checkpoint tensors
are loaded only during inference. One selected checkpoint processes every
condition within its evaluation split. No per-condition checkpoint selection
occurs.

## 4. Separate clear-trained transfer

`--train-dataset coco_nw` selects an unchanged original-COCO-trained checkpoint
while evaluating the expanded weather manifest. The source setting must match
the checkpoints used for the manuscript transfer experiment.

```bash
"$P1PY" "$WEATHER" evaluate \
  --datasets coco_weather --train-dataset coco_nw \
  --annotations "$OUT3/manifests/mini_val2017_weather_fog_rain_snow.json" \
  --images "$DATA/COCO/weather-mini/images/val2017_weather-500k6c" \
  --methods baseline lgf_original --seeds 42 1337 2025 \
  --selection best_map --level C3 --data-root "$DATA" --package-root "$PKG" \
  --original-root "$ORIGINAL_ROOT" --ugs-root "$UGS_RUNS" \
  --baseline-source notebook --lgf-source notebook \
  --legacy-python "$P1PY" --device 0 --batch-size 4 \
  --output-root /absolute/path/to/new-clear-trained-weather-transfer --check
```

Remove `--check` for inference. Do not combine these transfer rows with matched
weather training rows. The command records the selected training dataset; the
underlying data and training lineage establish claims about unseen weather.

## 5. Recover the C4 placement training workflow

The placement runner retains the original single-level constructors for the
saved C3/C4 run labels. Model loading and profiling reconstruct saved module
keys directly. Do not replace this source with `paper1_ugs_semantic_v2` or use
`paper1_placement_checks_v2` when reproducing the supplied manuscript runs.
Archived feature audits are available for provenance; their labels do not
replace the retained experimental labels in manuscript tables.

The command below starts a fresh C4 run. Use a separate directory from the
saved manuscript checkpoints. Repeat for each dataset and seed if reproducing
training. The run root gains `<dataset>/lgf_original/seed_<seed>/`.

```bash
"$P1PY" "$REPO/experiments/paper1_ugs_multilevel_v1/run.py" train \
  --dataset coco_weather --method lgf_original --level C4 --seed 42 \
  --data-root "$DATA" --run-root /absolute/path/to/new-runs_C4 \
  --epochs 80 --batch-size 4 --accum-steps 4 --base-lr 0.005 \
  --device cuda --no-amp
```

Use `--resume /absolute/path/to/last.pth` for a matching interrupted run.
Changing source hashes or the recorded protocol blocks resume. Existing saved
checkpoints support evaluation without retraining:

```bash
"$P1PY" "$REPO/experiments/paper1_ugs_multilevel_v1/run.py" evaluate \
  --dataset coco_nw --method lgf_original --level C4 --seed 42 \
  --data-root "$DATA" --device cuda --selection best_ap_small \
  --checkpoint "$RUNS_BASE/runs_C4/coco_nw/lgf_original/seed_42/best_ap_small.pth" \
  --output /absolute/path/to/new-C4-coco-nw-evaluation
```

The recovered `train_lgf_multilevel.sh` and `C3_C4`/`C3_C4_C5` code remain
available for exploratory reproducibility. The shell launcher's default starts
combined-placement training. Use the explicit C4 command above for the
published placement scope. Exploratory C5 and combined-placement results are
excluded from reported placement comparisons.

## 6. Evaluate C3 and C4 on fog, rain, and snow

This runner bundles the archived strict EMA reconstruction loader. Its default
selection is `best_ap_small`, matching the placement experiment rather than
the later `best_map` detector comparison. Its checkpoint paths are:

| Saved run label | Checkpoint below `RUNS_BASE` |
| --- | --- |
| C3 | `runs/coco_weather/lgf_original/seed_SEED/best_ap_small.pth` |
| C4 | `runs_C4/coco_weather/lgf_original/seed_SEED/best_ap_small.pth` |

```bash
export C4_WEATHER_PY="$P1PY"
export C4_OUT=/absolute/path/to/new-C3-C4-three-weather

bash "$REPO/experiments/c4_weather_evaluation_v1/run_c4_weather.sh" prepare \
  --data-root "$DATA" --runs-base "$RUNS_BASE" --output "$C4_OUT"
bash "$REPO/experiments/c4_weather_evaluation_v1/run_c4_weather.sh" check \
  --data-root "$DATA" --runs-base "$RUNS_BASE" --output "$C4_OUT" \
  --placements C3 C4 --selection best_ap_small
bash "$REPO/experiments/c4_weather_evaluation_v1/run_c4_weather.sh" run \
  --data-root "$DATA" --runs-base "$RUNS_BASE" --output "$C4_OUT" \
  --placements C3 C4 --selection best_ap_small --device cuda:0 --batch-size 4
bash "$REPO/experiments/c4_weather_evaluation_v1/run_c4_weather.sh" summarize \
  --data-root "$DATA" --runs-base "$RUNS_BASE" --output "$C4_OUT" \
  --placements C3 C4 --selection best_ap_small
```

`--source-annotations`, `--images`, and `--train-annotations` override the
derived dataset paths. Supply the same overrides to every applicable action.
Changing selection to `best_map` requires a separate output directory and the
matching saved checkpoints. The preparation step verifies existing paired
image variants and immutable annotations. Every reported seed uses one
checkpoint across all conditions.

## 7. Profile the published C3/C4 scope

Use profiler v1 and supply `--placements baseline C3 C4` explicitly. The
historical default also includes exploratory C5 and combined placements.

```bash
export P1_PROFILE_PY="$P1PY"

bash "$REPO/experiments/paper1_placement_profile_v1/run_profile.sh" \
  --runs-base "$RUNS_BASE" --data-root "$DATA" \
  --datasets coco_nw coco_weather acdc --placements baseline C3 C4 \
  --seeds 42 1337 2025 --selection best_ap_small --inventory-only \
  --output /absolute/path/to/new-placement-inventory

bash "$REPO/experiments/paper1_placement_profile_v1/run_profile.sh" \
  --runs-base "$RUNS_BASE" --data-root "$DATA" \
  --datasets coco_nw coco_weather acdc --placements baseline C3 C4 \
  --seeds 42 1337 2025 --selection best_ap_small \
  --device cuda:0 --batches 1 16 --sample-images 112 --warmup 20 --repeats 3 \
  --output /absolute/path/to/new-placement-profile
```

Each `--output` directory must be new or empty. The inventory check reports
missing checkpoints without loading Torch. Use `--manifest` with an edited
copy of `manifest.example.json` to select absolute paths for the exact retained
checkpoints. A manifest entry requires `dataset`, `placement`, `seed`, and
`checkpoint`; optional metric records are checked against checkpoint identity.
Use `--counts-only --device cpu` to obtain checkpoint parameter/complexity
counts without timing images. CPU timings do not reproduce GPU latency.

The profiler reports total parameters and LGF parameters. The reference input
is one RGB image at 800 by 1333, passed through the normal detector transform.
It records the resulting padded shape. fvcore uses one fused multiply-add per
FLOP. Independent hooks report convolution/linear GMACs and a separate
two-operations-per-MAC total. Unsupported operators are recorded. These totals
must retain their stated counting conventions.

FP32 timing includes the detector transform, backbone, LGF, FPN, head, decoding,
NMS, and output rescaling. Disk reads, image decoding, CPU tensor preparation,
host-to-device copies, and COCO scoring are outside the timed region. Batch-one
time is detector-call latency. Batch-16 time divided by 16 is amortized time per
image. Hardware and versions are recorded, so a different GPU supplies a new
measurement rather than reproducing the published RTX 5090 timing value.

## Outputs and interpretation

Weather summary files include per-seed AP/AR rows and means with sample standard
deviations. Ordinary AP values use the 0 to 1 scale. LGF-minus-baseline deltas
use percentage points. Undefined metrics are represented as unavailable rather
than replaced by zero. `n_seeds` identifies incomplete results.

Placement profiling reads checkpoint-associated validation metrics. It does
not recalculate accuracy. Parameter/FLOP/timing comparisons must reference the
intended saved checkpoints and retain the selected accuracy protocol.

The additional weather manifests evaluate existing validation scenes. These
rows do not establish independent-test performance. Repeated weather variants
share source scenes; uncertainty estimates over pooled variants should resample
source groups. Seed standard deviation does not substitute for a paired
bootstrap interval. See the statistics guide for the recovered bootstrap code.

## Tests

Run the archived suites from each package directory so their local imports
resolve independently:

```bash
cd "$REPO/experiments/paper1_weather_eval_v1"
"$P1PY" -m pytest -q tests

cd "$REPO/experiments/paper1_coco_three_weather_v1"
WEATHER="$REPO/experiments/paper1_weather_eval_v1/paper1_weather_eval.py" \
  "$P1PY" -m unittest -v test_prepare

cd "$REPO/experiments/c4_weather_evaluation_v1"
WEATHER="$REPO/experiments/paper1_weather_eval_v1/paper1_weather_eval.py" \
  "$P1PY" -m unittest discover -s tests -v

cd "$REPO/experiments/paper1_ugs_multilevel_v1"
"$P1PY" -m unittest discover -s tests -p 'test_multilevel.py' -v

cd "$REPO/experiments/paper1_placement_profile_v1"
"$P1PY" profile_placements.py --smoke --device cpu \
  --output /absolute/path/to/new-synthetic-profiler-check
```

The synthetic fixture tests check source joins, annotation preservation,
condition scoring, protocol consistency, tamper detection, and checkpoint-path
preflight. Full saved-model inference and hardware timing need the real images,
weights, and matching GPU environment. The source archive's old validation logs
are excluded from this release; current verification is recorded separately.
The profiler smoke command exercises strict saved-state reconstruction, MAC/FLOP
cross-checks, decoding, and NMS with random synthetic inputs. These measurements
are software checks rather than manuscript accuracy or timing results.
