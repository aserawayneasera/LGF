# Paper 1: evaluate the existing COCO fog, rain, and snow variants

This addition addresses the verified folder-versus-JSON mismatch. The user's
validation folder contains 500 fog, 500 rain, and 500 snow images. The current
validation JSON lists only the 500 fog images. The helper writes NEW evaluation
manifests referencing all three existing variants of the same 500 source images.

No original JSON, image, checkpoint, training code, or evaluation code is edited.
No images are generated, copied, or relabelled in place. No training is started.
The same saved checkpoint is evaluated across all conditions, with `best_map`
selection fixed. This is additional validation analysis, not an independent test.

## Files

- `prepare_coco_weather_variants.py`: prepares the manifests and source mapping.
- `run_coco_three_weather.sh`: uses the previously delivered evaluator and the
  paths from the current shinjo transcript. Supports prepare/check/run/summarize.
- `test_prepare.py`: local synthetic-fixture tests.
- `validation_log.txt`: local test results, not model performance results.

## Install

Save `paper1_coco_three_weather_v1.zip` on shinjo as
`/nas.dbms/asera/paper1_coco_three_weather_v1.zip`, then run:

```bash
unzip -n /nas.dbms/asera/paper1_coco_three_weather_v1.zip -d /nas.dbms/asera
```

The ZIP contains the directory `paper1_coco_three_weather_v1`.
Keep the existing `/nas.dbms/asera/paper1_weather_eval.py` unchanged.

## Prepare the additional manifests

```bash
bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh prepare
```

Preparation reads the existing audit at:

```
/nas.dbms/asera/paper1_original/weather_evaluation_v1-2/audits/coco_weather/condition_audit.json
```

The JSON and image directory are read from that audit, not guessed. The audit
must describe 500 validation source images with numeric COCO weather filenames.

The launcher asserts use of the original geometry-preserving fog/rain/snow
transformations. Do not use the launcher for independently cropped, resized,
warped, or unrelated images. Box reuse is based on these transformations and the
source identity in filenames. Equal dimensions alone do not prove alignment.

Preparation checks that every source has all three readable image counterparts
and the dimensions recorded by the source JSON. It hashes the files and checks
for byte-identical variants within a source. Missing, unreadable, mismatched, or
suspicious files cause failure instead of silently reducing the sample.

The helper preserves original fog image/annotation IDs when the source is fog.
Rain and snow receive globally unique IDs. Every variant has `source_group_id`
and `source_image_id`. Each annotation retains its original bbox, area,
category ID, crowd flag, and any segmentation data; `source_annotation_id` is
added for traceability. Original image metadata are recorded in the preparation
report. Old fog-specific path metadata are not copied into new rain/snow records
as contradictory condition labels.

The helper reads the current training JSON named in the original audit. The
launcher also checks the current unmodified-COCO training JSON when available.
It prints filename condition counts and stops on recognised training/validation
source-ID overlap. This is NOT a complete historical provenance, augmentation,
pretraining, or scene-independence audit. A fog-only current training JSON does
not by itself establish what every historical checkpoint saw during training.

New output directory:

```
/nas.dbms/asera/paper1_original/weather_evaluation_coco_three_v1/manifests/
    mini_val2017_weather_fog_rain_snow.json
    mini_val2017_weather_fog.json
    mini_val2017_weather_rain.json
    mini_val2017_weather_snow.json
    source_groups.csv
    preparation_report.json
```

The combined manifest has 1,500 variants of 500 source scenes. Each condition
has the original number of annotated objects, since geometry and annotations
are retained. The combined object count counts each source's objects three
times. Repeated source scenes are NOT independent samples. Use source-group
resampling for pooled/paired uncertainty analyses. The current evaluator does
not automatically perform that bootstrap.

Preparation refuses to overwrite a different existing manifest bundle. An
unchanged repeated preparation verifies the existing bytes. A different input
or interrupted/incomplete output requires a NEW output location, not forced
replacement of the historical files.

## Check, then run baseline versus LGF

```bash
bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh check
bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh run
```

`prepare` and `check` do not load models or use the GPU. `run` repeats preflight,
then invokes the existing evaluator sequentially. Defaults: baseline and
lgf_original, seeds 42/1337/2025, best_map, level C3, device 0, batch size 4.
Finish the existing GPU evaluation before starting another evaluation on the
same GPU. Detaching from a tmux session does not terminate its processes.

The original evaluator receives `--datasets coco_weather`, the new combined
`--annotations`, and the same verified `--images` directory. `--train-dataset`
is explicitly coco_weather. The original training JSON provides class order;
it is never replaced with the new evaluation manifest.

The outputs stay under the new `weather_evaluation_coco_three_v1` directory.
The old fog-only and ACDC outputs remain untouched. Do not point the expanded
analysis at the old output root: the evaluator's integrity checks reject a
changed dataset under an existing run.

## Compare additional saved methods

```bash
bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh check \
  baseline lgf_original ugs_reimpl yolo11m rtdetr-l

bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh run \
  baseline lgf_original ugs_reimpl yolo11m rtdetr-l
```

Previously completed jobs in this NEW output directory are skipped only after
the original evaluator verifies the recorded protocol and prediction hash.
Fog-only results from the OLD directory are not imported; the first new job
processes all 1,500 variants.

SE/CBAM are supported through original notebook checkpoints. Add `se cbam` only
when their saved checkpoint files exist. `ORIGINAL_ROOT` must directly contain
`BEST_<dataset>_C3_<method>_s<seed>_epoch_*_map_*.pth` files.

## Defaults and overrides

These defaults reproduce paths in the recent transcript. Their existence on
shinjo is checked by the original evaluator; no remote file access occurred
while this helper was written.

```
P1PY=/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python
P34PY=/tmp/asera/paper1_reviewer34_env/bin/python
WEATHER=/nas.dbms/asera/paper1_weather_eval.py
PKG=/nas.dbms/asera/NEW/paper1_all_codes_with_test_evaluation
DATA=/tmp/asera/paper1_original/DATASET
SOURCE_OUT=/nas.dbms/asera/paper1_original/weather_evaluation_v1-2
OUT3=/nas.dbms/asera/paper1_original/weather_evaluation_coco_three_v1
UGS_RUNS=/tmp/asera/paper1_original/ugs_comparison/runs
RECENT_RUNS=/tmp/asera/paper1_original/reviewer34/runs
ORIGINAL_ROOT=/tmp/asera/paper1_original/NEW
BASELINE_SOURCE=ugs
LGF_SOURCE=ugs
GPU_DEVICE=0
EVAL_BATCH=4
```

Use an exported variable or a command-local environment assignment for an
override. An ordinary unexported shell assignment is not passed to a child bash
script. For example:

```bash
P34PY=/actual/working/reviewer34/environment/bin/python \
  bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh check \
    lgf_original yolo11m
```

The baseline/UGS/LGF default checkpoint layout is
`UGS_RUNS/coco_weather/<method>/seed_<seed>/best_map.pth`.
The recent-detector layout is
`RECENT_RUNS/coco_weather/<method>/seed_<seed>/best_map.pt`.
`BASELINE_SOURCE=ugs` changes the checkpoint source, not the baseline architecture.

## Summaries and interpretation

```bash
bash /nas.dbms/asera/paper1_coco_three_weather_v1/run_coco_three_weather.sh summarize
```

See the new output root's `summary/weather_per_seed.csv`,
`summary/weather_mean_std.csv`, and, when matching baseline/LGF results exist,
`summary/lgf_minus_baseline.csv`. Condition rows: fog, rain, snow, ALL.
`ALL` is a pooled evaluation, not the arithmetic mean of the three condition APs.
The current evaluator writes pairwise deltas only for LGF versus baseline.

These results extend the current validation analysis. They do not establish
mixed-weather training, generalization to conditions never seen in training,
or independent-test performance. The historical fog-only manifest and its
results must remain accurately identified. Check training JSONs, data lineage,
checkpoint provenance, and training-time augmentations before revising training
claims. Do not choose new best checkpoints separately for each weather.

## Testing scope

18 local tests passed. They cover manifest IDs/grouping, annotation preservation,
read-only source handling, missing/corrupt/misaligned inputs, source overlap,
repeated execution, and compatibility with the supplied weather evaluator's audit
and shell preflight. The shell test uses placeholder checkpoint paths solely for
path checking. No real checkpoint is loaded in these tests.

Real detector inference and numerical COCO AP scoring were not tested in this
environment. The user's 1,500 image files are not present here; decoding and
shape checks occur when `prepare` runs on shinjo. The user's output establishes
filename existence, not pixel-level geometry verification.

To run the tests on a server with the existing evaluator:

```bash
cd /nas.dbms/asera/paper1_coco_three_weather_v1
WEATHER=/nas.dbms/asera/paper1_weather_eval.py \
  /nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python -m unittest -v test_prepare
```
