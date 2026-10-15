# Paper 1: SE/CBAM extension, version 1.0.2

This extension calls the existing `paper1_ugs_comparison_v1/run.py` training loop.
It adds explicit model selection for the archived SE/CBAM blocks, method labels,
checkpoint provenance, validation evaluation, and shared test-export support.
It does not replace the original training loop or copy a different recipe.

## 1. Extract beside your existing package

Save `paper1_se_cbam_extension_nas_v1_0_2.zip` in `/nas.dbms/asera/NEW`, then run:

```bash
cd /nas.dbms/asera/NEW
unzip paper1_se_cbam_extension_nas_v1_0_2.zip
cd paper1_se_cbam_extension_nas_v1_0_2
sha256sum -c CHECKSUMS.sha256

export P1_UGS_PACKAGE=/nas.dbms/asera/NEW/paper1_all_codes_with_test_evaluation/paper1_ugs_comparison_v1
export P1_UGS_PY=/nas.dbms/asera/paper1_ugs_env/bin/python
export TORCH_HOME=/nas.dbms/asera/paper1_original/torch_cache
```

Keep this extension directory OUTSIDE `paper1_ugs_comparison_v1`.
The original runner hashes every Python file beneath its package. Adding files
inside the original package would change its source fingerprint and invalidate
existing strict resumes. This extension imports the existing code and changes
only the SE/CBAM selections in its own Python process. No original `.py`, notebook,
configuration, data, checkpoint, or existing experiment result is overwritten.

The UGS package remains version 1.0.0. The recovery_1_0_1 repair applies to the
reviewer34 package, not the UGS runner's version field.

## 2. Check your existing environment

Reuse the Python environment used for your successful LGF/baseline/UGS runs.
There is no dependency upgrade or new PyTorch installation.

For the reported first-job failure, first compare the recorded LGF environment
with the environment of the Python interpreter launching training:

```bash
"$P1_UGS_PY" compare_environment.py \
  --package "$P1_UGS_PACKAGE" \
  --reference /nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_nw/lgf_original/seed_42/protocol.json
```

The report shows each blocking difference. If the only difference is `gpu`,
this version prints it and proceeds. Python, package versions, CUDA availability,
and PyTorch CUDA build must match the reference. If they differ, run with the
original matching environment or inspect the difference before training. The
saved reference protocol and its signature are never rewritten. The new SE/CBAM
protocol records the actual GPU for its run.

```bash
"$P1_UGS_PY" run_se_cbam.py --package "$P1_UGS_PACKAGE" check --device cuda
```

The original dependency requirements remain torch 2.10.x, torchvision 0.25.x,
Albumentations 2.0.8, pycocotools, TensorBoard, matplotlib and tqdm.

Check both models with synthetic images before launching the training grid:

```bash
for METHOD in se cbam; do
  "$P1_UGS_PY" -u run_se_cbam.py --package "$P1_UGS_PACKAGE" \
    smoke --method "$METHOD" --synthetic --device cuda \
    --output "/nas.dbms/asera/paper1_original/ugs_comparison/smoke_${METHOD}.json" || break
done
```

The synthetic smoke uses the existing runner's two training phases, inference and
checkpoint-reload checks. It uses 64-pixel synthetic inputs only for the software
check. Real training retains the original 800/1333 transform.

## 3. Train both methods, all three datasets and all three seeds

Preview all 18 commands:

```bash
"$P1_UGS_PY" -u train_se_cbam_grid.py \
  --package "$P1_UGS_PACKAGE" \
  --check
```

Run the same grid:

```bash
"$P1_UGS_PY" -u train_se_cbam_grid.py \
  --package "$P1_UGS_PACKAGE"
```

The grid is SE/CBAM x coco_nw/coco_weather/acdc x 42/1337/2025.
Jobs run sequentially, not as 18 concurrent GPU processes. Rerunning the command
resumes incomplete runs from `last.pth` and skips completed matching runs.
Missing or conflicting metadata causes an error instead of overwriting a run.

For each dataset and seed, the launcher reads the matching `lgf_original`
`protocol.json`. It copies epochs, batch size, accumulation, learning rate, AMP,
worker count, prefetch factor and insertion-level label. The reference must come
from the UGS runner. Original-notebook logs are not interchangeable references.

Automatic reference lookup checks these roots:

```text
/nas.dbms/asera/paper1_original/ugs_comparison/runs
```

An explicit `--run-root` is checked first. Duplicate references with different
protocol fingerprints are rejected. Select the intended root explicitly:

```bash
"$P1_UGS_PY" -u train_se_cbam_grid.py \
  --package "$P1_UGS_PACKAGE" \
  --reference-root /nas.dbms/asera/paper1_original/ugs_comparison/runs
```

To match your baseline records instead, add `--reference-method baseline`.
`--threads` defaults to 4. Thread count was not recorded in the old training
protocol, so it is not automatically inherited. The original default was 4.

The preview checks paths, source fingerprints and saved settings. At real launch,
the grid checks *all* reference environments before the first training process.
Each run then checks training/validation data hashes and the detector's transform,
anchor and prediction settings against the reference. Only a GPU *name* difference
is accepted and reported. Version and CUDA-build differences stop with field-level
details. GPU timing comparisons need the same hardware; AP comparisons retain
matching training settings and code.

For one initial run with the matching LGF settings:

```bash
"$P1_UGS_PY" -u train_se_cbam_grid.py \
  --package "$P1_UGS_PACKAGE" \
  --methods se --datasets coco_nw --seeds 42
```

## 4. Output paths

Defaults use the NAS storage workflow:

```text
/nas.dbms/asera/paper1_original/DATASET/
/nas.dbms/asera/paper1_original/ugs_comparison/runs/<dataset>/se/seed_<seed>/
/nas.dbms/asera/paper1_original/ugs_comparison/runs/<dataset>/cbam/seed_<seed>/
/nas.dbms/asera/paper1_original/ugs_comparison/logs/
```

Override with `--data-root`, `--run-root` and `--log-root` as needed. Reference
records and new outputs need not reside on the same disk, but data content must
match the reference. Paths alone do not establish matching data.

Each native run writes:

```text
last.pth
best_map.pth
best_map.metrics.json
best_ap_small.pth
best_ap_small.metrics.json
protocol.json
data_audit.json
history.jsonl
completed.json
evaluation/epoch_005/predictions.json
... further evaluation epochs ...
```

`data_audit.json` also records the reference protocol path and SHA-256 when the
matched-grid route is used. Checkpoints retain the native schema, including
`config`, `model_state_dict`, `ema_state_dict`, optimizer/scheduler/RNG state and
both selection rules. They additionally identify the SE/CBAM extension source.

To watch training:

```bash
tail -n 100 -f /nas.dbms/asera/paper1_original/ugs_comparison/logs/se_coco_nw_seed42.log
```

## 5. Individual native-style commands

These commands use NAS storage defaults and the original training defaults. The matched grid above is preferred
when the goal is to inherit settings from already trained LGF/baseline records.

```bash
"$P1_UGS_PY" -u run_se_cbam.py --package "$P1_UGS_PACKAGE" \
  train --method se --dataset coco_nw --level C3 --seed 42 --device cuda

"$P1_UGS_PY" -u run_se_cbam.py --package "$P1_UGS_PACKAGE" \
  train --method cbam --dataset coco_nw --level C3 --seed 42 --device cuda
```

All existing native training flags are forwarded. To add a reference assertion
to an individual command, supply `--reference-protocol /path/protocol.json` and
set its recorded options explicitly. This flag validates settings; it does not
change them. The grid launcher handles that copying automatically.

Do not add `--method se` to the old `run.py`: the old file deliberately remains
unchanged. Continue using your existing commands for baseline, LGF, UGS, YOLO11m
and RT-DETR-L. They do not need retraining because this extension was added.

## 6. Validation evaluation and aggregation

Training already saves predictions during validation. For an explicit rerun:

```bash
"$P1_UGS_PY" -u run_se_cbam.py --package "$P1_UGS_PACKAGE" \
  evaluate --method se --dataset coco_nw --level C3 --seed 42 --device cuda \
  --checkpoint /nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_nw/se/seed_42/best_map.pth \
  --output /nas.dbms/asera/paper1_original/ugs_comparison/validation_exports/coco_nw/se/seed_42/best_map
```

Use `cbam` in the method and checkpoint path for CBAM. Choose a new output directory
for a rerun. Do not use `--original-checkpoint` for checkpoints trained here.

Aggregate completed native UGS-runner results, including the new methods:

```bash
"$P1_UGS_PY" run_se_cbam.py --package "$P1_UGS_PACKAGE" \
  summarize --selection best_map \
  --run-root /nas.dbms/asera/paper1_original/ugs_comparison/runs \
  --output /nas.dbms/asera/paper1_original/ugs_comparison/summary_with_se_cbam_best_map
```

The original summary requires matching settings within each method's seed group.
All methods to be aggregated must be present under the selected run root.

## 7. Independent test exports

Use `evaluate_test_se_cbam.py` for the NEW runner checkpoints. The earlier
`run_paper1_test_grid.py` assumes notebook checkpoints for SE/CBAM and does not
select these new run directories automatically. Do not use that old automatic
SE/CBAM route for the new checkpoints.

This entry point installs the SE/CBAM construction adapter in memory, then calls
the original shared test exporter. Its manifests, checkpoint freezing, overlap
checks, category handling, EMA weights, resizing, NMS, detection limits and scoring
functions remain unchanged. Extension source hashes enter its export signature.

Example using an already prepared COCO non-weather test manifest:

```bash
"$P1_UGS_PY" -u evaluate_test_se_cbam.py \
  --package "$P1_UGS_PACKAGE" \
  export --method se --train-dataset coco_nw --seed 42 --level C3 --device 0 \
  --manifest /nas.dbms/asera/paper1_original/TEST_DATASET/coco_nw_test.json \
  --data-root /nas.dbms/asera/paper1_original/DATASET \
  --checkpoint /nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_nw/se/seed_42/best_map.pth \
  --selection best_map \
  --selection-note 'Best validation mAP from the completed SE runner training' \
  --output /nas.dbms/asera/paper1_original/test_evaluations/se_cbam_runner/coco_nw/se/seed_42
```

Point `--manifest` to your existing manifest if stored elsewhere. The default
shared evaluation directory is the sibling `paper1_test_evaluation_v1`; override
with `--eval-dir` when needed. Set `--method cbam` and update paths for CBAM.
Use a new output directory for the first export. Add `--resume` only for an
existing interrupted export with the same checkpoint and arguments.

The supplied exporter still distinguishes prediction exports from measured
scores. Hidden-label official tests do not produce local AP without ground truth.
Use one checkpoint-selection rule across compared methods. The example uses
best validation mAP, matching your existing test grid, not best APsmall.

## 8. Preserved scientific implementation

The archived `SEBlock` and `CBAMBlock` are used directly, at the existing insertion
point. No alternative SE/CBAM implementation was imported. The archived CBAM is
the determinism-oriented variant with softmax pooling, beta=20 and a 5x5 spatial
kernel. This extension preserves those actual choices; it does not claim an
unaltered canonical CBAM implementation.

The existing C3 label and its physical placement are preserved. There is no C3/C4
remapping change. Original checkpoint schemas and state names are preserved.

The same loss, heads-plus-attention phase, full-backbone unfreezing, warm-up,
scheduler reset, gradient clipping, EMA, validation cadence and best-checkpoint
rules are invoked from the original runner. SE/CBAM attention parameters are
trainable in its initial heads-plus-attention phase, as for the LGF block.

Strict restart checks remain. The old runner already states that multiworker
augmentation continuation is not guaranteed to be bitwise identical after resume.
This extension does not change or strengthen that guarantee.

## 9. Compatibility and checks

This version keeps the NAS storage defaults of version 1.0.1. The original UGS,
reviewer34 and test-evaluation package files are not edited. Dataset, source,
model and checkpoint checks remain. Never edit a saved reference `protocol.json`
or checkpoint to make an environment comparison pass.

The first failing job stopped during reference validation, before the native
trainer created its run directory. If you previously started any SE/CBAM runs
with version 1.0.1, keep that exact extension for those resumptions. Version
1.0.2 has a different source fingerprint and will reject a version 1.0.1 resume.
Extract this folder separately and do not overwrite an active extension. Existing
baseline/LGF/UGS runs continue to use their original runner.

To check this version's environment comparison and NAS routing:

```bash
python3 environment_tests.py
"$P1_UGS_PY" nas_path_tests.py
```

To run the original extension self-tests with your configured dependencies:

```bash
"$P1_UGS_PY" self_test.py \
  --package "$P1_UGS_PACKAGE" \
  --eval-dir "$(dirname "$P1_UGS_PACKAGE")/paper1_test_evaluation_v1" \
  --device cpu \
  --output /nas.dbms/asera/paper1_original/ugs_comparison/se_cbam_server_self_test.json
```

The environment comparison checks ran locally without PyTorch or server data.
Run the native self-test and one-dataset training on your own server before the
full grid if the diagnostic reports no blocking differences.
