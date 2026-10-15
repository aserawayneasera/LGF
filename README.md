# Local-global Fusion for Small-object Detection

Updated research source collection for the October 2026 revision of **PRLETTERS-D-26-00171**. This release combines the existing LGF repository with the recovered experimental code used during revision. The source supports training, checkpoint evaluation, recent-detector comparisons, statistical analysis, weather analysis, diagnostics, and placement profiling.

The manuscript reports validation results. Official-test prediction export is included as a separate workflow. Exporting predictions does not supply independent-test AP.

## Start here

1. Read [installation and training](docs/training_and_evaluation.md).
2. Follow [weather and placement workflows](docs/weather_and_placement.md).
3. Follow [gate controls, bootstrap, maps, and failure analysis](docs/diagnostics.md).
4. Use [the experiment inventory](docs/experiment_inventory.md) to match code to manuscript tables and reviewer comments.
5. Read [reproducibility and source provenance](docs/reproducibility.md) before interpreting regenerated results.
6. Follow [GitHub upload instructions](docs/github_upload.md) to update the existing repository.

## Installation

Use Python 3.10 or later. The recovered runners enforce **PyTorch 2.10.x and Torchvision 0.25.x**. Use a matching CUDA build for your GPU. A CPU environment supports software checks without downloading pretrained weights.

```bash
python3 -m venv .venv
source .venv/bin/activate
# CPU software checks. Use the matching CUDA installation for research training.
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-detectors.txt
python -m pip install -r requirements-profile.txt
python run.py doctor
python tools/check_release.py
python run.py list
```

The CUDA environment already used for the experiments is preferable for historical evaluations. Record its complete dependency versions with each run. Requirement ranges are installation constraints, not a frozen historical environment lock.

## Portable command launcher

Run from the repository root. The launcher forwards arguments to the recovered implementation and fills local **source-package paths** where needed. Dataset, checkpoint, output, and saved-run paths belong to you. Supply those paths explicitly. Relative paths resolve from your current working directory.

```bash
python run.py original -- --help
python run.py detectors -- --help
python run.py controls -- --help
python run.py bootstrap -- --help
python run.py weather -- evaluate --help
python run.py failures -- --help
python run.py profile -- --help
python run.py c4-weather -- --help
python run.py --print-command detectors -- train --model yolo11m --dataset coco_nw
```

Choose a separate interpreter when the detector environment differs:

```bash
python run.py --python /absolute/path/to/reviewer34_env/bin/python detectors -- check --device 0
```

| Workflow | Source and purpose |
|---|---|
| `original`, `ugs` | Archived RetinaNet/LGF/UGS model definitions with the revision training/evaluation runner |
| `detectors` | YOLO11m and RT-DETR-L native training and shared COCO evaluation |
| `se-cbam` | Separate SE and CBAM-style training/evaluation extension |
| `controls` | Native, uniform, shuffled, swapped, three fixed-alpha settings, and unscaled sum |
| `bootstrap` | 10,000 paired seed/image APsmall resamples and percentile intervals |
| `weather` | Per-condition and pooled scoring with fixed selected checkpoints |
| `prepare-weather` | Paired COCO fog/rain/snow preparation and source-scene audit |
| `gate-maps` | Matched clear/fog/rain/snow gate visualization |
| `acdc-gate-maps` | Additional ACDC inspection, separate from published matched-scene figures |
| `failures` | TP/FP/FN, precision/recall/F1 by size, and balanced examples |
| `multilevel` | Archived insertion experiments, preserving saved run labels |
| `profile` | Parameters, convolution/linear GMACs, and separately scoped timing |
| `c4-weather` | Full 1,500-image pooled weather scoring for selected C4 checkpoints |
| `official-export` | Hidden-label test prediction packaging or genuinely labelled holdout scoring |
| `legacy-train` | Unchanged older GitHub training script with historical absolute paths |

## Data and artifacts you supply

This source release excludes dataset images, trained checkpoints, raw prediction exports, manuscript PDFs, and private reviewer correspondence. Use your prepared COCO six-class and ACDC annotations, saved checkpoints, and actual run exports.

- Classes: person, car, truck, bicycle, bus, motorcycle. Numeric IDs come from your annotations.
- Manuscript COCO: 2,400 training scenes, 500 validation scenes.
- Later pooled synthetic evaluation: 500 fog + 500 rain + 500 snow images derived from the same 500 scenes.
- Manuscript ACDC: 2,737 training and 399 validation images. Conditions: fog 91, night 94, rain 112, snow 102.
- Seeds: 42, 1337, 2025.

Run a read-only annotation check before analysis:

```bash
python tools/audit_inputs.py --annotations /absolute/path/to/validation.json --require-positive-ann-ids
```

For COCO files following the 12-digit filename convention, check train/validation source-scene overlap:

```bash
python tools/audit_inputs.py --annotations /absolute/path/to/train.json /absolute/path/to/val.json \
  --train /absolute/path/to/train.json --val /absolute/path/to/val.json
```

The audit rejects unresolved source filenames rather than treating missing lineage as evidence of independence.

## Protocols retained

The original main-table results stay distinct from the later reviewer experiments. The primary comparison and frozen controls use APsmall selection. Condition and failure analyses use best overall AP selection. Follow each workflow's documented rule rather than mixing checkpoint families.

The original notebook definitions use 800/1333 aspect-ratio-preserving resize and custom anchors. YOLO11m/RT-DETR-L use native 1280-square training and their recorded shared evaluation procedure. UGS is an independent RetinaNet reimplementation, not an author-code reproduction. CBAM is the recovered CBAM-style variant.

C3 and C4 refer to the **saved run labels** used in the manuscript. The released archived code preserves those labels and routes. The published placement comparison covers C3/C4. C5 and multilevel options remain in the historical runner for traceability and do not provide verified manuscript results.

Original FLOP counts, reference-image GMAC counts, batch-one timing, and amortized batch-16 time have different scopes. Keep their names, units, hardware, and input geometry in exported reports.

## Tests and provenance

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
python run.py original -- smoke --synthetic --device cpu --num-workers 0
```

Package-specific tests and exact commands are documented in [validation](docs/validation.md). Tests use synthetic fixtures and do not recreate the manuscript's training results. The current release's recorded checks are distinguished from historical archive reports.

All archived model cores retain their recorded SHA-256. [Maintenance fixes](docs/maintenance_fixes.md) describe the shared SE/CBAM export adapter correction. Source changes outside the frozen model cores are recorded separately in `provenance/`.

## Citation and license

Use `CITATION.cff`. This revision is associated with the manuscript *Local-global Fusion for Small-object Detection*, PRLETTERS-D-26-00171. No publication DOI has been inserted. The existing MIT license is retained. External datasets, weights, and dependencies retain their own terms.

Existing repository: https://github.com/aserawayneasera/LGF
