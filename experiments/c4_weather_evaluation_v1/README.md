# Evaluate C4 on fog, rain and snow

This package evaluates your saved C4 EMA checkpoints for seeds 42, 1337 and
2025 on the existing weather images. Each checkpoint processes 500 fog,
500 rain and 500 snow images. The images represent the same 500 source scenes.
The default selection is `best_ap_small`, matching the C4 protocol in S13.
Each seed uses one checkpoint across all three conditions.

The package also supports a matched evaluation of the saved runs C3 and C4.
Tables use your saved run names. The checkpoint loader reconstructs the saved
module keys and loads every EMA tensor strictly. The model source, loader,
manifest builder and COCO scoring functions come from your existing packages.

## Run on shinjo

Save `c4_weather_evaluation_v1.zip` in `/nas.dbms/asera/NEW`. Activate the
Python environment used for your placement experiments, then run:

```bash
cd /nas.dbms/asera/NEW
unzip -n c4_weather_evaluation_v1.zip
cd c4_weather_evaluation_v1

bash run_c4_weather.sh prepare
bash run_c4_weather.sh check
bash run_c4_weather.sh run
```

`prepare` references the existing geometry-preserving fog/rain/snow images.
The launcher supplies the alignment assertion used by the previous builder.
It creates new JSON manifests and checks all image files, dimensions, source
joins and training/validation source overlap. Original images and annotation
files retain their contents. Missing or altered variants stop preparation.

`check` verifies the manifests, all 1,500 image hashes, class mapping and the
three checkpoint paths. It performs no detector inference.

`run` evaluates C4, reports fog/rain/snow separately and computes `ALL` by
pooling images before COCO scoring. Each seed runs inference once over the
1,500 images. `ALL` is a pooled score, rather than an average of condition APs.

For matched C3/C4 placement results, use:

```bash
bash run_c4_weather.sh check --placements C3 C4
bash run_c4_weather.sh run --placements C3 C4
```

The second command evaluates six checkpoints: C3/C4 times three seeds.
Existing verified C4 results are reused. Results use a separate folder from
the training runs and existing manuscript tables. No training runs start.

## Default paths

| Input | Path |
| --- | --- |
| Source validation JSON | `/nas.dbms/asera/paper1_original/DATASET/COCO/weather-mini/annotations/mini_val2017_weather-500k6c.json` |
| Images with all three filename variants | `/nas.dbms/asera/paper1_original/DATASET/COCO/weather-mini/images/val2017_weather-500k6c` |
| Training class order | `/nas.dbms/asera/paper1_original/DATASET/COCO/weather-mini/annotations/mini_train2017_weather-2400k6c.json` |
| C4 checkpoints | `/nas.dbms/asera/paper1_original/ugs_comparison/runs_C4/coco_weather/lgf_original/seed_SEED/best_ap_small.pth` |
| C3 checkpoints | `/nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_weather/lgf_original/seed_SEED/best_ap_small.pth` |
| New output folder | `/nas.dbms/asera/paper1_original/ugs_comparison/c4_weather_evaluation_v1` |

The checkpoint paths come from the uploaded profiling results. The image root
and source JSON match the corresponding annotation audit. File existence on
your server is verified when you run the commands.

Override paths by adding `--data-root`, `--runs-base`, `--output`, `--images`,
`--source-annotations`, or `--train-annotations` to every applicable command.
An alternate Python executable uses `C4_WEATHER_PY=/absolute/path/to/python`.

For a different checkpoint selection, specify `--selection best_map` for
check/run/summarize and use a separate `--output` folder. Keep the selection
fixed across every condition. The Table S8 experiment used `best_map`; the
default run here uses `best_ap_small` to extend Table S17.

## Results to upload

The final console line provides the results ZIP path. The default is:

`/nas.dbms/asera/paper1_original/ugs_comparison/c4_weather_evaluation_v1.zip`

The ZIP contains:

- `summary/placement_weather_per_seed.csv`: all six AP and six AR metrics.
- `summary/placement_weather_mean_std.csv`: means and sample SDs over seeds.
- `summary/c4_minus_c3_percentage_points.csv`: matched placement differences,
  when both saved runs were evaluated.
- `summary/fog_reproduction_checks.json`: new fog scores compared with the
  selected checkpoints' saved fog metrics. Differences beyond 0.0001 AP are
  marked `different` for review before publication.
- Per-checkpoint predictions, metrics, hashes and inference settings.
- New evaluation manifests, source groups and image hashes.

The original S13 fog-only values remain historical results until the new
evaluation completes. Add the condition results as a separate validation
analysis, and identify its checkpoint-selection rule. These results measure
performance on additional conditions; claims about unseen-weather transfer
require evidence about the weather conditions used during training.

## Runtime and verification

Use the existing placement environment: PyTorch 2.10.x, torchvision 0.25.x,
NumPy, OpenCV, albumentations, Pillow and pycocotools. The runner uses EMA,
FP32, 800/1333 model resizing and the saved detector's score/NMS/output limits.
No FLOP counter is needed. Batch size defaults to four.

Local validation covers paired-scene preparation, annotation preservation,
source immutability, missing/changed images, class-map mismatches, checkpoint
identity, incomplete seed groups, sample SDs and paired percentage-point
reporting. See `VALIDATION.txt` for the executed checks. The user's checkpoint
weights, weather images and CUDA environment are on shinjo; actual detector
inference and numerical COCO scoring require the server run above.

To repeat the local checks in your evaluation environment:

```bash
WEATHER="$PWD/weather_metrics.py" python -m unittest discover -s tests -v
```

Reference implementations:

- Torchvision 0.25 RetinaNet:
  https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/detection/retinanet.html
- COCO bounding-box evaluator:
  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
