# Release validation

Fresh checks ran on 13 October 2026 using Python 3.12.14, CPU PyTorch 2.10.0 and Torchvision 0.25.0. Albumentations 2.0.8, pycocotools 2.0.11, Ultralytics 8.3.253, and fvcore 0.1.5.post20221221 supplied the main test dependencies. No historical datasets, trained checkpoints, official server submissions, or RTX 5090 timings were run.

## Completed checks

| Suite | Passing checks | Coverage |
|---|---:|---|
| Root release tools | 7 | Dispatch flags, argument forwarding, path behavior, annotation ID and scene checks |
| Original/UGS package | 25 | Model gradients, UGS behavior, evaluation, selection, checkpoint continuation |
| Recent detector package | 29 | COCO conversion, shared evaluation, exact adapter predictions, native synthetic interruption/resume |
| SE/CBAM model suite | 13 | Both attention builders, gradient phases, strict reload and resume |
| SE/CBAM environment/path suites | 15 | Dependency/layout and NAS contract checks |
| Shared official exporter | 24 | Export/scoring checks, repaired attention adapter, all seven random-weight model adapters at 800/1333 |
| Frozen gate controls | 11 | Equations, real synthetic RetinaNet execution, inventory and restart |
| Balanced failure analysis | 15 | COCO matching, size accounting, candidate selection, rendering and hashes |
| Paired bootstrap | 8 | Accumulation parity, duplicate draws, paired seed/image samples, crowd IoU fallback |
| Matched gate pairing | 5 | GT inheritance, geometry, conditions and missing-input rejection |
| General weather scoring | 22 | Condition AP, pooled scoring, audit and selection contracts |
| Three-weather preparation | 18 | Source-scene variants, annotations, lineage, restart and filename rules |
| C4 pooled weather | 35 | Preparation, report construction and preflight |
| Archived multilevel | 3 | Architecture, gradients and strict checkpoint reload |
| Total automated checks | 230 | Synthetic and software checks |

The archived profiling runner also passed its separate synthetic smoke: strict reconstruction, parameter formulas, convolution/linear MAC cross-checks, decoding/NMS, and batch-one/batch-two execution. This smoke exercises random-weight CPU models. Its timings are not manuscript measurements.

Python syntax, shell syntax, source-package checksums, all workflow entrypoints, and all five archived model-core identities were checked. CLI help and supported-environment checks also passed. The shared exporter initially skipped its opt-in model test during normal discovery; the same test then passed with `P1_REAL_MODEL_TESTS=1`, so the table counts successful unique tests rather than duplicate discoveries.

## Repeat checks

Install the development and profiling dependencies in the supported environment:

```bash
python -m pip install -r requirements-detectors.txt -r requirements-profile.txt -r requirements-dev.txt
python -m unittest discover -s tests -v
python tools/check_release.py
```

Run each package in a separate process because archived modules use short import names such as `data` and `utils`:

```bash
(cd experiments/paper1_ugs_comparison_v1 && python -m unittest discover -s tests -v)
P1_UGS_PACKAGE="$PWD/experiments/paper1_ugs_comparison_v1" \
  python -m unittest discover -s experiments/paper1_reviewer34_v1/tests -v
python experiments/paper1_se_cbam_extension_nas_v1_0_2/self_test.py \
  --package "$PWD/experiments/paper1_ugs_comparison_v1" --device cpu
python experiments/paper1_se_cbam_extension_nas_v1_0_2/environment_tests.py
python experiments/paper1_se_cbam_extension_nas_v1_0_2/nas_path_tests.py
python -m unittest discover -s experiments/paper1_test_evaluation_v1/tests -v
P1_REAL_MODEL_TESTS=1 python -m unittest discover \
  -s experiments/paper1_test_evaluation_v1/tests -p test_models.py -v
python -m unittest discover -s experiments/lgf_original_controls_v1/tests -v
python -m unittest discover -s experiments/paper1_balanced_failures_v1/tests -v
python -m unittest discover -s experiments/paper1_paired_bootstrap/tests -v
python -m unittest discover -s experiments/paper1_gate_maps_four_conditions_v2/tests -v
python -m pytest experiments/paper1_weather_eval_v1/tests -q
python -m unittest discover -s experiments/paper1_coco_three_weather_v1 -p test_prepare.py -v
WEATHER="$PWD/experiments/paper1_weather_eval_v1/paper1_weather_eval.py" \
  python -m unittest discover -s experiments/c4_weather_evaluation_v1/tests -v
(cd experiments/paper1_ugs_multilevel_v1 && python -m unittest discover -s tests -p test_multilevel.py -v)
```

If a package has a stricter current-working-directory contract, use its own README's exact test command. The profiled smoke command is in [weather and placement](weather_and_placement.md). Real-data reruns require the actual annotated datasets and saved checkpoint/export families identified in [reproducibility](reproducibility.md).
