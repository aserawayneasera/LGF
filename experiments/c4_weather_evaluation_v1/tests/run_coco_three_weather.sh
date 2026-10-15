#!/usr/bin/env bash
# Additional evaluation on existing validation scenes. No training or source edits.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P1PY="${P1PY:-/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python}"
P34PY="${P34PY:-/tmp/asera/paper1_reviewer34_env/bin/python}"
WEATHER="${WEATHER:-/nas.dbms/asera/paper1_weather_eval.py}"
PKG="${PKG:-/nas.dbms/asera/NEW/paper1_all_codes_with_test_evaluation}"
DATA="${DATA:-/tmp/asera/paper1_original/DATASET}"
SOURCE_OUT="${SOURCE_OUT:-/nas.dbms/asera/paper1_original/weather_evaluation_v1-2}"
OUT3="${OUT3:-/nas.dbms/asera/paper1_original/weather_evaluation_coco_three_v1}"
UGS_RUNS="${UGS_RUNS:-/tmp/asera/paper1_original/ugs_comparison/runs}"
RECENT_RUNS="${RECENT_RUNS:-/tmp/asera/paper1_original/reviewer34/runs}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-/tmp/asera/paper1_original/NEW}"
BASELINE_SOURCE="${BASELINE_SOURCE:-ugs}"
LGF_SOURCE="${LGF_SOURCE:-ugs}"
GPU_DEVICE="${GPU_DEVICE:-0}"
EVAL_BATCH="${EVAL_BATCH:-4}"
ANNOTATIONS="$OUT3/manifests/mini_val2017_weather_fog_rain_snow.json"

ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=(baseline lgf_original); fi

if [ "$ACTION" = help ] || [ "$ACTION" = --help ]; then
  printf '%s\n' \
    'Usage: bash run_coco_three_weather.sh prepare' \
    '       bash run_coco_three_weather.sh check [METHOD ...]' \
    '       bash run_coco_three_weather.sh run [METHOD ...]' \
    '       bash run_coco_three_weather.sh summarize' \
    'Default methods: baseline lgf_original. Seeds: 42 1337 2025.' \
    'Other methods: ugs_reimpl yolo11m rtdetr-l se cbam.' \
    'Use environment variables to override the paths at the top of this file.'
  exit 0
fi
[[ -x "$P1PY" ]] || { printf 'Missing Python: %s\n' "$P1PY" >&2; exit 2; }

if [ "$ACTION" = prepare ]; then
  EXTRA_TRAIN=()
  CLEAR_TRAIN="$DATA/COCO/non_weather-mini/annotations/mini_train2017_non_weather-2400k6c.json"
  # Check the current clear training manifest too, when present. Historical
  # training inputs and pretraining overlap remain separate questions.
  if [ -f "$CLEAR_TRAIN" ]; then EXTRA_TRAIN=(--train-annotations "$CLEAR_TRAIN"); fi
  "$P1PY" -u "$HERE/prepare_coco_weather_variants.py" \
    --audit "$SOURCE_OUT/audits/coco_weather/condition_audit.json" \
    --output-dir "$OUT3/manifests" \
    --expect-sources 500 \
    --confirm-aligned-variants \
    "${EXTRA_TRAIN[@]}"
  exit 0
fi

[[ -f "$WEATHER" ]] || { printf 'Missing existing evaluator: %s\n' "$WEATHER" >&2; exit 2; }
if [ "$ACTION" = summarize ]; then
  "$P1PY" -u "$WEATHER" summarize --output-root "$OUT3"
  exit 0
fi
if [ "$ACTION" != check ] && [ "$ACTION" != run ]; then
  printf 'Unknown action: %s. Use prepare, check, run, or summarize.\n' "$ACTION" >&2
  exit 2
fi
[[ -f "$ANNOTATIONS" ]] || { printf 'Run prepare first. Missing: %s\n' "$ANNOTATIONS" >&2; exit 2; }
# Read the verified image root recorded during preparation instead of guessing.
IMAGES="$("$P1PY" - "$OUT3/manifests/preparation_report.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as stream:
    print(json.load(stream)['image_root'])
PY
)"
COMMON=(
  --datasets coco_weather
  --train-dataset coco_weather
  --annotations "$ANNOTATIONS"
  --images "$IMAGES"
  --split validation
  --data-root "$DATA"
  --methods "${MODELS[@]}"
  --seeds 42 1337 2025
  --selection best_map
  --level C3
  --package-root "$PKG"
  --legacy-python "$P1PY"
  --recent-python "$P34PY"
  --baseline-source "$BASELINE_SOURCE"
  --lgf-source "$LGF_SOURCE"
  --ugs-root "$UGS_RUNS"
  --recent-root "$RECENT_RUNS"
  --original-root "$ORIGINAL_ROOT"
  --device "$GPU_DEVICE"
  --batch-size "$EVAL_BATCH"
  --output-root "$OUT3"
)
# Explicit annotations require one dataset. ACDC remains in the original output.
"$P1PY" -u "$WEATHER" evaluate "${COMMON[@]}" --check
if [ "$ACTION" = run ]; then
  "$P1PY" -u "$WEATHER" evaluate "${COMMON[@]}"
fi
