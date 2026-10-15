#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
P1_UGS_PY="${P1_UGS_PY:-$(command -v python)}"
P1_GATE="$SCRIPT_DIR/paper1_gate_visualization_conditions_v2.py"
WEATHER_INDEX="${P1_WEATHER_INDEX:-$SCRIPT_DIR/coco_weather_image_conditions.csv}"
PKG="${P1_LGF_PACKAGE:-$SCRIPT_DIR/../lgf_original_controls_v1/source_runner}"
DATA="${P1_DATA_ROOT:-$REPO_ROOT/data}"
CKPT="${P1_GATE_CHECKPOINT:-${P1_UGS_RUN_ROOT:-$REPO_ROOT/runs/ugs_comparison}/coco_weather/lgf_original/seed_42/best_ap_small.pth}"
OUT="${P1_GATE_OUTPUT:-$REPO_ROOT/results/gate_visualization/reviewer4_paired_seed42_v2}"

# Prefer the complete manifest when present. The original fog-only JSON also works
# because WEATHER_INDEX supplies the paired rain and snow filenames and IDs.
WEATHER_ANN="${P1_WEATHER_ANN:-$REPO_ROOT/results/weather_evaluation_coco_three_v1/manifests/mini_val2017_weather_fog_rain_snow.json}"
if [ ! -f "$WEATHER_ANN" ]; then
  WEATHER_ANN="$DATA/COCO/weather-mini/annotations/mini_val2017_weather-500k6c.json"
fi

test -x "$P1_UGS_PY" || { echo "Python not found: $P1_UGS_PY" >&2; exit 1; }
test -f "$P1_GATE" || { echo "Script not found: $P1_GATE" >&2; exit 1; }
test -f "$WEATHER_INDEX" || { echo "Audited condition index not found: $WEATHER_INDEX" >&2; exit 1; }
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }
test -f "$WEATHER_ANN" || { echo "Weather annotations not found: $WEATHER_ANN" >&2; exit 1; }

COMMON=(
  --checkpoint "$CKPT"
  --package-root "$PKG"
  --data-root "$DATA"
  --weather-ann "$WEATHER_ANN"
  --weather-index "$WEATHER_INDEX"
  --output "$OUT"
  --device cuda:0
  --level C3
  --num-scenes 4
  --seed 42
)

"$P1_UGS_PY" -u "$P1_GATE" "${COMMON[@]}" --plan-only
"$P1_UGS_PY" -u "$P1_GATE" "${COMMON[@]}"
