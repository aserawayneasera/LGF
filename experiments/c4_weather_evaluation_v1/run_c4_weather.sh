#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
C4_WEATHER_PY="${C4_WEATHER_PY:-python}"
ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$ACTION" = help ] || [ "$ACTION" = --help ]; then
  printf '%s\n' \
    'Usage: bash run_c4_weather.sh prepare' \
    '       bash run_c4_weather.sh check [--placements C3 C4]' \
    '       bash run_c4_weather.sh run [--placements C3 C4]' \
    '       bash run_c4_weather.sh summarize [--placements C3 C4]' \
    'Default: C4, seeds 42 1337 2025, best_ap_small EMA weights.' \
    'Run from the Python environment used for your placement experiments.'
  exit 0
fi
if [ "$ACTION" = prepare ]; then
  "$C4_WEATHER_PY" -u "$HERE/c4_weather_eval.py" prepare --confirm-aligned-variants "$@"
else
  "$C4_WEATHER_PY" -u "$HERE/c4_weather_eval.py" "$ACTION" "$@"
fi
