#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${P1_UGS_PY:=/tmp/asera/paper1_ugs_env/bin/python}"
method="${1:-ugs_reimpl}"
dataset_choice="${2:-coco_nw}"
case "$method" in ugs_reimpl|baseline|lgf_original) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
case "$dataset_choice" in
  all) datasets=(coco_nw coco_weather acdc) ;;
  coco_nw|coco_weather|acdc) datasets=("$dataset_choice") ;;
  *) echo "Unknown dataset: $dataset_choice" >&2; exit 2 ;;
esac
mkdir -p /tmp/asera/paper1_original/ugs_comparison/console_logs
for dataset in "${datasets[@]}"; do
  for seed in 42 1337 2025; do
    log="/tmp/asera/paper1_original/ugs_comparison/console_logs/${method}_${dataset}_s${seed}_$(date -u +%Y%m%dT%H%M%SZ).log"
    "$P1_UGS_PY" -u run.py train --method "$method" --dataset "$dataset" --seed "$seed" --device cuda 2>&1 | tee "$log"
  done
done
"$P1_UGS_PY" run.py summarize
