#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
P1_UGS_PY="${P1_UGS_PY:-$(command -v python)}"
P1_ACDC="$SCRIPT_DIR/paper1_acdc_gate_visualization.py"
PKG="${P1_LGF_PACKAGE:-$SCRIPT_DIR/../lgf_original_controls_v1/source_runner}"
DATA="${P1_DATA_ROOT:-$REPO_ROOT/data}"
AUDIT="${P1_ACDC_INDEX:-$SCRIPT_DIR/acdc_image_conditions.csv}"
CKPT="${P1_GATE_CHECKPOINT:-${P1_UGS_RUN_ROOT:-$REPO_ROOT/runs/ugs_comparison}/acdc/lgf_original/seed_42/best_ap_small.pth}"
OUT="${P1_GATE_OUTPUT:-$REPO_ROOT/results/gate_visualization/acdc_by_condition_seed42}"

test -x "$P1_UGS_PY" || { echo "Python not found: $P1_UGS_PY" >&2; exit 1; }
test -f "$P1_ACDC" || { echo "Script not found: $P1_ACDC" >&2; exit 1; }
test -f "$AUDIT" || { echo "ACDC condition audit not found: $AUDIT" >&2; exit 1; }
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }

COMMON=(
  --checkpoint "$CKPT"
  --package-root "$PKG"
  --data-root "$DATA"
  --condition-index "$AUDIT"
  --output "$OUT"
  --level C3
  --device cuda:0
  --selection-seed 42
  --per-condition 2
)

"$P1_UGS_PY" -u "$P1_ACDC" "${COMMON[@]}" --plan-only
"$P1_UGS_PY" -u "$P1_ACDC" "${COMMON[@]}"
