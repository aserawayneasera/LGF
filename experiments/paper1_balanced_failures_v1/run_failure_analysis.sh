#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P1_FAILURE_PY="${P1_FAILURE_PY:-${P1_UGS_PY:-$(command -v python)}}"
if [[ ! -x "$P1_FAILURE_PY" ]]; then
  printf 'Python environment not found: %s\nSet P1_FAILURE_PY to your working Paper 1 Python.\n' "$P1_FAILURE_PY" >&2
  exit 1
fi
exec "$P1_FAILURE_PY" -u "$SCRIPT_DIR/paper1_failure_analysis.py" "$@"
