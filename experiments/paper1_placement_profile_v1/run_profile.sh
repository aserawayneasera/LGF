#!/usr/bin/env bash
set -euo pipefail
PROFILE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P1_PROFILE_PY="${P1_PROFILE_PY:-/nas.dbms/asera/paper1_ugs_env/bin/python}"
if [[ ! -x "$P1_PROFILE_PY" ]]; then
  P1_PROFILE_PY="$(command -v python)"
fi
"$P1_PROFILE_PY" -u "$PROFILE_ROOT/profile_placements.py" "$@"
