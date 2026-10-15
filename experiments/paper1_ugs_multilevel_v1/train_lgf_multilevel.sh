#!/usr/bin/env bash
# Launch true simultaneous pre-FPN C3+C4 and/or C3+C4+C5 training.
# Does not overwrite the existing single-level runner, outputs, or checkpoints.
set -uo pipefail

P1_UGS_PY="${P1_UGS_PY:-/nas.dbms/asera/paper1_ugs_env/bin/python}"
P1_UGS_ROOT="${P1_UGS_ROOT:-/nas.dbms/asera/paper1_original/ugs_comparison}"
P1_DATA_ROOT="${P1_DATA_ROOT:-/nas.dbms/asera/paper1_original/DATASET}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run.py"

if [[ ! -x "$P1_UGS_PY" || ! -f "$RUNNER" || ! -d "$P1_DATA_ROOT" ]]; then
  printf 'Missing Python, runner, or dataset root:\n  python: %s\n  runner: %s\n  data:   %s\n' \
    "$P1_UGS_PY" "$RUNNER" "$P1_DATA_ROOT" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  levels=(C3_C4 C3_C4_C5)
else
  levels=("$@")
fi

for level in "${levels[@]}"; do
  case "$level" in
    C3_C4|C3_C4_C5) ;;
    *) echo "Unsupported multi-level configuration: $level" >&2; exit 2 ;;
  esac
done

failed=0
for level in "${levels[@]}"; do
  run_root="$P1_UGS_ROOT/runs_${level}"
  log_root="$P1_UGS_ROOT/console_logs_multilevel"
  mkdir -p "$log_root"

  for dataset in acdc coco_nw coco_weather; do
    for seed in 42 1337 2025; do
      method=lgf_original
      run_dir="$run_root/$dataset/$method/seed_$seed"
      log_file="$log_root/${method}_${level}_${dataset}_seed${seed}.log"
      printf '\n========================================\n%s %s %s seed %s\n========================================\n' \
        "$method" "$level" "$dataset" "$seed"

      if [[ -f "$run_dir/completed.json" ]]; then
        echo "Already completed: $run_dir"
        continue
      fi

      resume_args=()
      if [[ -d "$run_dir" ]]; then
        if [[ -f "$run_dir/last.pth" ]]; then
          echo "Resuming: $run_dir/last.pth"
          resume_args=(--resume "$run_dir/last.pth")
        else
          echo "Skipping pre-existing run without a last.pth: $run_dir" | tee -a "$log_file"
          echo "Inspect this folder before removing or restarting its contents."
          failed=$((failed + 1))
          continue
        fi
      fi

      "$P1_UGS_PY" -u "$RUNNER" train \
        --method "$method" \
        --dataset "$dataset" \
        --level "$level" \
        --seed "$seed" \
        --device cuda \
        --data-root "$P1_DATA_ROOT" \
        --run-root "$run_root" \
        --epochs 80 \
        --batch-size 4 \
        --accum-steps 4 \
        --base-lr 0.005 \
        --no-amp \
        "${resume_args[@]}" \
        2>&1 | tee -a "$log_file"

      rc=${PIPESTATUS[0]}
      if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: $method $level $dataset seed=$seed (exit $rc). Log: $log_file"
        failed=$((failed + 1))
      else
        echo "FINISHED: $method $level $dataset seed=$seed"
      fi
    done
  done
done

if [[ "$failed" -ne 0 ]]; then
  echo "$failed run(s) failed or were skipped, inspect the log directory." >&2
  exit 1
fi

echo "All requested multi-level experiments completed or were previously complete."
