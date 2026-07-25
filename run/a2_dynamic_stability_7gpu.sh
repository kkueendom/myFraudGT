#!/usr/bin/env bash
set -euo pipefail

REPO="${A2_STABILITY_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${A2_STABILITY_OUTPUT:?set A2_STABILITY_OUTPUT}"
SPEC="${A2_STABILITY_SPEC:-$REPO/run/a2_dynamic_stability_spec.json}"
PYTHON="${A2_STABILITY_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
REPEATS="${A2_STABILITY_REPEATS:-}"
EVAL_STEP_CAP="${A2_STABILITY_EVAL_STEP_CAP:-}"

mkdir -p "$OUTPUT"

run_task() {
  local gpu="$1"
  local task="$2"
  local command=(
    "$PYTHON" "$REPO/run/a2_dynamic_stability_audit.py"
    --spec "$SPEC"
    --task-index "$task"
    --output-dir "$OUTPUT"
    --device "cuda:$gpu"
  )
  if [[ -n "$REPEATS" ]]; then
    command+=(--repeats "$REPEATS")
  fi
  if [[ -n "$EVAL_STEP_CAP" ]]; then
    command+=(--eval-step-cap "$EVAL_STEP_CAP")
  fi
  "${command[@]}" >"$OUTPUT/task${task}.stdout" 2>"$OUTPUT/task${task}.stderr"
}

run_queue() {
  local gpu="$1"
  shift
  local task
  for task in "$@"; do
    run_task "$gpu" "$task"
  done
}

run_queue 0 4 &
run_queue 1 10 &
run_queue 2 5 &
run_queue 3 11 &
run_queue 4 2 0 1 &
run_queue 5 8 6 7 &
run_queue 6 3 9 &
wait
