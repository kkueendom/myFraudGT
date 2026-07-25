#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="${TIER_RELIABILITY_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${TIER_RELIABILITY_OUTPUT:?set TIER_RELIABILITY_OUTPUT}"
SPEC="${TIER_RELIABILITY_SPEC:-$REPO/run/tier_dynamic_reliability_spec.json}"
PYTHON="${TIER_RELIABILITY_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
REPEATS="${TIER_RELIABILITY_REPEATS:-}"
EVAL_STEP_CAP="${TIER_RELIABILITY_EVAL_STEP_CAP:-}"

mkdir -p "$OUTPUT"

run_task() {
  local gpu="$1"
  local task="$2"
  local command=(
    "$PYTHON" "$REPO/run/tier_dynamic_reliability_audit.py"
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

run_worker() {
  local gpu="$1"
  local initial_task="$2"
  run_task "$gpu" "$initial_task"
  if mkdir "$OUTPUT/.task3_claim" 2>/dev/null; then
    run_task "$gpu" 3
  fi
}

run_worker 0 4 &
run_worker 1 5 &
run_worker 2 6 &
run_worker 3 7 &
run_worker 4 0 &
run_worker 5 1 &
run_worker 6 2 &
wait
