#!/usr/bin/env bash
set -euo pipefail

repo="${CET_REPO:-/e/yky/FraudGT_cet_reliability}"
output="${CET_OUTPUT:-/e/yky/FraudGT_cet_results/dynamic_reliability}"
python="${CET_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python3.9}"
spec="$repo/run/cet_dynamic_reliability_spec.json"

mkdir -p "$output"
pids=()
for task in 0 1 2 3 4 5 6; do
  gpu="$task"
  log="$output/task${task}_gpu${gpu}.stdout"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    "$python" "$repo/run/cet_dynamic_reliability_audit.py" \
    --spec "$spec" \
    --task-index "$task" \
    --output-dir "$output" \
    --device cuda:0 >"$log" 2>&1 &
  pids+=("$!")
  printf 'started task=%s gpu=%s pid=%s log=%s\n' \
    "$task" "$gpu" "${pids[-1]}" "$log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
