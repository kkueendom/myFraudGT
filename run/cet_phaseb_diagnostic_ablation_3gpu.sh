#!/usr/bin/env bash
set -euo pipefail

repo="${CET_REPO:-/e/yky/FraudGT_cet_phaseb_diagnostics}"
output="${CET_OUTPUT:-/e/yky/FraudGT_cet_results/phaseb_diagnostics}"
oof_root="${CET_OOF_ROOT:-/e/yky/FraudGT_tier2b_results/crossfit_c78214c1}"
python="${CET_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python3.9}"
spec="$repo/run/cet_phaseb_diagnostic_ablation_spec.json"

mkdir -p "$output"
pids=()
for task in 0 1 2; do
  gpu="$((task + 4))"
  log="$output/task${task}_gpu${gpu}.stdout"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    "$python" "$repo/run/cet_phaseb_screen.py" \
    --spec "$spec" \
    --task-index "$task" \
    --output-dir "$output" \
    --oof-root "$oof_root" \
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
