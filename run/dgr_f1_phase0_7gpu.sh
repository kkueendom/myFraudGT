#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="${DGR_REPO:?DGR_REPO is required}"
OUTPUT="${DGR_OUTPUT:?DGR_OUTPUT is required}"
COMMIT="${DGR_COMMIT:?DGR_COMMIT is required}"
SPEC="${DGR_SPEC:-${REPO}/run/dgr_f1_phase0_spec.json}"
PYTHON_BIN="${DGR_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"

mkdir -p "${OUTPUT}/logs"
pids=()
for task in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES="${task}" "${PYTHON_BIN}" \
    "${REPO}/run/dgr_f1_phase0.py" \
    --spec "${SPEC}" \
    --task-index "${task}" \
    --output-root "${OUTPUT}" \
    --expected-commit "${COMMIT}" \
    >"${OUTPUT}/logs/task${task}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
