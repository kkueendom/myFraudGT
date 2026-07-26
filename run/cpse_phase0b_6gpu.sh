#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="${CPSE_REPO:?CPSE_REPO is required}"
OUTPUT="${CPSE_OUTPUT:?CPSE_OUTPUT is required}"
OOF_ROOT="${CPSE_OOF_ROOT:?CPSE_OOF_ROOT is required}"
SPEC="${CPSE_SPEC:-${REPO}/run/cpse_phase0b_spec.json}"
PYTHON_BIN="${CPSE_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"

mkdir -p "${OUTPUT}/logs"
pids=()
for task in 0 1 2 3 4 5; do
  gpu="${task}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${REPO}/run/cpse_phase0b_oof.py" \
    --spec "${SPEC}" \
    --task-index "${task}" \
    --output-dir "${OUTPUT}" \
    --oof-root "${OOF_ROOT}" \
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

