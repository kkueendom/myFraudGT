#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="${TREFIC_REPO:?TREFIC_REPO is required}"
OUTPUT="${TREFIC_OUTPUT:?TREFIC_OUTPUT is required}"
INPUT="${TREFIC_INPUT:?TREFIC_INPUT is required}"
COMMIT="${TREFIC_COMMIT:?TREFIC_COMMIT is required}"
SPEC="${TREFIC_SPEC:-${REPO}/run/trefic_phase0_dev_spec.json}"
PYTHON_BIN="${TREFIC_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"

mkdir -p "${OUTPUT}/logs"
pids=()
for task in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES="${task}" "${PYTHON_BIN}" \
    "${REPO}/run/trefic_phase0.py" \
    --spec "${SPEC}" \
    --task-index "${task}" \
    --output-root "${OUTPUT}" \
    --input-root "${INPUT}" \
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
