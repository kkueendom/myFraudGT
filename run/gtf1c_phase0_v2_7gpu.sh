#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="${GTF1C_REPO:?GTF1C_REPO is required}"
OUTPUT="${GTF1C_OUTPUT:?GTF1C_OUTPUT is required}"
INPUT="${GTF1C_INPUT:?GTF1C_INPUT is required}"
COMMIT="${GTF1C_COMMIT:?GTF1C_COMMIT is required}"
SPEC="${GTF1C_SPEC:-${REPO}/run/gtf1c_phase0_v2_dev_spec.json}"
PYTHON_BIN="${GTF1C_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"

mkdir -p "${OUTPUT}/logs"
pids=()
for task in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES="${task}" "${PYTHON_BIN}" \
    "${REPO}/run/gtf1c_phase0_v2.py" \
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
