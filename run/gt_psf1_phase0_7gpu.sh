#!/usr/bin/env bash
set -euo pipefail

REPO="${GT_PSF1_REPO:?set GT_PSF1_REPO}"
OUTPUT="${GT_PSF1_OUTPUT:?set GT_PSF1_OUTPUT}"
PYTHON="${GT_PSF1_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${OUTPUT}/logs"

pids=()
for task in 0 1 2 3 4 5 6; do
  task_output="${OUTPUT}/task_${task}"
  log="${OUTPUT}/logs/task_${task}.log"
  CUDA_VISIBLE_DEVICES="${task}" "${PYTHON}" \
    "${REPO}/run/gt_psf1_phase0.py" \
    --task-id "${task}" \
    --spec "${REPO}/run/gt_psf1_phase0_spec.json" \
    --output "${task_output}" \
    --expected-commit "${COMMIT}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
