#!/usr/bin/env bash
set -euo pipefail

REPO="${BUDGET_REPO:?set BUDGET_REPO}"
OUTPUT="${BUDGET_OUTPUT:?set BUDGET_OUTPUT}"
PYTHON="${BUDGET_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
SPEC="${REPO}/run/dynamic_budget_convergence_spec.json"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${OUTPUT}/logs"

worker() {
  local gpu="$1"
  local task
  for ((task=gpu; task<36; task+=7)); do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
      "${REPO}/run/dynamic_budget_convergence.py" \
      --spec "${SPEC}" \
      --task-index "${task}" \
      --output-dir "${OUTPUT}" \
      --device cuda:0 \
      >"${OUTPUT}/logs/task_${task}.log" 2>&1
  done
}

pids=()
for gpu in 0 1 2 3 4 5 6; do
  worker "${gpu}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"

