#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$1"
python_bin="${PYTHON_BIN:-python3}"
read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5}"
mkdir -p "$output_dir"

if [[ "${#gpu_ids[@]}" -ne 6 ]]; then
  echo "GPU_IDS must contain exactly 6 GPU ids" >&2
  exit 2
fi

pids=()
for task_index in 0 1 2 3 4 5; do
  gpu="${gpu_ids[$task_index]}"
  (
    cd "$repo_root"
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
      run/tier_phase2b_crossfit_teachers.py \
      --spec run/tier_phase2b_crossfit_spec.json \
      --task-index "$task_index" \
      --output-dir "$output_dir" \
      --device cuda:0
  ) >"$output_dir/gpu${gpu}.stdout" 2>&1 &
  pids+=("$!")
done

printf '%s\n' "${pids[@]}" >"$output_dir/worker_pids.txt"
wait "${pids[@]}"
