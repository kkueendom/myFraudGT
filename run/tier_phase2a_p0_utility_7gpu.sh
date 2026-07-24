#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PHASE1B_ROOT OUTPUT_DIR" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
phase1b_root="$1"
output_dir="$2"
python_bin="${PYTHON_BIN:-python3}"
read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5 6}"
mkdir -p "$output_dir"

if [[ "${#gpu_ids[@]}" -lt 1 || "${#gpu_ids[@]}" -gt 7 ]]; then
  echo "GPU_IDS must contain between 1 and 7 GPU ids" >&2
  exit 2
fi

run_worker() {
  local gpu="$1"
  shift
  (
    cd "$repo_root"
    for task_index in "$@"; do
      PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        run/tier_phase2a_p0_utility_probe.py \
        --spec run/tier_phase2a_p0_utility_spec.json \
        --task-index "$task_index" \
        --phase1b-root "$phase1b_root" \
        --output-dir "$output_dir" \
        --device cuda:0
    done
  ) >"$output_dir/gpu${gpu}.stdout" 2>&1 &
  pids+=("$!")
}

job_groups=()
for index in "${!gpu_ids[@]}"; do
  jobs=()
  task_index="$index"
  while [[ "$task_index" -lt 12 ]]; do
    jobs+=("$task_index")
    task_index=$((task_index + ${#gpu_ids[@]}))
  done
  job_groups+=("${jobs[*]}")
done

pids=()
for index in "${!gpu_ids[@]}"; do
  read -r -a jobs <<< "${job_groups[$index]}"
  run_worker "${gpu_ids[$index]}" "${jobs[@]}"
done

printf '%s\n' "${pids[@]}" >"$output_dir/worker_pids.txt"
wait "${pids[@]}"
