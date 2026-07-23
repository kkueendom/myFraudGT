#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$1"
python_bin="${PYTHON_BIN:-python3}"
read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5 6}"
mkdir -p "$output_dir"

run_worker() {
  local gpu="$1"
  shift
  (
    cd "$repo_root"
    for job in "$@"; do
      local spec="${job%%:*}"
      local task_index="${job##*:}"
      PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        run/tier_phase1_evidence_qualification.py \
        --spec "run/$spec" \
        --task-index "$task_index" \
        --output-dir "$output_dir" \
        --device cuda:0
    done
  ) >"$output_dir/gpu${gpu}.stdout" 2>&1 &
  pids+=("$!")
}

if [[ "${#gpu_ids[@]}" -eq 7 ]]; then
  job_groups=(
    "tier_phase1b_structure_spec.json:0 tier_phase1b_temporal_spec.json:2"
    "tier_phase1b_structure_spec.json:1 tier_phase1b_temporal_spec.json:3"
    "tier_phase1b_structure_spec.json:2 tier_phase1b_flow_role_spec.json:2"
    "tier_phase1b_structure_spec.json:3 tier_phase1b_flow_role_spec.json:3"
    "tier_phase1b_temporal_spec.json:0 tier_phase1b_flow_role_spec.json:1"
    "tier_phase1b_temporal_spec.json:1"
    "tier_phase1b_flow_role_spec.json:0"
  )
elif [[ "${#gpu_ids[@]}" -eq 6 ]]; then
  job_groups=(
    "tier_phase1b_structure_spec.json:0 tier_phase1b_temporal_spec.json:2"
    "tier_phase1b_structure_spec.json:1 tier_phase1b_temporal_spec.json:3"
    "tier_phase1b_structure_spec.json:2 tier_phase1b_flow_role_spec.json:2"
    "tier_phase1b_structure_spec.json:3 tier_phase1b_flow_role_spec.json:3"
    "tier_phase1b_temporal_spec.json:0 tier_phase1b_flow_role_spec.json:1"
    "tier_phase1b_temporal_spec.json:1 tier_phase1b_flow_role_spec.json:0"
  )
else
  echo "GPU_IDS must contain exactly 6 or 7 GPU ids" >&2
  exit 2
fi

pids=()
for index in "${!gpu_ids[@]}"; do
  read -r -a jobs <<< "${job_groups[$index]}"
  run_worker "${gpu_ids[$index]}" "${jobs[@]}"
done

printf '%s\n' "${pids[@]}" >"$output_dir/worker_pids.txt"
wait "${pids[@]}"
