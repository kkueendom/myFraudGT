#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_phase3}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
phase2_root="${CDVT_PHASE2_ROOT:?CDVT_PHASE2_ROOT is required}"
ablation_root="${CDVT_ABLATION_ROOT:?CDVT_ABLATION_ROOT is required}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/runtime_${commit}}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(
  Small-LI:account_only Small-LI:dual_view
  Medium-LI:account_only Medium-LI:dual_view
  Large-LI:account_only Large-LI:dual_view
)

if [[ "$branch" != "feature/cdvt-phase3-experiments" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse runtime result root: $root" >&2
  exit 2
fi

gate="$("$python" "$repo/run/cdvt_phase3_gate.py" \
  --phase1-root "$phase1_root" --phase2-root "$phase2_root")"
"$python" "$repo/run/cdvt_ablation_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --ablation-root "$ablation_root" \
  > /dev/null
mkdir -p "$root"
printf '%s\n' "$gate" > "$root/phase2_gate.json"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"mode":"normal_only","tasks":6,"seed":42,' \
  >> "$root/queue_manifest.json"
printf '"phase2_root":"%s","ablation_root":"%s"}\n' \
  "$phase2_root" "$ablation_root" >> "$root/queue_manifest.json"

source_manifest() {
  local dataset="$1"
  local variant="$2"
  local source_root
  if [[ "$variant" == "dual_view" ]]; then
    if [[ "$dataset" == "Medium-LI" ]]; then
      source_root="$phase2_root"
    else
      source_root="$phase1_root"
    fi
  elif [[ "$dataset" == "Medium-LI" ]]; then
    source_root="$ablation_root"
  else
    source_root="$phase1_root"
  fi
  printf '%s/%s_%s_seed42/manifest.json\n' \
    "$source_root" "$dataset" "$variant"
}

declare -A pid_task=()
declare -A pid_gpu=()
declare -A busy_gpu=()
next_task=0
queue_status=0

gpu_is_idle() {
  local gpu="$1"
  local line used utilization
  line="$(nvidia-smi -i "$gpu" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null)" || return 1
  IFS=',' read -r used utilization <<< "$line"
  used="${used//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
  (( used <= max_idle_memory_mib && utilization <= 5 ))
}

start_task() {
  local task="$1"
  local gpu="$2"
  local dataset="${task%%:*}"
  local variant="${task##*:}"
  local name="${dataset}_${variant}_seed42"
  local output="$root/$name"
  local manifest
  manifest="$(source_manifest "$dataset" "$variant")"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite benchmark artifacts: $output" >&2
    return 2
  fi
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
    "$python" "$repo/run/cdvt_runtime_benchmark.py" \
      --manifest "$manifest" \
      --device cuda:0 \
      --output-dir "$output" \
      --warmup-batches 4 \
      --max-batches 256 \
      > "$output/stdout.log" 2>&1 &
  local pid="$!"
  pid_task["$pid"]="$task"
  pid_gpu["$pid"]="$gpu"
  busy_gpu["$gpu"]="$pid"
  echo "$pid" > "$output/pid"
  printf '%s\tstarted\tgpu=%s\tpid=%s\n' \
    "$name" "$gpu" "$pid" >> "$root/queue_status.tsv"
}

while (( next_task < ${#tasks[@]} || ${#pid_task[@]} > 0 )); do
  for pid in "${!pid_task[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    task="${pid_task[$pid]}"
    gpu="${pid_gpu[$pid]}"
    if wait "$pid"; then
      code=0
    else
      code="$?"
      queue_status=1
    fi
    dataset="${task%%:*}"
    variant="${task##*:}"
    printf '%s_%s_seed42\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$dataset" "$variant" "$gpu" "$pid" "$code" \
      >> "$root/queue_status.tsv"
    unset 'pid_task[$pid]' 'pid_gpu[$pid]' 'busy_gpu[$gpu]'
  done

  for gpu in "${gpus[@]}"; do
    (( next_task < ${#tasks[@]} )) || break
    [[ -z "${busy_gpu[$gpu]+set}" ]] || continue
    gpu_is_idle "$gpu" || continue
    start_task "${tasks[$next_task]}" "$gpu"
    next_task=$((next_task + 1))
  done

  if (( next_task < ${#tasks[@]} || ${#pid_task[@]} > 0 )); then
    sleep "$poll_seconds"
  fi
done

"$python" "$repo/run/cdvt_runtime_summary.py" \
  --runtime-root "$root" \
  --allow-incomplete \
  --write \
  > "$root/runtime_summary.stdout" 2>&1 || queue_status=1

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
