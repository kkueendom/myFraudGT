#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_phase2}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/phase2_${commit}}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
datasets=(Medium-LI Small-HI Medium-HI Large-HI)

if [[ "$branch" != "feature/cdvt-causal-dual-view" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root/queue_manifest.json" ]]; then
  echo "queue already exists: $root" >&2
  exit 2
fi
if [[ ! -d "$phase1_root" ]]; then
  echo "missing Phase 1 result root: $phase1_root" >&2
  exit 2
fi

mkdir -p "$root"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"variant":"dual_view","lambda_cons":0.0,"tasks":4}\n' \
  >> "$root/queue_manifest.json"

declare -A pid_dataset=()
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
  local dataset="$1"
  local gpu="$2"
  local name="${dataset}_dual_view_seed42"
  local output="$root/$name"
  local config="$repo/configs/CDVT/phase2/AML-${dataset}.yaml"
  if [[ -e "$output/manifest.json" || -e "$output/progress.json" ]]; then
    echo "refusing to overwrite task artifacts: $output" >&2
    return 2
  fi
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
    "$python" "$repo/run/cdvt_phase1_screen.py" \
      --config "$config" \
      --device cuda:0 \
      --variant dual_view \
      --experiment-label dual_view \
      --lambda-cons 0.0 \
      --output-dir "$output" \
      --max-epochs 500 \
      --early-stop-min-epoch 80 \
      --early-stop-patience-evals 10 \
      --phase CDVT_phase2 \
      > "$output/stdout.log" 2>&1 &
  local pid="$!"
  pid_dataset["$pid"]="$dataset"
  pid_gpu["$pid"]="$gpu"
  busy_gpu["$gpu"]="$pid"
  echo "$pid" > "$output/pid"
  printf '%s\tstarted\tgpu=%s\tpid=%s\n' \
    "$name" "$gpu" "$pid" >> "$root/queue_status.tsv"
}

while (( next_task < ${#datasets[@]} || ${#pid_dataset[@]} > 0 )); do
  for pid in "${!pid_dataset[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    dataset="${pid_dataset[$pid]}"
    gpu="${pid_gpu[$pid]}"
    if wait "$pid"; then
      code=0
    else
      code="$?"
      queue_status=1
    fi
    printf '%s_dual_view_seed42\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$dataset" "$gpu" "$pid" "$code" >> "$root/queue_status.tsv"
    unset 'pid_dataset[$pid]' 'pid_gpu[$pid]' 'busy_gpu[$gpu]'
  done

  for gpu in "${gpus[@]}"; do
    (( next_task < ${#datasets[@]} )) || break
    [[ -z "${busy_gpu[$gpu]+set}" ]] || continue
    gpu_is_idle "$gpu" || continue
    start_task "${datasets[$next_task]}" "$gpu"
    next_task=$((next_task + 1))
  done

  if (( next_task < ${#datasets[@]} || ${#pid_dataset[@]} > 0 )); then
    sleep "$poll_seconds"
  fi
done

"$python" "$repo/run/cdvt_phase2_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$root" \
  --allow-incomplete \
  --write \
  > "$root/phase2_summary.stdout" 2>&1 || queue_status=1

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
