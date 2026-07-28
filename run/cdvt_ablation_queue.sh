#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_phase3}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
phase2_root="${CDVT_PHASE2_ROOT:?CDVT_PHASE2_ROOT is required}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/ablation_${commit}}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(
  Medium-LI:account_only Medium-LI:event_only
  Small-LI:dual_view_no_relation
  Medium-LI:dual_view_no_relation
  Large-LI:dual_view_no_relation
  Small-LI:dual_view_k2 Medium-LI:dual_view_k2 Large-LI:dual_view_k2
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
  echo "refusing to reuse ablation result root: $root" >&2
  exit 2
fi

gate="$("$python" "$repo/run/cdvt_phase3_gate.py" \
  --phase1-root "$phase1_root" --phase2-root "$phase2_root")"
mkdir -p "$root/configs"
printf '%s\n' "$gate" > "$root/phase2_gate.json"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"tasks":8,"seed":42,"phase2_root":"%s"}\n' "$phase2_root" \
  >> "$root/queue_manifest.json"

base_config() {
  local dataset="$1"
  case "$dataset" in
    Small-LI|Large-LI)
      printf '%s/configs/CDVT/phase1/AML-%s.yaml\n' "$repo" "$dataset"
      ;;
    Medium-LI)
      printf '%s/configs/CDVT/phase2/AML-%s.yaml\n' "$repo" "$dataset"
      ;;
    *)
      echo "unknown representative dataset: $dataset" >&2
      return 2
      ;;
  esac
}

for task in "${tasks[@]}"; do
  dataset="${task%%:*}"
  variant="${task##*:}"
  "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$(base_config "$dataset")" \
    --output "$root/configs/AML-${dataset}-${variant}-seed42.yaml" \
    --seed 42 \
    --variant "$variant"
done

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

architecture_variant() {
  case "$1" in
    account_only|event_only) printf '%s\n' "$1" ;;
    dual_view_no_relation|dual_view_k2) printf 'dual_view\n' ;;
    *) echo "unknown ablation variant: $1" >&2; return 2 ;;
  esac
}

start_task() {
  local task="$1"
  local gpu="$2"
  local dataset="${task%%:*}"
  local variant="${task##*:}"
  local architecture
  architecture="$(architecture_variant "$variant")"
  local name="${dataset}_${variant}_seed42"
  local output="$root/$name"
  local config="$root/configs/AML-${dataset}-${variant}-seed42.yaml"
  if [[ -e "$output" ]]; then
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
      --variant "$architecture" \
      --experiment-label "$variant" \
      --lambda-cons 0.0 \
      --output-dir "$output" \
      --max-epochs 500 \
      --early-stop-min-epoch 80 \
      --early-stop-patience-evals 10 \
      --phase CDVT_ablation \
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

"$python" "$repo/run/cdvt_ablation_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --ablation-root "$root" \
  --allow-incomplete \
  --write \
  > "$root/ablation_summary.stdout" 2>&1 || queue_status=1

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
