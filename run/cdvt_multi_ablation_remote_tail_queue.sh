#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yyk/FraudGT_cdvt_ablation_run_691c0c7}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
root="${CDVT_OUTPUT_ROOT:-/e/yyk/FraudGT_cdvt_results/multi_ablation_seed42_691c0c7_remote_tail}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8
read -r -a gpus <<< "${CDVT_GPUS:-0 1}"

# These are exactly the five tasks not present in the first wave on host 101.
tasks=(
  "Medium-LI|multi_causal_event_add"
  "Large-LI|multi_causal_event_add"
  "Small-LI|multi_dual_view_no_relation"
  "Medium-LI|multi_dual_view_no_relation"
  "Large-LI|multi_dual_view_no_relation"
)

commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
[[ "$branch" == "experiment/multi-cdvt-ablation" ]] || { echo "unexpected branch" >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo "dirty worktree" >&2; exit 2; }
[[ ! -e "$root/queue_status.tsv" ]] || { echo "queue already exists: $root" >&2; exit 2; }

architecture() {
  case "$1" in
    multi_causal_event_add) echo additive_view ;;
    multi_dual_view_no_relation) echo dual_view ;;
    *) return 2 ;;
  esac
}

edge_options() {
  case "$1" in
    Large-LI) echo "65536 1" ;;
    *) echo "0 0" ;;
  esac
}

gpu_idle() {
  local line used util
  line="$(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r used util <<< "$line"
  used="$(printf '%s' "$used" | tr -d '[:space:]')"
  util="$(printf '%s' "$util" | tr -d '[:space:]')"
  [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && (( used <= max_idle_memory_mib && util <= 5 ))
}

mkdir -p "$root/configs"
printf '%s\n' '{"phase":"CDVT_multi_ablation_seed42_remote_tail","commit":"'"$commit"'","tasks":5,"seed":42,"max_epochs":500,"early_stopping_enabled":false,"sampling_protocol":"dynamic_random"}' > "$root/queue_manifest.json"
printf 'queue_started\tcommit=%s\n' "$commit" > "$root/queue_status.tsv"

for task in "${tasks[@]}"; do
  dataset="${task%%|*}"
  label="${task##*|}"
  config="$root/configs/AML-$dataset-$label-seed42.yaml"
  [[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
done

declare -A ptask=() pgpu=() busy=()
next=0
queue_status=1
finish() { printf '{"complete":true,"status":%s}\n' "$queue_status" > "$root/queue_complete.json"; }
trap finish EXIT

start_task() {
  local task="$1" gpu="$2"
  local dataset="${task%%|*}" label="${task##*|}"
  local out="$root/${dataset}_${label}_seed42"
  local config="$root/configs/AML-${dataset}-${label}-seed42.yaml"
  local chunk checkpoint
  read -r chunk checkpoint <<< "$(edge_options "$dataset")"
  local args=(--config "$config" --device cuda:0 --variant "$(architecture "$label")" --experiment-label "$label" --lambda-cons 0.0 --output-dir "$out" --max-epochs 500 --disable-early-stop --phase CDVT_multi_ablation)
  (( chunk > 0 )) && args+=(--edge-ff-chunk-size "$chunk")
  (( checkpoint == 1 )) && args+=(--edge-ff-checkpoint)
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" "$python" "$repo/run/cdvt_phase1_screen.py" "${args[@]}" > "$out/stdout.log" 2>&1 &
  local pid="$!"
  ptask[$pid]="$task"
  pgpu[$pid]="$gpu"
  busy[$gpu]=$pid
  printf '%s\tstarted\tgpu=%s\tpid=%s\n' "$task" "$gpu" "$pid" >> "$root/queue_status.tsv"
}

while (( next < 5 || ${#ptask[@]} > 0 )); do
  for pid in "${!ptask[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    task="${ptask[$pid]}"
    gpu="${pgpu[$pid]}"
    if wait "$pid"; then code=0; else code="$?"; queue_status=1; fi
    printf '%s\tfinished\tgpu=%s\tpid=%s\texit=%s\n' "$task" "$gpu" "$pid" "$code" >> "$root/queue_status.tsv"
    unset 'ptask[$pid]' 'pgpu[$pid]' 'busy[$gpu]'
  done
  for gpu in "${gpus[@]}"; do
    (( next < 5 )) || break
    [[ -z "${busy[$gpu]+x}" ]] || continue
    gpu_idle "$gpu" || continue
    start_task "${tasks[$next]}" "$gpu"
    next=$((next + 1))
  done
  (( next < 5 || ${#ptask[@]} > 0 )) && sleep "$poll_seconds"
done
queue_status=0
