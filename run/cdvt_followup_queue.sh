#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_followup}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
phase2_root="${CDVT_PHASE2_ROOT:?CDVT_PHASE2_ROOT is required}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/followup_${commit}}"
phase3_root="$root/phase3"
ablation_root="$root/ablation"
runtime_root="$root/runtime"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(
  'phase3|Small-LI|43|dual_view'
  'phase3|Small-LI|44|dual_view'
  'phase3|Medium-LI|43|dual_view'
  'phase3|Medium-LI|44|dual_view'
  'phase3|Large-LI|43|dual_view'
  'phase3|Large-LI|44|dual_view'
  'ablation|Medium-LI|42|account_only'
  'ablation|Medium-LI|42|event_only'
  'ablation|Small-LI|42|dual_view_no_relation'
  'ablation|Medium-LI|42|dual_view_no_relation'
  'ablation|Large-LI|42|dual_view_no_relation'
  'ablation|Small-LI|42|dual_view_k2'
  'ablation|Medium-LI|42|dual_view_k2'
  'ablation|Large-LI|42|dual_view_k2'
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
  echo "refusing to reuse follow-up result root: $root" >&2
  exit 2
fi

gate="$("$python" "$repo/run/cdvt_phase3_gate.py" \
  --phase1-root "$phase1_root" --phase2-root "$phase2_root")"
mkdir -p "$phase3_root/configs" "$ablation_root/configs"
printf '%s\n' "$gate" > "$root/phase2_gate.json"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"training_tasks":14,"runtime_tasks":6,"phase2_root":"%s"}\n' \
  "$phase2_root" >> "$root/queue_manifest.json"

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

architecture_variant() {
  case "$1" in
    account_only|event_only|dual_view) printf '%s\n' "$1" ;;
    dual_view_no_relation|dual_view_k2) printf 'dual_view\n' ;;
    *) echo "unknown follow-up variant: $1" >&2; return 2 ;;
  esac
}

for task in "${tasks[@]}"; do
  IFS='|' read -r phase dataset seed variant <<< "$task"
  if [[ "$phase" == "phase3" ]]; then
    config="$phase3_root/configs/AML-${dataset}-seed${seed}.yaml"
  else
    config="$ablation_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
  fi
  "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$(base_config "$dataset")" \
    --output "$config" \
    --seed "$seed" \
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

start_task() {
  local task="$1"
  local gpu="$2"
  local phase dataset seed variant architecture output config runner_phase name
  IFS='|' read -r phase dataset seed variant <<< "$task"
  architecture="$(architecture_variant "$variant")"
  name="${dataset}_${variant}_seed${seed}"
  if [[ "$phase" == "phase3" ]]; then
    output="$phase3_root/$name"
    config="$phase3_root/configs/AML-${dataset}-seed${seed}.yaml"
    runner_phase="CDVT_phase3"
  else
    output="$ablation_root/$name"
    config="$ablation_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
    runner_phase="CDVT_ablation"
  fi
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
      --phase "$runner_phase" \
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
    IFS='|' read -r _phase dataset seed variant <<< "$task"
    printf '%s_%s_seed%s\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$dataset" "$variant" "$seed" "$gpu" "$pid" "$code" \
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

"$python" "$repo/run/cdvt_phase3_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --phase3-root "$phase3_root" \
  --allow-incomplete \
  --write \
  > "$phase3_root/phase3_summary.stdout" 2>&1 || queue_status=1

"$python" "$repo/run/cdvt_ablation_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --ablation-root "$ablation_root" \
  --allow-incomplete \
  --write \
  > "$ablation_root/ablation_summary.stdout" 2>&1 || queue_status=1

if (( queue_status == 0 )); then
  env \
    CDVT_REPO="$repo" \
    CDVT_PYTHON="$python" \
    CDVT_PHASE1_ROOT="$phase1_root" \
    CDVT_PHASE2_ROOT="$phase2_root" \
    CDVT_ABLATION_ROOT="$ablation_root" \
    CDVT_OUTPUT_ROOT="$runtime_root" \
    CDVT_GPUS="${gpus[*]}" \
    CDVT_POLL_SECONDS="$poll_seconds" \
    CDVT_MAX_IDLE_MEMORY_MIB="$max_idle_memory_mib" \
    bash "$repo/run/cdvt_runtime_queue.sh" \
    > "$root/runtime_queue.stdout" 2>&1 || queue_status=1
fi

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
