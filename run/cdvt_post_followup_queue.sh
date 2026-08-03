#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_post_followup}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
phase2_root="${CDVT_PHASE2_ROOT:?CDVT_PHASE2_ROOT is required}"
followup_root="${CDVT_FOLLOWUP_ROOT:?CDVT_FOLLOWUP_ROOT is required}"
runtime_root="${CDVT_RUNTIME_ROOT:-$followup_root/runtime}"
followup_completion="${CDVT_FOLLOWUP_COMPLETION:-$followup_root/queue_complete.json}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/post_followup_${commit}}"
additive_root="$root/additive"
multi_root="$root/multi"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(
  'additive|Small-LI|42|causal_event_add'
  'additive|Medium-LI|42|causal_event_add'
  'additive|Large-LI|42|causal_event_add'
  'multi|Small-LI|42|multi_account_only'
  'multi|Small-LI|42|multi_cdvt'
  'multi|Medium-LI|42|multi_account_only'
  'multi|Medium-LI|42|multi_cdvt'
  'multi|Large-LI|42|multi_account_only'
  'multi|Large-LI|42|multi_cdvt'
)

if [[ "$branch" != "feature/cdvt-multi-backbone-screen" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse post-followup result root: $root" >&2
  exit 2
fi
if [[ ! -f "$followup_completion" ]]; then
  echo "follow-up completion record is missing: $followup_completion" >&2
  exit 2
fi
"$python" - "$followup_completion" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("complete") is not True or int(payload.get("status", 1)) != 0:
    raise SystemExit("follow-up queue did not complete successfully")
PY

# Strict summaries are the completion audit for all pre-existing final artifacts.
phase3_audit="$("$python" "$repo/run/cdvt_phase3_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --phase3-root "$followup_root/phase3" \
  --ablation-root "$followup_root/ablation")"
ablation_audit="$("$python" "$repo/run/cdvt_ablation_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --ablation-root "$followup_root/ablation")"
runtime_audit="$("$python" "$repo/run/cdvt_runtime_summary.py" \
  --runtime-root "$runtime_root")"

mkdir -p "$additive_root/configs" "$multi_root/configs"
printf '%s\n' "$phase3_audit" > "$root/followup_phase3_audit.json"
printf '%s\n' "$ablation_audit" > "$root/followup_ablation_audit.json"
printf '%s\n' "$runtime_audit" > "$root/followup_runtime_audit.json"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"training_tasks":9,"additive_tasks":3,"multi_tasks":6,' \
  >> "$root/queue_manifest.json"
printf '"seed":42,"single_gpu_allocator":true,"followup_root":"%s",' \
  "$followup_root" >> "$root/queue_manifest.json"
printf '"runtime_root":"%s","followup_completion":"%s"}\n' \
  "$runtime_root" "$followup_completion" >> "$root/queue_manifest.json"

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
    causal_event_add) printf 'additive_view\n' ;;
    multi_account_only) printf 'account_only\n' ;;
    multi_cdvt) printf 'dual_view\n' ;;
    *) echo "unknown post-followup variant: $1" >&2; return 2 ;;
  esac
}

for task in "${tasks[@]}"; do
  IFS='|' read -r family dataset seed variant <<< "$task"
  if [[ "$family" == "additive" ]]; then
    config="$additive_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
  else
    config="$multi_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
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
  local family dataset seed variant architecture output config runner_phase name
  IFS='|' read -r family dataset seed variant <<< "$task"
  architecture="$(architecture_variant "$variant")"
  name="${dataset}_${variant}_seed${seed}"
  if [[ "$family" == "additive" ]]; then
    output="$additive_root/$name"
    config="$additive_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
    runner_phase="CDVT_ablation"
  else
    output="$multi_root/$name"
    config="$multi_root/configs/AML-${dataset}-${variant}-seed${seed}.yaml"
    runner_phase="CDVT_multi_screen"
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
    IFS='|' read -r _family dataset seed variant <<< "$task"
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

"$python" "$repo/run/cdvt_additive_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --additive-root "$additive_root" \
  --write > "$additive_root/additive_summary.stdout" 2>&1 || queue_status=1

"$python" "$repo/run/cdvt_multi_summary.py" \
  --multi-root "$multi_root" \
  --write > "$multi_root/multi_summary.stdout" 2>&1 || queue_status=1

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
