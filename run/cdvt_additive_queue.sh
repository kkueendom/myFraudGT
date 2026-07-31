#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_additive}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
phase1_root="${CDVT_PHASE1_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd}"
phase2_root="${CDVT_PHASE2_ROOT:?CDVT_PHASE2_ROOT is required}"
followup_root="${CDVT_FOLLOWUP_ROOT:?CDVT_FOLLOWUP_ROOT is required}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/additive_${commit}}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(Small-LI Medium-LI Large-LI)

if [[ "$branch" != "feature/cdvt-phase3-experiments" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse additive result root: $root" >&2
  exit 2
fi
if [[ ! -f "$followup_root/queue_complete.json" ]]; then
  echo "follow-up queue is not complete: $followup_root" >&2
  exit 2
fi
"$python" - "$followup_root/queue_complete.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("complete") is not True or int(payload.get("status", 1)) != 0:
    raise SystemExit("follow-up queue did not complete successfully")
PY

gate="$("$python" "$repo/run/cdvt_phase3_gate.py" \
  --phase1-root "$phase1_root" --phase2-root "$phase2_root")"
mkdir -p "$root/configs"
printf '%s\n' "$gate" > "$root/phase2_gate.json"
printf '{"commit":"%s","sampling_protocol":"dynamic_random",' "$commit" \
  > "$root/queue_manifest.json"
printf '"tasks":3,"seed":42,"control":"additive_view",' \
  >> "$root/queue_manifest.json"
printf '"followup_root":"%s"}\n' "$followup_root" \
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

for dataset in "${tasks[@]}"; do
  "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$(base_config "$dataset")" \
    --output "$root/configs/AML-${dataset}-causal_event_add-seed42.yaml" \
    --seed 42 \
    --variant causal_event_add
done

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
  local name="${dataset}_causal_event_add_seed42"
  local output="$root/$name"
  local config="$root/configs/AML-${dataset}-causal_event_add-seed42.yaml"
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
      --variant additive_view \
      --experiment-label causal_event_add \
      --lambda-cons 0.0 \
      --output-dir "$output" \
      --max-epochs 500 \
      --early-stop-min-epoch 80 \
      --early-stop-patience-evals 10 \
      --phase CDVT_ablation \
      > "$output/stdout.log" 2>&1 &
  local pid="$!"
  pid_dataset["$pid"]="$dataset"
  pid_gpu["$pid"]="$gpu"
  busy_gpu["$gpu"]="$pid"
  echo "$pid" > "$output/pid"
  printf '%s\tstarted\tgpu=%s\tpid=%s\n' \
    "$name" "$gpu" "$pid" >> "$root/queue_status.tsv"
}

while (( next_task < ${#tasks[@]} || ${#pid_dataset[@]} > 0 )); do
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
    printf '%s_causal_event_add_seed42\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$dataset" "$gpu" "$pid" "$code" >> "$root/queue_status.tsv"
    unset 'pid_dataset[$pid]' 'pid_gpu[$pid]' 'busy_gpu[$gpu]'
  done

  for gpu in "${gpus[@]}"; do
    (( next_task < ${#tasks[@]} )) || break
    [[ -z "${busy_gpu[$gpu]+set}" ]] || continue
    gpu_is_idle "$gpu" || continue
    start_task "${tasks[$next_task]}" "$gpu"
    next_task=$((next_task + 1))
  done

  if (( next_task < ${#tasks[@]} || ${#pid_dataset[@]} > 0 )); then
    sleep "$poll_seconds"
  fi
done

"$python" "$repo/run/cdvt_additive_summary.py" \
  --phase1-root "$phase1_root" \
  --phase2-root "$phase2_root" \
  --additive-root "$root" \
  --write \
  > "$root/additive_summary.stdout" 2>&1 || queue_status=1

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
