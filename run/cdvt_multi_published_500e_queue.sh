#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_multi_published_500e}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/multi_published_500e_${commit}}"
completion="${CDVT_COMPLETION:-$root/queue_complete.json}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
large_li_chunk_size="${CDVT_LARGE_LI_EDGE_FF_CHUNK_SIZE:-65536}"
large_hi_chunk_size="${CDVT_LARGE_HI_EDGE_FF_CHUNK_SIZE:-32768}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-0 1 2 3 4 5 6}"
tasks=(
  'Large-LI|42'
  'Large-HI|42'
  'Medium-LI|42'
  'Medium-HI|42'
  'Small-LI|42'
  'Small-HI|42'
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
  echo "refusing to reuse formal result root: $root" >&2
  exit 2
fi
if (( large_li_chunk_size != 65536 || large_hi_chunk_size != 32768 )); then
  echo "unexpected formal Large edge FF chunk sizes" >&2
  exit 2
fi

base_config() {
  local dataset="$1"
  case "$dataset" in
    Small-LI|Large-LI)
      printf '%s/configs/CDVT/phase1/AML-%s.yaml\n' "$repo" "$dataset"
      ;;
    Small-HI|Medium-LI|Medium-HI|Large-HI)
      printf '%s/configs/CDVT/phase2/AML-%s.yaml\n' "$repo" "$dataset"
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      return 2
      ;;
  esac
}

mkdir -p "$root/configs"
"$python" - "$root/queue_manifest.json" "$commit" "$branch" <<'PY'
import json
import sys

path, commit, branch = sys.argv[1:]
payload = {
    "branch": branch,
    "commit": commit,
    "variant": "multi_cdvt",
    "datasets": [
        "Small-LI", "Small-HI", "Medium-LI",
        "Medium-HI", "Large-LI", "Large-HI",
    ],
    "seed": 42,
    "max_epochs": 500,
    "early_stopping_enabled": False,
    "sampling_protocol": "dynamic_random",
    "primary_baseline": "published_multi_fraudgt",
    "published_multi_fraudgt": {
        "Small-LI": 0.47010,
        "Small-HI": 0.76130,
        "Medium-LI": 0.44060,
        "Medium-HI": 0.75930,
        "Large-LI": 0.37430,
        "Large-HI": 0.73340,
    },
    "large_memory_execution": {
        "Large-LI": 65536,
        "Large-HI": 32768,
        "edge_ff_checkpoint": True,
        "preallocated_chunk_output": True,
        "chunkwise_ff_residual": True,
        "mathematical_definition_changed": False,
    },
}
with open(path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

for task in "${tasks[@]}"; do
  IFS='|' read -r dataset seed <<< "$task"
  config="$root/configs/AML-${dataset}-multi_cdvt-seed${seed}.yaml"
  materialize=(
    "$python" "$repo/run/cdvt_materialize_config.py"
    --base "$(base_config "$dataset")"
    --output "$config"
    --seed "$seed"
    --variant multi_cdvt
  )
  if [[ "$dataset" == "Large-LI" ]]; then
    materialize+=(
      --edge-ff-chunk-size "$large_li_chunk_size"
      --edge-ff-checkpoint
    )
  elif [[ "$dataset" == "Large-HI" ]]; then
    materialize+=(
      --edge-ff-chunk-size "$large_hi_chunk_size"
      --edge-ff-checkpoint
    )
  fi
  "${materialize[@]}"
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
  local dataset seed name output config
  IFS='|' read -r dataset seed <<< "$task"
  name="${dataset}_multi_cdvt_seed${seed}"
  output="$root/$name"
  config="$root/configs/AML-${dataset}-multi_cdvt-seed${seed}.yaml"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite task artifacts: $output" >&2
    return 2
  fi
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
    "$python" "$repo/run/cdvt_phase1_screen.py" \
      --config "$config" \
      --device cuda:0 \
      --variant dual_view \
      --experiment-label multi_cdvt \
      --lambda-cons 0.0 \
      --output-dir "$output" \
      --max-epochs 500 \
      --disable-early-stop \
      --phase CDVT_multi_published_screen \
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
    IFS='|' read -r dataset seed <<< "$task"
    printf '%s_multi_cdvt_seed%s\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$dataset" "$seed" "$gpu" "$pid" "$code" \
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

if ! "$python" "$repo/run/cdvt_multi_published_summary.py" \
    --result-root "$root" --write \
    > "$root/published_summary.stdout" 2>&1; then
  queue_status=1
fi

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$completion"
exit "$queue_status"
