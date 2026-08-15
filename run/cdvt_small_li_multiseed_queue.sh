#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_small_li_seed_stability}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
model_parent="${CDVT_MODEL_PARENT:-c76df053fb6287fa9c8b4771a77a11856d437fa7}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/small_li_multiseed_${commit}}"
seed42_root="${CDVT_SEED42_ROOT:-/e/yky/FraudGT_cdvt_results/multi_published_500e_c76df05}"
completion="${CDVT_COMPLETION:-$root/queue_complete.json}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
cpu_threads=8

read -r -a gpus <<< "${CDVT_GPUS:-1 4}"
seeds=(43 44)

if [[ "$branch" != "experiment/cdvt-small-li-seed-stability" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if ! git -C "$repo" diff --quiet "$model_parent" -- \
    fraudGT configs run/cdvt_phase1_screen.py run/cdvt_materialize_config.py; then
  echo "model or frozen protocol differs from seed-42 commit" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse result root: $root" >&2
  exit 2
fi
if [[ ! -f "$seed42_root/Small-LI_multi_cdvt_seed42/manifest.json" ]]; then
  echo "missing completed Small-LI seed-42 reference manifest" >&2
  exit 2
fi
if (( ${#gpus[@]} < ${#seeds[@]} )); then
  echo "two distinct GPUs are required for concurrent seeds" >&2
  exit 2
fi

mkdir -p "$root/configs"
"$python" - "$root/queue_manifest.json" "$commit" "$branch" \
  "$model_parent" "$seed42_root" <<'PY'
import json
import sys

path, commit, branch, model_parent, seed42_root = sys.argv[1:]
payload = {
    "branch": branch,
    "commit": commit,
    "model_code_parent": model_parent,
    "reference_seed42_root": seed42_root,
    "dataset": "Small-LI",
    "variant": "multi_cdvt",
    "seeds": [43, 44],
    "max_epochs": 500,
    "early_stopping_enabled": False,
    "sampling_protocol": "dynamic_random",
    "primary_metric": "Val-selected Test F1",
    "purpose": "descriptive seed-stability diagnostic",
    "architecture_frozen": True,
    "test_result_tuning_allowed": False,
}
with open(path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

base="$repo/configs/CDVT/phase1/AML-Small-LI.yaml"
for seed in "${seeds[@]}"; do
  config="$root/configs/AML-Small-LI-multi_cdvt-seed${seed}.yaml"
  "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$base" \
    --output "$config" \
    --seed "$seed" \
    --variant multi_cdvt
done

declare -A pid_seed=()
declare -A pid_gpu=()
declare -A busy_gpu=()
next_seed=0
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

start_seed() {
  local seed="$1"
  local gpu="$2"
  local name="Small-LI_multi_cdvt_seed${seed}"
  local output="$root/$name"
  local config="$root/configs/AML-Small-LI-multi_cdvt-seed${seed}.yaml"
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
  pid_seed["$pid"]="$seed"
  pid_gpu["$pid"]="$gpu"
  busy_gpu["$gpu"]="$pid"
  echo "$pid" > "$output/pid"
  printf 'Small-LI_multi_cdvt_seed%s\tstarted\tgpu=%s\tpid=%s\n' \
    "$seed" "$gpu" "$pid" >> "$root/queue_status.tsv"
}

while (( next_seed < ${#seeds[@]} || ${#pid_seed[@]} > 0 )); do
  for pid in "${!pid_seed[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    seed="${pid_seed[$pid]}"
    gpu="${pid_gpu[$pid]}"
    if wait "$pid"; then
      code=0
    else
      code="$?"
      queue_status=1
    fi
    printf 'Small-LI_multi_cdvt_seed%s\tfinished\tgpu=%s\tpid=%s\texit=%s\n' \
      "$seed" "$gpu" "$pid" "$code" >> "$root/queue_status.tsv"
    unset 'pid_seed[$pid]' 'pid_gpu[$pid]' 'busy_gpu[$gpu]'
  done

  for gpu in "${gpus[@]}"; do
    (( next_seed < ${#seeds[@]} )) || break
    [[ -z "${busy_gpu[$gpu]+set}" ]] || continue
    gpu_is_idle "$gpu" || continue
    start_seed "${seeds[$next_seed]}" "$gpu"
    next_seed=$((next_seed + 1))
  done

  if (( next_seed < ${#seeds[@]} || ${#pid_seed[@]} > 0 )); then
    sleep "$poll_seconds"
  fi
done

if (( queue_status == 0 )); then
  if ! "$python" "$repo/run/cdvt_small_li_multiseed_summary.py" \
      --seed42-root "$seed42_root" \
      --multiseed-root "$root" \
      --output-root "$root" \
      --write > "$root/multiseed_summary.stdout" 2>&1; then
    queue_status=1
  fi
fi

printf '{"complete":true,"status":%s}\n' "$queue_status" > "$completion"
exit "$queue_status"
