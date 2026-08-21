#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_multi_runtime_formal}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
gpu="${CDVT_GPU:-2}"
poll_seconds="${CDVT_POLL_SECONDS:-300}"
max_idle_memory_mib="${CDVT_MAX_IDLE_MEMORY_MIB:-512}"
wait_pattern="${CDVT_WAIT_PROCESS_PATTERN:-/run/cdvt_phase1_screen.py}"
warmup_batches=4
batches_per_repeat=32
repeats=8
datasets=(
  Small-LI Small-HI Medium-LI Medium-HI Large-LI Large-HI
)
variants=(multi_account_only multi_cdvt)

commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/multi_runtime_formal_256_${commit}}"

if [[ "$branch" != "experiment/multi-cdvt-runtime" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse runtime root: $root" >&2
  exit 2
fi

queue_status=1
finish() {
  printf '{"complete":true,"status":%s}\n' "$queue_status" \
    > "$root/queue_complete.json"
}
trap finish EXIT

source_config() {
  local dataset="$1"
  if [[ "$dataset" == "Large-HI" ]]; then
    printf '%s\n' \
      "/e/yky/FraudGT_cdvt_results/multi_published_500e_large_hi_recovery_decabdb/configs/AML-Large-HI-multi_cdvt-seed42.yaml"
  else
    printf '%s\n' \
      "/e/yky/FraudGT_cdvt_results/multi_published_500e_c76df05/configs/AML-${dataset}-multi_cdvt-seed42.yaml"
  fi
}

gpu_is_idle() {
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

mkdir -p "$root/configs"
for dataset in "${datasets[@]}"; do
  base="$(source_config "$dataset")"
  if [[ ! -f "$base" ]]; then
    echo "missing source config: $base" >&2
    exit 2
  fi
done

env PYTHONDONTWRITEBYTECODE=1 \
  "$python" - "$root/queue_manifest.json" "$commit" "$gpu" \
  "$wait_pattern" <<'PY'
import json
import sys

path, commit, gpu, wait_pattern = sys.argv[1:]
payload = {
    "phase": "CDVT_multi_runtime_formal_256",
    "evidence_tier": "formal_256",
    "commit": commit,
    "datasets": [
        "Small-LI", "Small-HI", "Medium-LI",
        "Medium-HI", "Large-LI", "Large-HI",
    ],
    "variants": ["multi_account_only", "multi_cdvt"],
    "seed": 42,
    "gpu": int(gpu),
    "warmup_batches": 4,
    "batches_per_repeat": 32,
    "repeats": 8,
    "total_measured_batches_per_model_dataset": 256,
    "sampling_protocol": "dynamic_random",
    "benchmark_mode": "normal_only_end_to_end_inference",
    "execution": "single_gpu_serial",
    "wait_process_pattern": wait_pattern,
}
with open(path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'waiting\tprocess_pattern=%s\tgpu=%s\n' \
  "$wait_pattern" "$gpu" > "$root/queue_status.tsv"
while pgrep -f "$wait_pattern" > /dev/null || ! gpu_is_idle; do
  sleep "$poll_seconds"
done
printf 'measurement_window_ready\tgpu=%s\n' "$gpu" \
  >> "$root/queue_status.tsv"

for dataset in "${datasets[@]}"; do
  base="$(source_config "$dataset")"
  for variant in "${variants[@]}"; do
    config="$root/configs/AML-${dataset}-${variant}-seed42.yaml"
    env PYTHONDONTWRITEBYTECODE=1 \
      "$python" "$repo/run/cdvt_materialize_config.py" \
      --base "$base" \
      --output "$config" \
      --seed 42 \
      --variant "$variant"

    output="$root/${dataset}_${variant}_seed42"
    mkdir -p "$output"
    printf '%s\t%s\tstarted\n' "$dataset" "$variant" \
      >> "$root/queue_status.tsv"
    env CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONDONTWRITEBYTECODE=1 \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
      "$python" "$repo/run/cdvt_multi_runtime_benchmark.py" \
        --config "$config" \
        --device cuda:0 \
        --variant "$variant" \
        --output-dir "$output" \
        --warmup-batches "$warmup_batches" \
        --batches-per-repeat "$batches_per_repeat" \
        --repeats "$repeats" \
        --evidence-tier formal_256 \
        > "$output/stdout.log" 2>&1
    printf '%s\t%s\tfinished\n' "$dataset" "$variant" \
      >> "$root/queue_status.tsv"
  done
done

env PYTHONDONTWRITEBYTECODE=1 \
  "$python" "$repo/run/cdvt_multi_runtime_summary.py" \
  --runtime-root "$root" \
  --datasets "${datasets[@]}" \
  --evidence-tier formal_256 \
  --write \
  > "$root/runtime_summary.stdout"
queue_status=0
