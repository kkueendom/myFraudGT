#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_multi_runtime}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
dataset="${CDVT_DATASET:-Small-LI}"
if [[ -n "${CDVT_SOURCE_CONFIG:-}" ]]; then
  source_config="$CDVT_SOURCE_CONFIG"
elif [[ "$dataset" == "Large-HI" ]]; then
  source_config="/e/yky/FraudGT_cdvt_results/multi_published_500e_large_hi_recovery_decabdb/configs/AML-Large-HI-multi_cdvt-seed42.yaml"
else
  source_config="/e/yky/FraudGT_cdvt_results/multi_published_500e_c76df05/configs/AML-${dataset}-multi_cdvt-seed42.yaml"
fi
gpu="${CDVT_GPU:-0}"
warmup_batches="${CDVT_WARMUP_BATCHES:-4}"
batches_per_repeat="${CDVT_BATCHES_PER_REPEAT:-32}"
repeats="${CDVT_REPEATS:-3}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/multi_runtime_quick_${dataset}_${commit}}"

case "$dataset" in
  Small-LI|Small-HI|Medium-LI|Medium-HI|Large-LI|Large-HI) ;;
  *)
    echo "unsupported runtime dataset: $dataset" >&2
    exit 2
    ;;
esac
if [[ "$branch" != "experiment/multi-cdvt-runtime" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ ! -f "$source_config" ]]; then
  echo "missing source config: $source_config" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse runtime root: $root" >&2
  exit 2
fi

gpu_state="$(nvidia-smi -i "$gpu" \
  --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits)"
IFS=',' read -r used utilization <<< "$gpu_state"
used="${used//[[:space:]]/}"
utilization="${utilization//[[:space:]]/}"
if (( used > 512 || utilization > 5 )); then
  echo "GPU $gpu is not idle: memory=${used}MiB util=${utilization}%" >&2
  exit 2
fi

mkdir -p "$root/configs"
for variant in multi_account_only multi_cdvt; do
  config="$root/configs/AML-${dataset}-${variant}-seed42.yaml"
  env PYTHONDONTWRITEBYTECODE=1 \
    "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$source_config" \
    --output "$config" \
    --seed 42 \
    --variant "$variant"
done

env PYTHONDONTWRITEBYTECODE=1 \
  "$python" - "$root/queue_manifest.json" "$commit" "$gpu" \
  "$warmup_batches" "$batches_per_repeat" "$repeats" "$dataset" <<'PY'
import json
import sys

path, commit, gpu, warmup, batches, repeats, dataset = sys.argv[1:]
payload = {
    "commit": commit,
    "dataset": dataset,
    "variants": ["multi_account_only", "multi_cdvt"],
    "seed": 42,
    "gpu": int(gpu),
    "warmup_batches": int(warmup),
    "batches_per_repeat": int(batches),
    "repeats": int(repeats),
    "sampling_protocol": "dynamic_random",
    "benchmark_mode": "normal_only_end_to_end_inference",
}
with open(path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

for variant in multi_account_only multi_cdvt; do
  output="$root/${dataset}_${variant}_seed42"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "$python" "$repo/run/cdvt_multi_runtime_benchmark.py" \
      --config "$root/configs/AML-${dataset}-${variant}-seed42.yaml" \
      --device cuda:0 \
      --variant "$variant" \
      --output-dir "$output" \
      --warmup-batches "$warmup_batches" \
      --batches-per-repeat "$batches_per_repeat" \
      --repeats "$repeats" \
      > "$output/stdout.log" 2>&1
done

env PYTHONDONTWRITEBYTECODE=1 \
  "$python" "$repo/run/cdvt_multi_runtime_summary.py" \
  --runtime-root "$root" \
  --datasets "$dataset" \
  --write \
  > "$root/runtime_summary.stdout"
printf '{"complete":true,"status":0}\n' > "$root/queue_complete.json"
