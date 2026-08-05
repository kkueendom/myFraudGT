#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_multi_published_500e}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
source_root="${CDVT_SOURCE_ROOT:-/e/yky/FraudGT_cdvt_results/multi_published_500e_c76df05}"
gpu="${CDVT_GPU:-0}"
chunk_size="${CDVT_EDGE_FF_CHUNK_SIZE:-32768}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/multi_published_500e_large_hi_recovery_${commit}}"
cpu_threads=8

if [[ "$branch" != "feature/cdvt-multi-backbone-screen" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse recovery result root: $root" >&2
  exit 2
fi
if (( chunk_size != 32768 )); then
  echo "formal Large-HI recovery requires edge_ff_chunk_size=32768" >&2
  exit 2
fi

failed="$source_root/Large-HI_multi_cdvt_seed42"
if [[ -f "$failed/manifest.json" ]]; then
  echo "source Large-HI unexpectedly has a final manifest" >&2
  exit 2
fi
if ! grep -q "torch.OutOfMemoryError" "$failed/stdout.log"; then
  echo "source Large-HI is not an audited OOM failure" >&2
  exit 2
fi

read -r used utilization <<< "$(
  nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ','
)"
if (( used > 512 || utilization > 5 )); then
  echo "GPU $gpu is not idle: memory=$used MiB utilization=$utilization" >&2
  exit 2
fi

mkdir -p "$root/configs"
config="$root/configs/AML-Large-HI-multi_cdvt-seed42.yaml"
output="$root/Large-HI_multi_cdvt_seed42"
"$python" "$repo/run/cdvt_materialize_config.py" \
  --base "$repo/configs/CDVT/phase2/AML-Large-HI.yaml" \
  --output "$config" \
  --seed 42 \
  --variant multi_cdvt \
  --edge-ff-chunk-size "$chunk_size" \
  --edge-ff-checkpoint

"$python" - "$root/recovery_manifest.json" "$commit" "$source_root" \
  "$gpu" "$chunk_size" <<'PY'
import json
import sys

path, commit, source_root, gpu, chunk_size = sys.argv[1:]
payload = {
    "commit": commit,
    "source_failure_root": source_root,
    "dataset": "Large-HI",
    "variant": "multi_cdvt",
    "seed": 42,
    "gpu": int(gpu),
    "max_epochs": 500,
    "early_stopping_enabled": False,
    "sampling_protocol": "dynamic_random",
    "edge_ff_chunk_size": int(chunk_size),
    "edge_ff_checkpoint": True,
    "preallocated_chunk_output": True,
    "chunkwise_ff_residual": True,
    "mathematical_definition_changed": False,
    "restart_epoch": 0,
}
with open(path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

mkdir -p "$output"
printf 'Large-HI_multi_cdvt_seed42\tstarted\tgpu=%s\n' "$gpu" \
  > "$root/queue_status.tsv"
status=0
if env CUDA_VISIBLE_DEVICES="$gpu" \
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
      > "$output/stdout.log" 2>&1; then
  code=0
else
  code=$?
  status=1
fi
printf 'Large-HI_multi_cdvt_seed42\tfinished\tgpu=%s\texit=%s\n' \
  "$gpu" "$code" >> "$root/queue_status.tsv"
printf '{"complete":true,"status":%s}\n' "$status" \
  > "$root/queue_complete.json"
exit "$status"
