#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_multi_large_recovery}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
source_root="${CDVT_SOURCE_ROOT:-/e/yky/FraudGT_cdvt_results/post_followup_79e729b4}"
gpu="${CDVT_GPU:-3}"
chunk_size="${CDVT_EDGE_FF_CHUNK_SIZE:-65536}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/multi_large_oom_recovery_${commit}}"
cpu_threads=8

if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing to reuse recovery result root: $root" >&2
  exit 2
fi
for variant in multi_account_only multi_cdvt; do
  failed="$source_root/multi/Large-LI_${variant}_seed42"
  if [[ -f "$failed/manifest.json" ]]; then
    echo "source task unexpectedly has a final manifest: $failed" >&2
    exit 2
  fi
  if ! grep -q "torch.OutOfMemoryError" "$failed/stdout.log"; then
    echo "source task is not an audited OOM failure: $failed" >&2
    exit 2
  fi
done

read -r used utilization <<< "$(
  nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ','
)"
if (( used > 512 || utilization > 5 )); then
  echo "GPU $gpu is not idle: memory=$used MiB utilization=$utilization" >&2
  exit 2
fi

mkdir -p "$root/configs"
printf '{"commit":"%s","source_root":"%s",' "$commit" "$source_root" \
  > "$root/recovery_manifest.json"
printf '"dataset":"Large-LI","seed":42,"gpu":%s,' "$gpu" \
  >> "$root/recovery_manifest.json"
printf '"edge_ff_chunk_size":%s,"edge_ff_checkpoint":true,' "$chunk_size" \
  >> "$root/recovery_manifest.json"
printf '"batch_size":2048,"sampling_protocol":"dynamic_random"}\n' \
  >> "$root/recovery_manifest.json"

queue_status=0
for variant in multi_account_only multi_cdvt; do
  architecture="account_only"
  [[ "$variant" == "multi_cdvt" ]] && architecture="dual_view"
  config="$root/configs/AML-Large-LI-${variant}-seed42.yaml"
  output="$root/Large-LI_${variant}_seed42"
  "$python" "$repo/run/cdvt_materialize_config.py" \
    --base "$repo/configs/CDVT/phase1/AML-Large-LI.yaml" \
    --output "$config" \
    --seed 42 \
    --variant "$variant" \
    --edge-ff-chunk-size "$chunk_size" \
    --edge-ff-checkpoint
  mkdir -p "$output"
  printf '%s\tstarted\tgpu=%s\n' "$variant" "$gpu" \
    >> "$root/queue_status.tsv"
  if env CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONDONTWRITEBYTECODE=1 \
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
        --phase CDVT_multi_screen \
        > "$output/stdout.log" 2>&1; then
    code=0
  else
    code=$?
    queue_status=1
  fi
  printf '%s\tfinished\tgpu=%s\texit=%s\n' "$variant" "$gpu" "$code" \
    >> "$root/queue_status.tsv"
  (( code == 0 )) || break
done

printf '{"complete":true,"status":%s}\n' "$queue_status" \
  > "$root/queue_complete.json"
exit "$queue_status"
