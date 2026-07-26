#!/usr/bin/env bash
set -euo pipefail

repo="${CDVT_REPO:-/e/yky/FraudGT_cdvt_phase0_40628d8}"
python="${CDVT_PYTHON:-/d/miniconda3/envs/fraudGT/bin/python}"
commit="$(git -C "$repo" rev-parse --short=8 HEAD)"
branch="$(git -C "$repo" branch --show-current)"
root="${CDVT_OUTPUT_ROOT:-/e/yky/FraudGT_cdvt_results/phase1_formal_${commit}}"
cpu_threads=8

if [[ "$branch" != "feature/cdvt-causal-dual-view" ]]; then
  echo "unexpected branch: $branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 2
fi
if [[ -e "$root/queue_manifest.json" ]]; then
  echo "queue already exists: $root" >&2
  exit 2
fi

mkdir -p "$root"
printf '{"commit":"%s","sampling_protocol":"dynamic_random","tasks":8}\n' \
  "$commit" > "$root/queue_manifest.json"

pids=()
names=()

launch() {
  local gpu="$1"
  local dataset="$2"
  local variant="$3"
  local label="$4"
  local lambda_cons="$5"
  local config="configs/CDVT/phase1/AML-${dataset}.yaml"
  local name="${dataset}_${label}_seed42"
  local output="$root/$name"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
    "$python" "$repo/run/cdvt_phase1_screen.py" \
      --config "$repo/$config" \
      --device cuda:0 \
      --variant "$variant" \
      --experiment-label "$label" \
      --lambda-cons "$lambda_cons" \
      --output-dir "$output" \
      --max-epochs 500 \
      --early-stop-min-epoch 80 \
      --early-stop-patience-evals 10 \
      > "$output/stdout.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "$!" > "$output/pid"
  printf '%s gpu=%s pid=%s\n' "$name" "$gpu" "$!"
}

launch 0 Small-LI event_only event_only 0.0
launch 1 Small-LI dual_view dual_view 0.0
launch 2 Small-LI dual_view full_cdvt 0.1
launch 3 Large-LI event_only event_only 0.0
launch 4 Large-LI dual_view dual_view 0.0
launch 5 Large-LI dual_view full_cdvt 0.1

run_account_task() {
  local dataset="$1"
  local config="configs/CDVT/phase1/AML-${dataset}.yaml"
  local name="${dataset}_account_only_seed42"
  local output="$root/$name"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
    "$python" "$repo/run/cdvt_phase1_screen.py" \
      --config "$repo/$config" \
      --device cuda:0 \
      --variant account_only \
      --experiment-label account_only \
      --lambda-cons 0.0 \
      --output-dir "$output" \
      --max-epochs 500 \
      --early-stop-min-epoch 80 \
      --early-stop-patience-evals 10 \
      > "$output/stdout.log" 2>&1
}

run_account_queue() {
  local account_status=0
  if ! run_account_task Small-LI; then
    account_status=1
  fi
  if ! run_account_task Large-LI; then
    account_status=1
  fi
  return "$account_status"
}

run_account_queue &
pids+=("$!")
names+=("account_only_queue")
echo "$!" > "$root/account_only_queue.pid"

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s exit=0\n' "${names[$index]}" >> "$root/queue_status.tsv"
  else
    code="$?"
    printf '%s exit=%s\n' "${names[$index]}" "$code" \
      >> "$root/queue_status.tsv"
    status=1
  fi
done

printf '{"complete":true,"status":%s}\n' "$status" \
  > "$root/queue_complete.json"
exit "$status"
