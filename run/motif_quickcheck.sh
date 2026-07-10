#!/usr/bin/env bash
# Fast go/no-go for the Motif-Aware Graph Transformer.
# Runs motif-ON vs motif-OFF on the two smallest datasets at a short budget,
# then prints the val-selected test F1 delta. Everything is identical between
# the two arms except gt.motif_bias, so the delta is the mechanism's effect.
#
# Override via env, e.g.:
#   PY=/d/miniconda3/envs/fraudGT/bin/python EPOCHS=80 DATASETS="Small-HI Small-LI" \
#     bash run/motif_quickcheck.sh
set -e

PY="${PY:-python}"
REPO="${REPO:-$(pwd)}"
EPOCHS="${EPOCHS:-80}"
ITER="${ITER:-128}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-Small-HI Small-LI}"
MOTIF_OUT="${MOTIF_OUT:-$REPO/results/motif_quick/motif}"
BASE_OUT="${BASE_OUT:-$REPO/results/motif_quick/baseline}"

cd "$REPO"
echo "PY=$PY EPOCHS=$EPOCHS ITER=$ITER SEED=$SEED GPU=$GPU DATASETS=[$DATASETS]"

for ds in $DATASETS; do
  cfg="configs/AML-$ds/AML-$ds-SparseNodeGT+ports+Ego+Motif.yaml"
  echo "======== $ds : motif ON ========"
  "$PY" -m fraudGT.main --cfg "$cfg" --repeat 1 --gpu "$GPU" seed "$SEED" \
    optim.max_epoch "$EPOCHS" train.iter_per_epoch "$ITER" \
    train.tqdm False val.tqdm False wandb.use False \
    gt.motif_bias True out_dir "$MOTIF_OUT/AML-$ds"
  echo "======== $ds : motif OFF (matched baseline) ========"
  "$PY" -m fraudGT.main --cfg "$cfg" --repeat 1 --gpu "$GPU" seed "$SEED" \
    optim.max_epoch "$EPOCHS" train.iter_per_epoch "$ITER" \
    train.tqdm False val.tqdm False wandb.use False \
    gt.motif_bias False out_dir "$BASE_OUT/AML-$ds"
done

echo "======== comparison (val-selected test F1) ========"
"$PY" run/motif_quickcheck_audit.py --motif "$MOTIF_OUT" --baseline "$BASE_OUT"
