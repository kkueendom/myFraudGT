#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining}"
CONDA_ENV="${CONDA_ENV:-fraudgt_dual_gate}"
GPU0_CHAIN_PID="${GPU0_CHAIN_PID:?GPU0_CHAIN_PID is required}"
GPU1_WAIT_PID="${GPU1_WAIT_PID:?GPU1_WAIT_PID is required}"
GPU1_QUEUE_LOG="${GPU1_QUEUE_LOG:?GPU1_QUEUE_LOG is required}"
RESCUE_LOG="${RESCUE_LOG:-$REPO/logs/unified_large_rescue_$(date +%Y%m%d_%H%M%S).log}"

CFG_LARGE_HI='configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedCalibMemSafe.yaml'
CFG_LARGE_LI='configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedCalibMemSafe.yaml'

ts() {
  date '+%F %T'
}

note() {
  echo "[$(ts)] $*" | tee -a "$RESCUE_LOG"
}

queue_note() {
  echo "[$(ts)] $*" | tee -a "$GPU1_QUEUE_LOG" >>"$RESCUE_LOG"
}

has_large_started() {
  grep -Eq 'start unified_largehi|finish unified_largehi' "$GPU1_QUEUE_LOG" 2>/dev/null
}

wait_for_gpu0_chain() {
  while kill -0 "$GPU0_CHAIN_PID" 2>/dev/null; do
    note "waiting gpu0 main chain pid=$GPU0_CHAIN_PID"
    sleep 120
  done
  note "gpu0 main chain finished"
}

run_cfg() {
  local cfg="$1"
  local tag="$2"
  local log="$REPO/logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
  queue_note "start $tag cfg=$cfg gpu=0 rescue"
  bash -lc "source ~/.bashrc >/dev/null 2>&1 || true; conda activate $CONDA_ENV && cd $REPO && python -m fraudGT.main --cfg $cfg --repeat 1 --gpu 0" >"$log" 2>&1
  local best
  best=$(grep "test:" "$log" | sed -nE "s/.*'f1': ([0-9.]+).*/\\1/p" | sort -nr | head -n1 || true)
  queue_note "finish $tag best=${best:-NA} log=$log"
}

wait_for_gpu0_chain

if has_large_started; then
  note "gpu1 queue already started unified_largehi; no rescue needed"
  exit 0
fi

if kill -0 "$GPU1_WAIT_PID" 2>/dev/null; then
  note "stopping stale gpu1 wait chain pid=$GPU1_WAIT_PID before rescue"
  kill "$GPU1_WAIT_PID"
  sleep 2
fi

queue_note "gpu1 wait chain preempted; large datasets reassigned to gpu0 rescue"
run_cfg "$CFG_LARGE_HI" "unified_largehi"
run_cfg "$CFG_LARGE_LI" "unified_largeli"
note "gpu0 rescue for large datasets completed"
