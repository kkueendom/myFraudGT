#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


TEST_LINE = re.compile(r"'epoch': (\d+).+?'f1': ([0-9.eE+-]+)")
OUT_DIR_LINE = re.compile(r"^out_dir:\s*(.+?)\s*$")
DATASET_LINE = re.compile(r"^\s*name:\s*(.+?)\s*$")

THRESHOLDS = {
    "Small-HI": 0.7813,
    "Small-LI": 0.4901,
    "Medium-HI": 0.7793,
    "Medium-LI": 0.4606,
    "Large-HI": 0.7534,
    "Large-LI": 0.3943,
}

FAMILIES = {
    "supportmixconsisproto_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisProto240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisclassmixprotoboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisClassMixProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdualprotoboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoconsensusboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
    "supportmixconsisdeltafusiondualprotoconsensusdisagreeconfhardscaleboundresid_full240": [
        "configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Small-LI/AML-Small-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-HI/AML-Medium-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-HI/AML-Large-HI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
        "configs/AML-Large-LI/AML-Large-LI-SparseNodeGT+ports+Ego+WBDirMeanMaxWinnerProjTemporalPairChainContextSeqPairSeqBridgeBankWindowSeqSelectDeltaFusionRoleFlowBoundaryLagSupportMixConsisDualProtoConsensusDisagreeConfHardScaleBoundResid240UnifiedFullCalibMemSafe.yaml",
    ],
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def read_config_meta(cfg_path):
    out_dir = None
    dataset = None
    for line in Path(cfg_path).read_text(errors="ignore").splitlines():
        out_match = OUT_DIR_LINE.match(line)
        if out_match:
            out_dir = out_match.group(1).strip()
        data_match = DATASET_LINE.match(line)
        if data_match:
            dataset = data_match.group(1).strip()
    if not out_dir or not dataset:
        raise ValueError(f"failed to parse config metadata from {cfg_path}")
    return out_dir, dataset


def run_dir_for(cfg_path, gpu):
    out_dir, _ = read_config_meta(cfg_path)
    return Path(out_dir) / f"{Path(cfg_path).stem}-gpu{gpu}"


def parse_raw_peak(log_path):
    best_f1 = None
    best_epoch = None
    if not log_path.exists():
        return None, None
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        if "test:" not in raw_line:
            continue
        match = TEST_LINE.search(raw_line)
        if not match:
            continue
        epoch = int(match.group(1))
        f1 = float(match.group(2))
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
    return best_f1, best_epoch


def process_cmdline(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def wait_for_pid_release(pid_file, cmd_substr, poll_seconds):
    pid_path = Path(pid_file)
    last_msg = None
    while True:
        if not pid_path.exists():
            log(f"pid file {pid_file} missing; continuing")
            return
        raw_pid = pid_path.read_text().strip()
        if not raw_pid.isdigit():
            msg = f"pid file {pid_file} not ready: {raw_pid!r}"
            if msg != last_msg:
                log(msg)
                last_msg = msg
            time.sleep(poll_seconds)
            continue
        pid = int(raw_pid)
        cmdline = process_cmdline(pid)
        if not cmdline:
            log(f"wait pid {pid} exited")
            return
        if cmd_substr and cmd_substr not in cmdline:
            log(f"wait pid {pid} command changed; continuing")
            return
        msg = f"waiting on pid {pid}: {cmdline}"
        if msg != last_msg:
            log(msg)
            last_msg = msg
        time.sleep(poll_seconds)


def query_gpu_memory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    memory = {}
    for line in result.stdout.strip().splitlines():
        idx_str, used_str = [part.strip() for part in line.split(",")]
        memory[int(idx_str)] = int(used_str)
    return memory


def wait_for_gpu(gpu, max_used_mem_mib, poll_seconds):
    last_msg = None
    while True:
        used_mem = query_gpu_memory().get(gpu)
        if used_mem is None:
            raise RuntimeError(f"gpu {gpu} not found")
        if used_mem <= max_used_mem_mib:
            log(f"gpu {gpu} is free enough: {used_mem} MiB used")
            return
        msg = f"gpu {gpu} busy: {used_mem} MiB used, waiting"
        if msg != last_msg:
            log(msg)
            last_msg = msg
        time.sleep(poll_seconds)


def run_one(cfg_path, gpu, repo_root):
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true\n"
        "conda activate fraudgt_dual_gate\n"
        f"cd {repo_root}\n"
        f"python -m fraudGT.main --cfg {cfg_path} --repeat 1 --gpu {gpu}"
    )
    log(f"starting {cfg_path} on gpu {gpu}")
    return subprocess.run(["bash", "-lc", cmd], check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wait-pid-file")
    parser.add_argument("--wait-cmd-substr", default="")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-used-mem-mib", type=int, default=1000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.wait_pid_file:
        wait_for_pid_release(
            args.wait_pid_file, args.wait_cmd_substr, args.poll_seconds
        )

    results = []
    for cfg_path in FAMILIES[args.family]:
        wait_for_gpu(args.gpu, args.max_used_mem_mib, args.poll_seconds)
        run_dir = run_dir_for(cfg_path, args.gpu)
        status = run_one(cfg_path, args.gpu, repo_root)
        log_path = run_dir / "42" / "logging.log"
        best_f1, best_epoch = parse_raw_peak(log_path)
        _, dataset = read_config_meta(cfg_path)
        target = THRESHOLDS[dataset]
        results.append(
            {
                "dataset": dataset,
                "cfg": cfg_path,
                "status": status,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "target": target,
                "run_dir": str(run_dir),
            }
        )
        if best_f1 is None:
            log(
                f"finished {dataset} status={status} raw_peak=missing "
                f"target={target:.4f} run_dir={run_dir}"
            )
        else:
            log(
                f"finished {dataset} status={status} raw_peak={best_f1:.5f} "
                f"best_epoch={best_epoch} target={target:.4f} run_dir={run_dir}"
            )

    log("family summary:")
    all_pass = True
    for item in results:
        best_f1 = item["best_f1"]
        passed = best_f1 is not None and best_f1 >= item["target"]
        all_pass = all_pass and passed
        if best_f1 is None:
            best_text = "missing"
        else:
            best_text = f"{best_f1:.5f}@{item['best_epoch']}"
        log(
            f"{item['dataset']}: raw_peak={best_text} "
            f"target={item['target']:.4f} pass={passed}"
        )
    log(f"family {args.family} all_pass={all_pass}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
