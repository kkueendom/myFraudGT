import argparse
import os
import subprocess
import time
from pathlib import Path


DEFAULT_QUEUE = {
    1: [
        "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplit240.yaml",
        "configs/AML-Large-LI/AML-Large-LI-UnifiedClassSplit240.yaml",
    ],
    0: [
        "configs/AML-Large-HI/AML-Large-HI-UnifiedClassSplit240.yaml",
        "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplit240.yaml",
    ],
}

RUN_TAGS = {
    "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplit240.yaml": (1, "smallhi"),
    "configs/AML-Large-LI/AML-Large-LI-UnifiedClassSplit240.yaml": (1, "largeli"),
    "configs/AML-Large-HI/AML-Large-HI-UnifiedClassSplit240.yaml": (0, "largehi"),
    "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplit240.yaml": (0, "mediumli"),
}

ACTIVE_PID_FILES = {
    1: ".last_classsplit_smallli_gpu1_pid",
    0: ".last_classsplit_mediumhi_gpu0_pid",
}


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path):
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def launch(repo, env_name, cfg, gpu):
    _, tag = RUN_TAGS[cfg]
    log_path = repo / f".last_classsplit_{tag}_gpu{gpu}_log"
    pid_path = repo / f".last_classsplit_{tag}_gpu{gpu}_pid"
    cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"conda activate {env_name}; "
        f"cd {repo}; "
        f"python -m fraudGT.main --cfg {cfg} --repeat 1 --gpu {gpu}"
    )
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    pid_path.write_text(f"{proc.pid}\n")
    print(time.strftime("%F %T"), "launch", cfg, "gpu", gpu, "pid", proc.pid, flush=True)
    return proc.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--env", default="fraudgt_dual_gate")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    queue = {gpu: list(cfgs) for gpu, cfgs in DEFAULT_QUEUE.items()}
    active = {
        gpu: read_pid(repo / pid_file)
        for gpu, pid_file in ACTIVE_PID_FILES.items()
    }
    print(time.strftime("%F %T"), "queue-supervisor-start", active, flush=True)

    while True:
        pending = False
        for gpu in sorted(queue):
            if alive(active.get(gpu)):
                pending = True
                continue
            if queue[gpu]:
                cfg = queue[gpu].pop(0)
                active[gpu] = launch(repo, args.env, cfg, gpu)
                pending = True
            else:
                active[gpu] = None
        if not pending:
            print(time.strftime("%F %T"), "queue-supervisor-finish", flush=True)
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
