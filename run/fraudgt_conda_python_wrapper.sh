#!/usr/bin/env bash
set -euo pipefail

python_bin="$(find /d/conda_yky -path '*/envs/fraudgt_dual_gate/bin/python' -print -quit 2>/dev/null)"
[[ -x "$python_bin" ]] || { echo "fraudgt_dual_gate Python was not found" >&2; exit 127; }
exec "$python_bin" "$@"
