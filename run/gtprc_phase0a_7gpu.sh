#!/usr/bin/env bash
set -euo pipefail

REPO="${GTPRC_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${GTPRC_OUTPUT:?GTPRC_OUTPUT is required}"
SPEC="${GTPRC_SPEC:-${REPO}/run/gtprc_phase0a_spec.json}"
COMMIT="$(git -C "${REPO}" rev-parse HEAD)"

mkdir -p "${OUTPUT}/logs"

readarray -t TASKS < <(
  python3 - "${SPEC}" <<'PY'
import json
import sys

spec = json.load(open(sys.argv[1]))
for task in spec["tasks"]:
    print("{}\t{}\t{}".format(
        task["gpu"], task["regime"], task["seed"]))
PY
)

pids=()
for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r gpu regime seed <<<"${task}"
  task_dir="${OUTPUT}/${regime}_seed${seed}_${COMMIT:0:8}"
  log="${OUTPUT}/logs/${regime}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 \
    "${REPO}/run/gtprc_phase0a_simulation.py" \
    --regime "${regime}" \
    --seed "${seed}" \
    --output-dir "${task_dir}" \
    --expected-commit "${COMMIT}" \
    --replicates "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['replicates'])")" \
    --batch-size "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['batch_size'])")" \
    --calibration-rows "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['calibration_rows'])")" \
    --test-rows "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['test_rows'])")" \
    --policy-count "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['policy_count'])")" \
    --alpha "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['alpha'])")" \
    --delta "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['delta'])")" \
    --min-coverage "$(python3 -c \
      "import json; print(json.load(open('${SPEC}'))['min_coverage'])")" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

python3 "${REPO}/run/summarize_gtprc_phase0a.py" \
  --input-root "${OUTPUT}" \
  --output-json "${OUTPUT}/gtprc_phase0a_aggregate.json"

