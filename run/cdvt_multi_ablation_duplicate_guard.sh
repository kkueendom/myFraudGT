#!/usr/bin/env bash
set -euo pipefail

status_file="${1:?queue status file required}"
queue_pid="${2:?queue pid required}"
poll_seconds="${CDVT_POLL_SECONDS:-60}"
reserved=(
  "Medium-LI|multi_causal_event_add"
  "Large-LI|multi_causal_event_add"
  "Small-LI|multi_dual_view_no_relation"
  "Medium-LI|multi_dual_view_no_relation"
  "Large-LI|multi_dual_view_no_relation"
)

while kill -0 "$queue_pid" 2>/dev/null; do
  declare -A last_status=() last_pid=()
  while IFS=$'\t' read -r task status gpu pid_field; do
    [[ -n "$task" ]] || continue
    last_status["$task"]="$status"
    last_pid["$task"]="${pid_field#pid=}"
  done < "$status_file"
  for task in "${reserved[@]}"; do
    [[ "${last_status[$task]:-}" == started ]] || continue
    pid="${last_pid[$task]:-}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      printf '%s\n' "duplicate_guard\tterminated\t$task\tpid=$pid" >> "$status_file"
    fi
  done
  sleep "$poll_seconds"
done
