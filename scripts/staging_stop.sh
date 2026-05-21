#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${SANDBOX_ROOT}/runtime/staging-preview.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No staging pid file found."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if [[ -z "${PID}" ]]; then
  rm -f "${PID_FILE}"
  echo "Removed empty staging pid file."
  exit 0
fi

if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}"
  sleep 1
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Staging process ${PID} did not stop after SIGTERM." >&2
    exit 1
  fi
  echo "Stopped staging preview pid ${PID}."
else
  echo "Staging pid ${PID} is not running."
fi

rm -f "${PID_FILE}"
