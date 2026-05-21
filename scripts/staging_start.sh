#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${SANDBOX_ROOT}/.env.staging}"
RUNTIME_DIR="${SANDBOX_ROOT}/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_FILE="${RUNTIME_DIR}/staging-preview.pid"
LOG_FILE="${LOG_DIR}/staging-preview.log"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Staging env file not found: ${ENV_FILE}" >&2
  echo "Create one from .env.staging.example before starting staging." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "Staging preview is already running with pid ${EXISTING_PID}."
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${SANDBOX_ROOT}"
nohup scripts/run_staging_preview.sh "${ENV_FILE}" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"
echo "Started staging preview with pid ${PID}."
echo "Logs: ${LOG_FILE}"
