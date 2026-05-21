#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${SANDBOX_ROOT}/.env.staging}"
PID_FILE="${SANDBOX_ROOT}/runtime/staging-preview.pid"

HOST="127.0.0.1"
PORT="5013"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  HOST="${SOUNDSKETCHER_HOST:-${HOST}}"
  PORT="${SOUNDSKETCHER_PORT:-${PORT}}"
fi

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    echo "Staging preview pid ${PID} is running."
  else
    echo "Staging pid file exists, but process is not running."
  fi
else
  echo "No staging pid file found."
fi

if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" || true
fi

echo "Configured URL: http://${HOST}:${PORT}"
