#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${SANDBOX_ROOT}/runtime/logs/staging-preview.log"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "No staging log file found: ${LOG_FILE}" >&2
  exit 1
fi

tail -n "${1:-80}" "${LOG_FILE}"
