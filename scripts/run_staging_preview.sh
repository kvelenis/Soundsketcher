#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${SANDBOX_ROOT}/.env.staging.example}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Staging env file not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ "$(cd "$(dirname "${ENV_FILE}")" && pwd)/$(basename "${ENV_FILE}")" == "${SANDBOX_ROOT}/.env.staging.example" ]]; then
  echo "Warning: using .env.staging.example unchanged. Copy it to .env.staging and edit paths before a real preview host." >&2
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

HOST="${SOUNDSKETCHER_HOST:-0.0.0.0}"
PORT="${SOUNDSKETCHER_PORT:-5012}"

cd "${SANDBOX_ROOT}"
exec scripts/run_python.sh -m uvicorn app.main:app --host "${HOST}" --port "${PORT}"
