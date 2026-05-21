#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${SOUNDSKETCHER_PYTHON:-}" && -f "${SANDBOX_ROOT}/.env.staging" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SANDBOX_ROOT}/.env.staging"
  set +a
fi

if [[ -n "${SOUNDSKETCHER_PYTHON:-}" ]]; then
  exec "${SOUNDSKETCHER_PYTHON}" "$@"
fi

if [[ -x "${SANDBOX_ROOT}/.venv/bin/python" ]]; then
  exec "${SANDBOX_ROOT}/.venv/bin/python" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$@"
fi

exec python "$@"
