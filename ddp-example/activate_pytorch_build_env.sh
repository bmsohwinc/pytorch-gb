#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${1:-${REPO_DIR}/p3}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH:-}"

echo "Activated: ${VENV_DIR}"
echo "Repo: ${REPO_DIR}"
echo "Next:"
echo "  cd ${REPO_DIR}"
echo "  python -m pip install --no-build-isolation -v -e ."
