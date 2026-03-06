#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${DEFAULT_REPO_DIR}"
VENV_DIR="${DEFAULT_REPO_DIR}/p3"
CLONE_IF_MISSING=0
CLONE_PARENT="$(dirname -- "${DEFAULT_REPO_DIR}")"
INSTALL_SYSTEM_PIP=0
INSTALL_TRITON=0
SUBMODULE_JOBS="8"

usage() {
  cat <<'EOF'
Usage: bash ddp-example/setup_until_pytorch_install.sh [options]

Options:
  --repo-dir PATH         PyTorch source directory
  --venv-dir PATH         Virtual environment directory
  --clone-if-missing      Clone PyTorch if repo-dir does not exist
  --clone-parent PATH     Parent directory used with --clone-if-missing
  --submodule-jobs N      Parallel jobs for submodule fetch (default: 8)
  --install-system-pip    Run: sudo apt install -y python3-pip
  --install-triton        Run: make triton
  -h, --help              Show this help
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

venv_python_path() {
  if [[ -x "$1/bin/python3" ]]; then
    echo "$1/bin/python3"
    return
  fi
  if [[ -x "$1/bin/python" ]]; then
    echo "$1/bin/python"
    return
  fi
  echo ""
}

is_pytorch_repo() {
  [[ -f "$1/requirements.txt" && -d "$1/torch" ]]
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  require_cmd curl
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  export PATH="${HOME}/.local/bin:${PATH}"

  require_cmd uv
}

clone_repo_if_needed() {
  if is_pytorch_repo "${REPO_DIR}"; then
    return
  fi

  if [[ "${CLONE_IF_MISSING}" != "1" ]]; then
    echo "PyTorch source not found at: ${REPO_DIR}" >&2
    echo "Use --repo-dir PATH or rerun with --clone-if-missing" >&2
    exit 1
  fi

  mkdir -p "${CLONE_PARENT}"
  git clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/pytorch/pytorch "${REPO_DIR}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --clone-if-missing)
      CLONE_IF_MISSING=1
      shift
      ;;
    --clone-parent)
      CLONE_PARENT="$2"
      shift 2
      ;;
    --submodule-jobs)
      SUBMODULE_JOBS="$2"
      shift 2
      ;;
    --install-system-pip)
      INSTALL_SYSTEM_PIP=1
      shift
      ;;
    --install-triton)
      INSTALL_TRITON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Warning: guide expects Ubuntu 22.04, found ${PRETTY_NAME:-unknown}" >&2
  fi
fi

require_cmd git
require_cmd python3

ensure_uv
clone_repo_if_needed

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv "${VENV_DIR}"
fi

if [[ -z "$(venv_python_path "${VENV_DIR}")" ]]; then
  echo "Virtualenv exists but is broken (missing python): ${VENV_DIR}" >&2
  echo "Recreating virtualenv at ${VENV_DIR}" >&2
  rm -rf "${VENV_DIR}"
  uv venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

VENV_PYTHON="$(venv_python_path "${VENV_DIR}")"
if [[ -z "${VENV_PYTHON}" ]]; then
  echo "Failed to locate python in virtualenv: ${VENV_DIR}" >&2
  exit 1
fi

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

if [[ "${INSTALL_SYSTEM_PIP}" == "1" ]]; then
  require_cmd sudo
  sudo apt install -y python3-pip
fi

cd "${REPO_DIR}"

git submodule sync
git -c submodule.fetchJobs="${SUBMODULE_JOBS}" submodule update --init --recursive --depth=1 --jobs "${SUBMODULE_JOBS}" --recommend-shallow

uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install cmake ninja mkl-static mkl-include

if [[ "${INSTALL_TRITON}" == "1" ]]; then
  make triton
fi

export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH:-}"

cat <<EOF

Setup complete.

Skipped:
  - driver install from Songyu's install_all.sh

Stopped before the PyTorch build/install step.

Current environment:
  repo: ${REPO_DIR}
  venv: ${VENV_DIR}
  CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}

Next:
  source "${SCRIPT_DIR}/activate_pytorch_build_env.sh"
  cd "${REPO_DIR}"
  python -m pip install --no-build-isolation -v -e .
EOF
