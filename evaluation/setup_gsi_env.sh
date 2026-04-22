#!/bin/bash
set -euo pipefail

# Work around conda hooks that read this variable directly under nounset.
export CONDA_MKL_INTERFACE_LAYER_BACKUP="${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}"

usage() {
  cat <<'EOF'
Usage: ./setup_gsi_env.sh [--recreate]

Creates and configures the gsi conda env, installs deps, and builds GroundingDINO.
If --recreate is set, the existing env will be removed first.
EOF
}

RECREATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please initialize conda first."
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="gsi"
PY_VERSION="3.10"
CUDA_VER="11.7"
GROUNDINGDINO_REF="856dde20aee659246248e20734ef9ba5214f5e44"
GROUNDINGDINO_LOCAL_DIR="$REPO_ROOT/src/groundingdino"
GROUNDINGDINO_VCS_URL="git+https://github.com/IDEA-Research/GroundingDINO.git@${GROUNDINGDINO_REF}#egg=groundingdino"

CONDA_BASE="$(conda info --base)"
# conda's activate/deactivate scripts can reference unset vars; disable nounset around them.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$RECREATE" -eq 1 ]]; then
    set +u
    export CONDA_MKL_INTERFACE_LAYER_BACKUP="${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
      conda deactivate || true
    fi
    set -u
    set +u
    export CONDA_MKL_INTERFACE_LAYER_BACKUP="${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}"
    conda env remove -n "$ENV_NAME" -y
    set -u
  else
    echo "Env $ENV_NAME already exists. Re-run with --recreate to remove it."
    exit 1
  fi
fi

set +u
conda create -n "$ENV_NAME" python="$PY_VERSION" pip -y
conda activate "$ENV_NAME"
set -u

set +u
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda="$CUDA_VER" -y
conda install -c nvidia cuda-version="$CUDA_VER" cuda-toolkit="$CUDA_VER" cuda-nvcc="$CUDA_VER" -y
set -u

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_TARGETS_INCLUDE="$CUDA_HOME/targets/x86_64-linux/include"

if [[ -f "$CUDA_HOME/include/thrust/complex.h" ]]; then
  export CPATH="$CUDA_HOME/include:$CUDA_TARGETS_INCLUDE${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$CUDA_TARGETS_INCLUDE${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export C_INCLUDE_PATH="$CUDA_HOME/include:$CUDA_TARGETS_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
elif [[ -f "$CUDA_TARGETS_INCLUDE/thrust/complex.h" ]]; then
  export CPATH="$CUDA_TARGETS_INCLUDE:$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$CUDA_TARGETS_INCLUDE:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export C_INCLUDE_PATH="$CUDA_TARGETS_INCLUDE:$CUDA_HOME/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
else
  echo "Missing thrust headers at expected CUDA 11.7 paths."
  if [[ -f "$CUDA_TARGETS_INCLUDE/cccl/thrust/complex.h" ]]; then
    echo "Found only CCCL thrust headers at:"
    echo "  $CUDA_TARGETS_INCLUDE/cccl/thrust/complex.h"
    echo "This usually means mixed CUDA packages (e.g. cuda-cccl 13.x with nvcc 11.7)."
    echo "Please recreate env and keep CUDA packages pinned to 11.7 (including cuda-version=11.7)."
  fi
  exit 1
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
nvcc --version
find "$CONDA_PREFIX" -name cuda_runtime_api.h | head
find "$CONDA_PREFIX" -path "*/thrust/complex.h" | head

pip install -U pip setuptools wheel ninja
pip install -r "$REPO_ROOT/requirements-cu117.txt"

if [[ -d "$GROUNDINGDINO_LOCAL_DIR" ]]; then
  echo "Installing GroundingDINO from local path: $GROUNDINGDINO_LOCAL_DIR"
  pip install -e "$GROUNDINGDINO_LOCAL_DIR" --no-build-isolation
else
  echo "Local GroundingDINO not found, installing pinned commit: $GROUNDINGDINO_REF"
  pip install -e "$GROUNDINGDINO_VCS_URL" --no-build-isolation
fi

python - <<'PY'
from groundingdino.models.GroundingDINO import ms_deform_attn
print("ms_deform_attn _C:", ms_deform_attn._C is not None)
PY

echo "Done. Activate with: conda activate $ENV_NAME"
