#!/bin/bash
# ==============================================================================
# Script Eksekusi Manual di Compute Node (JupyterHub / Interactive PBS Job)
# Didesain khusus untuk dieksekusi langsung di dalam node (misal cx3-19-5)
# ==============================================================================

set -euo pipefail

LOCATION="${1:-2A400}"
DATE="${2:-2026-04-22}"
MODELS="${3:-all}"

echo "=============================================================================="
echo "🚀 RUNNING SPATIAL ECOACOUSTIC PIPELINE (IN-NODE DIRECT)"
echo "   Node:     $(hostname)"
echo "   Location: ${LOCATION}"
echo "   Date:     ${DATE}"
echo "   Models:   ${MODELS}"
echo "=============================================================================="

# 1. Environment & Canonical Ephemeral Paths
export MONITORING_DATA="/rds/general/user/ri322/ephemeral/monitoring_data"
export ANALYSIS_OUTPUT="/rds/general/user/ri322/ephemeral/sea-work"

export TMP_DIR="/tmp/${USER:-ri322}"
mkdir -p "${TMP_DIR}/numba" "${TMP_DIR}/torch_ext" "${TMP_DIR}/hf_cache"
export NUMBA_CACHE_DIR="${TMP_DIR}/numba"
export TORCH_EXTENSIONS_DIR="${TMP_DIR}/torch_ext"
export HF_HOME="${TMP_DIR}/hf_cache"

# Explicit CUDA 12.6 NVVM / ptxas for TensorFlow XLA
export CUDA_HOME="/rds/easybuild/noarch/apps/software/CUDA/12.6.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export CUDA_MODULE_LOADING=LAZY
export BACPIPE_DEVICE=auto
export ORT_DISABLE_THREAD_AFFINITY=1
export OMP_NUM_THREADS=4
export ORT_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 2. HPC Modules & Shared Libraries (NVIDIA CUDA 12 & cuDNN 9 untuk ONNX Runtime GPU)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load tools/prod Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true

SCRIPT_DIR="/rds/general/user/ri322/home/spatial-ecoacoustic-analysis"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/bacpipe/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

NV_LIBS=$(find "${VENV_DIR}/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr "\n" ":")
TRITON_BIN=$(find "${VENV_DIR}/lib/python3.11/site-packages/triton" -maxdepth 4 -type d -name "bin" 2>/dev/null | head -1)

export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"
if [ -n "${TRITON_BIN}" ]; then
    export PATH="${TRITON_BIN}:${PATH}"
fi

echo ""
echo "=== 1. VERIFIKASI GPU & LINGKUNGAN ==="
"${VENV_PYTHON}" -c 'import torch, onnxruntime; print("  ✓ Torch CUDA:", torch.cuda.is_available(), f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""); print("  ✓ ONNX Providers:", onnxruntime.get_available_providers())'

echo ""
echo "=== 2. EKSEKUSI PIPELINE (RESUME DARI CHECKPOINT) ==="
"${VENV_PYTHON}" bacpipe/pipeline_bacpipe.py \
    --location "${LOCATION}" \
    --date "${DATE}" \
    --models "${MODELS}" \
    --methods mono,sa,bf_LabIR,bf_SPIR \
    --device auto

echo ""
echo "=============================================================================="
echo "✅ PIPELINE IN-NODE COMPLETED SUCCESSFULLY: ${LOCATION} / ${DATE}"
echo "=============================================================================="
