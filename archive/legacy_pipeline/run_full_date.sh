#!/bin/bash
# ==============================================================================
# End-to-End Spatial Ecoacoustic Analysis Pipeline (Single Command Execution)
# Processes raw FLACs -> Beamforming/Mono/SA -> Multi-Model GPU Embeddings -> Visuals
# ==============================================================================

set -e

LOCATION="${1:-2A400}"
DATE="${2:-2026-04-21}"
MODELS="${3:-all}"

echo "=============================================================================="
echo "🚀 STARTING COMPLETE SPATIAL ECOACOUSTIC PIPELINE"
echo "   Location: ${LOCATION}"
echo "   Date:     ${DATE}"
echo "   Models:   ${MODELS}"
echo "=============================================================================="

# 1. Environment & Path Configuration on CX3
export MONITORING_DATA="${MONITORING_DATA:-/rds/general/user/ri322/ephemeral/monitoring_data}"
export ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-/rds/general/user/ri322/ephemeral/sea-work}"

# 2. High-Performance Local Cache & GPU Framework Stability
export TMP_DIR="/tmp/${USER:-hpc_user}"
mkdir -p "${TMP_DIR}/numba" "${TMP_DIR}/torch_ext" "${TMP_DIR}/hf_cache"

export NUMBA_CACHE_DIR="${TMP_DIR}/numba"
export TORCH_EXTENSIONS_DIR="${TMP_DIR}/torch_ext"
export HF_HOME="${TMP_DIR}/hf_cache"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export CUDA_MODULE_LOADING=LAZY
export BACPIPE_DEVICE=auto

# 3. Initialize HPC Environment Modules (Required for GCCcore & C shared libraries)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if command -v module &> /dev/null; then
    module load tools/prod Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
fi

# Locate repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_PYTHON="${SCRIPT_DIR}/bacpipe/.venv/bin/python"
if [ ! -f "${VENV_PYTHON}" ]; then
    echo "❌ Error: Virtual environment python not found at ${VENV_PYTHON}"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "📡 PHASE 1: SIGNAL PROCESSING CHECK (Beamforming LabIR/SPIR, Mono, Signal Averaging)"
echo "=============================================================================="

# Check if processed WAV files are already complete
PROCESSED_WAV_COUNT=$(find "${ANALYSIS_OUTPUT}/${LOCATION}/${DATE}" -type f -name "*.wav" 2>/dev/null | wc -l || echo 0)

if [ "${PROCESSED_WAV_COUNT}" -ge 400 ]; then
    echo "✅ Signal processing already complete! Found ${PROCESSED_WAV_COUNT} processed WAV files in ${ANALYSIS_OUTPUT}/${LOCATION}/${DATE}."
    echo "⚡ Skipping Phase 1 and jumping directly to Deep Learning Inference."
else
    echo "Processing raw FLACs to WAVs (incremental mode: skipping existing files)..."
    "${VENV_PYTHON}" pipeline_signal_processing.py \
        --location "${LOCATION}" \
        --date "${DATE}" \
        --ir-types LabIR,SPIR1,SPIR2 \
        --labir-speakers S01,S05,S09,S12 \
        --labir-degrees 0,60,120,180,240,300 \
        --max-files 0
fi

echo ""
echo "=============================================================================="
echo "🧠 PHASE 2: MULTI-MODEL DEEP LEARNING EMBEDDING EXTRACTION"
echo "=============================================================================="
"${VENV_PYTHON}" bacpipe/pipeline_bacpipe.py \
    --location "${LOCATION}" \
    --date "${DATE}" \
    --models "${MODELS}" \
    --methods mono,sa,bf_LabIR,bf_SPIR \
    --device auto

echo ""
echo "=============================================================================="
echo "📊 PHASE 3: MATCHED-WINDOW DIRECTION SELECTION & PLOTLY DASHBOARD VISUALS"
echo "=============================================================================="
"${VENV_PYTHON}" visualize_bacpipe.py \
    --location "${LOCATION}" \
    --date "${DATE}"

echo ""
echo "=============================================================================="
echo "✅ PIPELINE COMPLETED SUCCESSFULLY FOR ${LOCATION} / ${DATE}"
echo "=============================================================================="
