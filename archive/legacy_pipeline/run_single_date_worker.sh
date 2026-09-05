#!/bin/bash
# PBS worker: proses satu tanggal 2A400 (dipanggil oleh run_2A400_all_dates.sh)
# Env vars dioper dari qsub -v: DATE, MODELS, SCRIPT_DIR

source /etc/profile.d/modules.sh 2>/dev/null || true
module load tools/prod Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export MONITORING_DATA="${MONITORING_DATA:-/rds/general/user/ri322/ephemeral/monitoring_data}"
export ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-/rds/general/user/ri322/ephemeral/sea-work}"
export EPHEMERAL_WORK="${EPHEMERAL_WORK:-/rds/general/user/ri322/ephemeral/sea-work}"
export TMP_DIR="/tmp/${USER:-ri322}"
mkdir -p "${TMP_DIR}/numba" "${TMP_DIR}/torch_ext"
export NUMBA_CACHE_DIR="${TMP_DIR}/numba"
export TORCH_EXTENSIONS_DIR="${TMP_DIR}/torch_ext"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export CUDA_MODULE_LOADING=LAZY
export BACPIPE_DEVICE=auto
export ORT_DISABLE_THREAD_AFFINITY=1
export OMP_NUM_THREADS=4
export ORT_NUM_THREADS=4

cd "${SCRIPT_DIR}"
VENV_DIR="${SCRIPT_DIR}/bacpipe/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

# Export all NVIDIA CUDA 12 & cuDNN 9 shared libraries for ONNX GPU
NV_LIBS=$(find "${VENV_DIR}/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH}"

mkdir -p "${SCRIPT_DIR}/logs"

# ── AUTO-SYMLINK KE EPHEMERAL (Proteksi Kuota HOME 1TB) ───────────
# Pastikan folder WAV audio berada di Ephemeral (11TB) & disymlink ke Home
EPHEM_DATE_DIR="${EPHEMERAL_WORK}/2A400/${DATE}"
HOME_DATE_DIR="${ANALYSIS_OUTPUT}/2A400/${DATE}"
mkdir -p "${EPHEM_DATE_DIR}"
if [ ! -L "${HOME_DATE_DIR}" ] && [ ! -d "${HOME_DATE_DIR}" ]; then
    mkdir -p "$(dirname "${HOME_DATE_DIR}")"
    ln -s "${EPHEM_DATE_DIR}" "${HOME_DATE_DIR}"
    echo "🔗 Auto-symlink aktif: ${HOME_DATE_DIR} -> ${EPHEM_DATE_DIR}"
fi
# ──────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════"
echo "  Job: 2A400 / ${DATE}"
echo "  Models: ${MODELS:-all}"
echo "  PBS ID: ${PBS_JOBID:-lokal}"
echo "  WAV Path (Ephemeral): ${EPHEM_DATE_DIR}"
echo "════════════════════════════════════════════"

# Phase 1: Signal processing (lewati jika WAV sudah ada)
WAV_COUNT=$(find "${HOME_DATE_DIR}" -type f -name "*.wav" 2>/dev/null | wc -l || echo 0)
if [ "${WAV_COUNT}" -ge 400 ]; then
    echo "Phase 1: Lewati — ${WAV_COUNT} WAV sudah ada di Ephemeral"
else
    echo "Phase 1: Proses sinyal (beamforming, mono, SA) langsung ke Ephemeral..."
    "${VENV_PYTHON}" pipeline_signal_processing.py \
        --location 2A400 \
        --date "${DATE}" \
        --ir-types LabIR,SPIR1,SPIR2 \
        --labir-speakers S01,S05,S09,S12 \
        --labir-degrees 0,60,120,180,240,300 \
        --max-files 0 || echo "Phase 1 error (non-fatal, lanjut ke Phase 2)"
fi

# Phase 2: Embedding (Lapisan 1+2 aktif — skip model done, resume WAV)
echo ""
echo "Phase 2: Ekstraksi embedding dengan sistem ingatan & GPU..."
"${VENV_PYTHON}" bacpipe/pipeline_bacpipe.py \
    --location 2A400 \
    --date "${DATE}" \
    --models "${MODELS:-all}" \
    --methods mono,sa,bf_LabIR,bf_SPIR \
    --device auto

EXIT_CODE=$?
echo ""
echo "════════════════════════════════════════════"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "  ✅ SELESAI: 2A400 / ${DATE}"
else
    echo "  ❌ ERROR (exit ${EXIT_CODE}): 2A400 / ${DATE}"
fi
echo "════════════════════════════════════════════"
exit "${EXIT_CODE}"
