#!/bin/bash
# ================================================================
# Submit tanggal 2A400 ke PBS — cerdas: skip yang sudah selesai.
# Jalankan BERULANG setiap sesi baru. Sistem ingatan (L1+L2) 
# memastikan setiap run melanjutkan dari titik terakhir.
#
# Usage: bash run_2A400_all_dates.sh [all|birdnet|...]  [--dry-run]
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${1:-all}"
DRY_RUN=0
[[ "${2}" == "--dry-run" || "${1}" == "--dry-run" ]] && DRY_RUN=1
[[ "${DRY_RUN}" -eq 1 ]] && MODELS="${1:-all}" && [[ "${1}" == "--dry-run" ]] && MODELS="all"

ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-/rds/general/user/ri322/ephemeral/sea-work}"
EMB_BASE="${ANALYSIS_OUTPUT}/2A400/emb"
META_BASE="${SEA_RESULTS:-$HOME/sea-emb}/2A400"
mkdir -p "${SCRIPT_DIR}/logs"

# 11 model yang digunakan
ALL_MODELS=(
    audioprotopnet avesecho_passt biolingual birdaves_especies birdmae
    birdnet birdnet_v3 perch_bird perch_v2 protoclr vggish
)

# Fungsi: cek apakah satu tanggal sudah 100% selesai untuk semua model
date_is_fully_done() {
    local DATE="$1"
    for MODEL in "${ALL_MODELS[@]}"; do
        local SUMMARY="${META_BASE}/${MODEL}/${DATE}_summary.json"
        local EMB_MONO="${EMB_BASE}/${MODEL}/${DATE}_mono.npy"
        if [[ ! -f "${SUMMARY}" || ! -f "${EMB_MONO}" ]]; then
            return 1  # belum selesai
        fi
    done
    return 0  # semua model selesai
}

# 42 tanggal 2A400, dari terlama ke terkini
DATES=(
    2026-04-21 2026-04-22 2026-04-26 2026-04-27 2026-04-29
    2026-04-30 2026-05-01 2026-05-03 2026-05-04 2026-05-07
    2026-05-08 2026-05-14 2026-05-15 2026-05-18 2026-05-22
    2026-05-23 2026-05-27 2026-05-30 2026-05-31 2026-06-03
    2026-06-04 2026-06-05 2026-06-06 2026-06-07 2026-06-11
    2026-06-12 2026-06-14 2026-06-15 2026-06-18 2026-06-19
    2026-06-23 2026-06-24 2026-06-30 2026-07-01 2026-07-04
    2026-07-08 2026-07-09 2026-07-12 2026-07-13 2026-07-15
    2026-07-18 2026-07-29
)

echo "==================================================="
echo "  2A400 — Submit PBS jobs (smart resubmit)"
echo "  Models : ${MODELS}"
echo "  Dry run: ${DRY_RUN}"
echo "  Tanggal: ${#DATES[@]}"
echo "==================================================="
echo ""

DONE=0
SUBMITTED=0
FAILED=0

for DATE in "${DATES[@]}"; do
    if date_is_fully_done "${DATE}"; then
        echo "  ✅ ${DATE} — semua model selesai, dilewati"
        DONE=$((DONE + 1))
        continue
    fi

    # Hitung berapa model yang sudah selesai untuk info
    DONE_MODELS=0
    for MODEL in "${ALL_MODELS[@]}"; do
        [[ -f "${EMB_BASE}/${MODEL}/${DATE}_summary.json" ]] && DONE_MODELS=$((DONE_MODELS + 1))
    done
    REMAINING=$((${#ALL_MODELS[@]} - DONE_MODELS))

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  🔵 ${DATE} — submit (${DONE_MODELS}/${#ALL_MODELS[@]} model selesai, ${REMAINING} sisa)"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    JOB_ID=$(qsub \
        -N "sea-${DATE}" \
        -q v1_gpu72 \
        -l select=1:ncpus=4:mem=32gb:ngpus=1 \
        -l walltime=08:00:00 \
        -v "DATE=${DATE},MODELS=${MODELS},SCRIPT_DIR=${SCRIPT_DIR}" \
        -o "${SCRIPT_DIR}/logs/sea-2A400-${DATE}.out" \
        -e "${SCRIPT_DIR}/logs/sea-2A400-${DATE}.err" \
        "${SCRIPT_DIR}/run_single_date_worker.sh" 2>&1)

    if echo "${JOB_ID}" | grep -q "pbs"; then
        echo "  🚀 ${DATE} → ${JOB_ID}  (${DONE_MODELS}/${#ALL_MODELS[@]} model sudah ada)"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ❌ ${DATE} → GAGAL: ${JOB_ID}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "==================================================="
echo "  ✅ Sudah selesai : ${DONE} tanggal (dilewati)"
echo "  🚀 Di-submit     : ${SUBMITTED} tanggal"
echo "  ❌ Gagal submit  : ${FAILED} tanggal"
echo "==================================================="
echo ""
if [[ "${SUBMITTED}" -gt 0 && "${DRY_RUN}" -eq 0 ]]; then
    echo "Pantau antrian : qstat -u ri322"
    echo "Lihat log      : tail -f ${SCRIPT_DIR}/logs/sea-2A400-YYYY-MM-DD.out"
    echo ""
    echo "Jalankan script ini lagi setelah job-job selesai untuk submit sisa tanggal."
fi
