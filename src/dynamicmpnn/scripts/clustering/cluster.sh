#!/bin/bash
# Usage: bash cluster.sh <seqres|observed> <threshold> <simple|sensitive>
# Env:   CLUSTERING_BASE - base directory for clustering data (required)
set -euo pipefail

BASE_DIR="${CLUSTERING_BASE:?Set CLUSTERING_BASE in .env or environment}"
INPUT_DIR="${BASE_DIR}/seq_extraction/output"
OUT_BASE="${BASE_DIR}/seq_clustering/output"
TMP_BASE="${BASE_DIR}/seq_clustering/tmp"

INPUT="${1:?}"
THRESH="${2:?}"
MODE="${3:?}"
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

ID_INT=$(awk -v t="${THRESH}" 'BEGIN{printf "%d", t*100}')
RUN_DIR="${OUT_BASE}/${INPUT}/${MODE}/id${ID_INT}"
PREFIX="${RUN_DIR}/${INPUT}"
TMP="${TMP_BASE}/${INPUT}_${MODE}_id${ID_INT}"

mkdir -p "${RUN_DIR}" "${TMP}"
echo "[$(date '+%F %T')] START  ${INPUT} ${MODE} id${ID_INT}"

if [[ "${MODE}" == "simple" ]]; then
    mmseqs easy-cluster "${INPUT_DIR}/${INPUT}.fasta" "${PREFIX}" "${TMP}" \
        --min-seq-id "${THRESH}" \
        -c 0.8                   \
        --cov-mode 0             \
        --threads "${THREADS}"   \
        --gpu 1
else
    mmseqs easy-cluster "${INPUT_DIR}/${INPUT}.fasta" "${PREFIX}" "${TMP}" \
        --min-seq-id "${THRESH}" \
        -c 0.8                   \
        --cov-mode 0             \
        -s 7.5                   \
        --cluster-mode 1         \
        --cluster-reassign 1     \
        --max-seqs 300           \
        --kmer-per-seq 80        \
        --alignment-mode 3       \
        --threads "${THREADS}"   \
        --gpu 1
fi

rm -rf "${TMP}"
echo "[$(date '+%F %T')] DONE   ${RUN_DIR}"
