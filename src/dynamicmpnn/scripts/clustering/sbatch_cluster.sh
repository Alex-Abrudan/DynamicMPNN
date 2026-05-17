#!/bin/bash
#SBATCH -p ampere
#SBATCH --job-name=mmseqs_cluster
#SBATCH -A LJC-ACA41-SL2-GPU
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mmseqs_cluster_%j.out

set -euo pipefail

# Directory containing this script (and cluster.sh, build_cluster_csv.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory for clustering data (input fastas, output clusters)
# Set in .env or export before running
CLUSTERING_BASE="${CLUSTERING_BASE:?Set CLUSTERING_BASE in .env or environment}"

PYTHON_BIN="${PYTHON_BIN:-python}"
COLLATE_SCRIPT="${SCRIPT_DIR}/build_cluster_csv.py"
CLUSTER_SCRIPT="${SCRIPT_DIR}/cluster.sh"

# Load mmseqs module - update this path for your cluster
MMSEQS_MODULE="${MMSEQS_MODULE:-}"
if [[ -n "${MMSEQS_MODULE}" ]]; then
    module purge
    source "${MMSEQS_MODULE}"
fi

mkdir -p "${SCRIPT_DIR}/logs"

THRESH_LIST=(0.3 0.4 0.8 0.9)

echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CLUSTERING_BASE: ${CLUSTERING_BASE}"
echo "Running clustering for thresholds: ${THRESH_LIST[*]}"
echo ""

for INPUT in seqres; do
    for MODE in simple; do
        for THRESH in "${THRESH_LIST[@]}"; do
            echo "=== ${INPUT} @ ${THRESH} [${MODE}] ==="
            CLUSTERING_BASE="${CLUSTERING_BASE}" bash "${CLUSTER_SCRIPT}" "${INPUT}" "${THRESH}" "${MODE}"
            echo ""
        done
    done
done

echo "=== Building cluster CSV ==="
"${PYTHON_BIN}" "${COLLATE_SCRIPT}" seqres simple && echo "Collation complete."

echo "All clustering runs complete."
