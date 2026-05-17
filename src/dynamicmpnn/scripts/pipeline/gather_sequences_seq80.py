"""
Stage 1+2 of the processing pipeline for mmseqs2 80% sequence clusters.

Stage 0 (already done): reduced_mmcifs/ was built from mmcif_files/ by
    gather_sequences_reduce_mmcif.py — CIFs are symlinked under
    reduced_mmcifs/{pdb_id[1:3]}/{PDB_ID.upper()}.cif

Stage 1 (this script): For each cluster80 representative, read all member
    chain CIFs from reduced_mmcifs/, combine into a single multi-model CIF
    → pdb_cache_seq80/{cluster80_id}/

Stage 2 (this script): Align per-cluster sequences with ClustalOmega
    → pdb_cache_seq80/{cluster80_id}/{cluster80_id}_aligned.fasta

The cluster80 IDs (integers) become the .pt filenames in process_pt_seq80.py.

Usage (SLURM array, e.g. 200 tasks):
    python gather_sequences_seq80.py --chunk_idx $SLURM_ARRAY_TASK_ID --n_chunks 200
    python gather_sequences_seq80.py --mode align_only --chunk_idx 0 --n_chunks 200
"""

import argparse
import os
import sys
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from loguru import logger

# --- Paths (from environment, see .env.example) ---
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
CLUSTERING_BASE = Path(os.environ.get("CLUSTERING_BASE", PUBLIC_DB / "clustering"))
CLUSTER_CSV = CLUSTERING_BASE / "seq_clustering/output/collated/seqres_simple_clusters.csv"
REDUCED_MMCIF_DIR = PUBLIC_DB / "reduced_mmcifs"
PDB_CACHE_DIR     = PUBLIC_DB / "pdb_cache_seq80"
PDB_SPLIT_DIR     = PUBLIC_DB / "pdb_splitting_seq80"

sys.path.insert(0, str(Path(__file__).parents[3]))
from dynamicmpnn.scripts.download_codnas_mmcif import try_combine_cluster_chains
from dynamicmpnn.scripts.process_pt import align_all_pdbflex


def build_cluster_dict(csv_path: Path) -> dict:
    """
    Build {str(cluster80): [pdb_auth, ...]} from seqres_simple_clusters.csv.
    Uses pdb_auth (auth_asym_id) so chain IDs match BioPython MMCIFParser output.
    pdb_auth format: '{pdb_id}_{auth_chain}' e.g. '1km8_A'.
    """
    logger.info(f"Loading cluster CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["pdb_auth", "cluster80"])
    cluster_dict = df.groupby("cluster80")["pdb_auth"].apply(list).to_dict()
    cluster_dict = {str(k): v for k, v in cluster_dict.items()}
    logger.info(f"Loaded {len(cluster_dict):,} cluster80 groups from {len(df):,} chains")
    return cluster_dict


def get_chunk(cluster_dict: dict, chunk_idx: int, n_chunks: int) -> dict:
    """Return the slice of cluster_dict for this chunk index."""
    keys = sorted(cluster_dict.keys(), key=lambda x: int(x))
    chunk_size = max(1, len(keys) // n_chunks)
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < n_chunks - 1 else len(keys)
    chunk_keys = keys[start:end]
    logger.info(f"Chunk {chunk_idx}/{n_chunks}: {len(chunk_keys):,} clusters (keys {chunk_keys[0]}..{chunk_keys[-1]})")
    return {k: cluster_dict[k] for k in chunk_keys}


def multiprocess_gather(cluster_dict: dict, n_proc: int) -> list:
    """
    For each cluster80 in cluster_dict, combine all member chain CIFs from
    reduced_mmcifs/ into pdb_cache_seq80/{cluster80_id}/
    Returns list of successfully processed cluster IDs.
    """
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PDB_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # pdb_chain_dict is used only for single-chain CIF saving (cluster_id == pdb_id_chain).
    # Since cluster80 IDs are integers, this never matches — pass empty dict.
    pdb_chain_dict = {}

    successful = []
    pbar = tqdm(total=len(cluster_dict), desc="Combining cluster chains")

    def _on_done(result_id):
        pbar.update()

    with mp.get_context("spawn").Pool(n_proc) as pool:
        results = []
        for cluster_id, members in cluster_dict.items():
            save_dir = PDB_CACHE_DIR / cluster_id
            save_dir.mkdir(parents=True, exist_ok=True)
            results.append(
                pool.apply_async(
                    try_combine_cluster_chains,
                    args=(cluster_id, members, REDUCED_MMCIF_DIR,
                          str(save_dir), str(PDB_SPLIT_DIR), pdb_chain_dict),
                    callback=_on_done,
                )
            )
        pool.close()
        for r in results:
            try:
                result_id = r.get(timeout=600)  # 10 min max per cluster — skip hung workers
                if result_id:
                    successful.append(result_id)
            except mp.TimeoutError:
                logger.warning("Cluster timed out after 600s in Stage 1 — skipping")
            except Exception as e:
                logger.error(f"Stage 1 worker error: {e}")
        pool.join()

    pbar.close()
    logger.info(f"Stage 1 complete: {len(successful):,} / {len(cluster_dict):,} clusters succeeded")
    return successful


def parse_args():
    p = argparse.ArgumentParser(description="Gather and align sequences for mmseqs2 80% clusters")
    p.add_argument("--chunk_idx", type=int, required=True,
                   help="Index of this chunk (0-based, used as $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n_chunks", type=int, default=200,
                   help="Total number of chunks (= SLURM array size)")
    p.add_argument("--mode", choices=["full_pipeline", "align_only"], default="full_pipeline",
                   help="full_pipeline: stage 1 + 2; align_only: stage 2 only")
    p.add_argument("--n_proc_stage1", type=int, default=6,
                   help="Parallel workers for combining CIFs (stage 1)")
    p.add_argument("--n_proc_stage2", type=int, default=6,
                   help="Parallel workers for ClustalOmega alignment (stage 2)")
    return p.parse_args()


def main():
    args = parse_args()

    cluster_dict = build_cluster_dict(CLUSTER_CSV)
    chunk = get_chunk(cluster_dict, args.chunk_idx, args.n_chunks)

    if args.mode == "full_pipeline":
        logger.info(f"Stage 1: combining CIFs for {len(chunk):,} clusters")
        successful_ids = multiprocess_gather(chunk, n_proc=args.n_proc_stage1)
    elif args.mode == "align_only":
        # Only align clusters whose CIF already exists
        successful_ids = [
            cid for cid in chunk
            if (PDB_CACHE_DIR / cid / f"{cid}.cif").exists()
        ]
        logger.info(f"align_only: {len(successful_ids):,} clusters have existing CIFs")

    logger.info(f"Stage 2: aligning sequences for {len(successful_ids):,} clusters")
    align_all_pdbflex(successful_ids, PDB_CACHE_DIR, n_proc=args.n_proc_stage2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
