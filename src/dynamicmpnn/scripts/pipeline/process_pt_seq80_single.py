"""
Stage 3 of the single-chain processing pipeline for mmseqs2 80% sequence clusters.

Reads per-cluster CIFs + aligned FASTAs from pdb_cache_seq80_single/
(produced by gather_sequences_seq80_single.py) and featurises each cluster
into a .pt file → train_pt_single_chain/{cluster80_id}.pt

This is the single-chain version: each member in the cluster is exactly one
chain from one PDB (specified by pdb_auth, e.g., "1ABC_A" -> chain A only).

The cluster80 IDs (integers) match the .pt filenames used by the
datamodule when cluster_sampling_csv references them.

Usage:
    python process_pt_seq80_single.py [--chunk_idx N --n_chunks 200] [--n_proc 16]

    # Process all clusters at once (if enough RAM/CPU):
    python process_pt_seq80_single.py --n_proc 32
"""

import argparse
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path
from functools import partial
from tqdm import tqdm
import pandas as pd
import torch
from loguru import logger

# --- Paths (from environment, see .env.example) ---
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
CLUSTERING_BASE = Path(os.environ.get("CLUSTERING_BASE", PUBLIC_DB / "clustering"))
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parents[4]))
CLUSTER_CSV = CLUSTERING_BASE / "seq_clustering/output/collated/seqres_simple_clusters.csv"
PDB_CACHE_DIR = PUBLIC_DB / "pdb_cache_seq80_single"
PROCESSED_DIR = PUBLIC_DB / "train_pt_single_chain"
CHAIN_MAP_JSON = PROJECT_PATH / "data/pdb_protein_chain_mappings.json"

from dynamicmpnn.datamodules.utils_mmcif_single import codnas_to_pyg


def load_chain_map(json_path: Path) -> dict:
    """
    Load pdb_protein_chain_mappings.json → {PDB_ID_UPPER: set(chain_ids)}.
    Chain IDs are stored as-is (case-sensitive auth asym IDs).
    PDB IDs are uppercased as keys since PDB IDs are case-insensitive.
    """
    logger.info(f"Loading chain allow-list from {json_path}")
    with open(json_path) as f:
        raw = json.load(f)

    protein_chain_map = {}
    for pdb_id, chain_list in raw.items():
        if chain_list:
            protein_chain_map[pdb_id.upper()] = set(chain_list)

    logger.info(f"Loaded allow-list for {len(protein_chain_map):,} PDBs")
    return protein_chain_map


def build_cluster_dict(csv_path: Path) -> dict:
    """Build {str(cluster80): [pdb_auth, ...]} from the CSV.
    Uses pdb_auth (auth_asym_id) so chain IDs match BioPython MMCIFParser output."""
    df = pd.read_csv(csv_path, usecols=["pdb_auth", "cluster80"])
    cluster_dict = df.groupby("cluster80")["pdb_auth"].apply(list).to_dict()
    return {str(k): v for k, v in cluster_dict.items()}


def get_chunk(cluster_dict: dict, chunk_idx: int, n_chunks: int) -> dict:
    keys = sorted(cluster_dict.keys(), key=lambda x: int(x))
    chunk_size = max(1, len(keys) // n_chunks)
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < n_chunks - 1 else len(keys)
    chunk_keys = keys[start:end]
    return {k: cluster_dict[k] for k in chunk_keys}


def save_cluster_pt(cluster_item, protein_chain_map):
    """Featurise one cluster and save its .pt file. Returns cluster_id or None."""
    cluster_id, cluster_members = cluster_item
    output_path = PROCESSED_DIR / f"{cluster_id}.pt"

    try:
        # Single-chain version: cif_dir and fasta_dir are the same
        # (pdb_cache_seq80_single/{cluster_id}/ contains both CIF and FASTAs)
        pyg_data = codnas_to_pyg(
            cluster_id, cluster_members,
            cif_dir=str(PDB_CACHE_DIR),
            fasta_dir=str(PDB_CACHE_DIR),
            protein_chain_map=protein_chain_map
        )
        if pyg_data is not None:
            torch.save(pyg_data, output_path)
            return cluster_id
        else:
            logger.warning(f"codnas_to_pyg returned None for cluster {cluster_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to process cluster {cluster_id}: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Featurise pdb_cache_seq80_single → train_pt_single_chain")
    p.add_argument("--chunk_idx", type=int, default=None,
                   help="Chunk index (optional; if omitted, process all clusters)")
    p.add_argument("--n_chunks", type=int, default=200,
                   help="Total number of chunks (for SLURM array)")
    p.add_argument("--n_proc", type=int, default=None,
                   help="Number of worker processes (default: all available CPUs)")
    return p.parse_args()


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cluster_dict = build_cluster_dict(CLUSTER_CSV)
    protein_chain_map = load_chain_map(CHAIN_MAP_JSON)

    if args.chunk_idx is not None:
        cluster_dict = get_chunk(cluster_dict, args.chunk_idx, args.n_chunks)
        logger.info(f"Chunk {args.chunk_idx}/{args.n_chunks}: {len(cluster_dict):,} clusters")
    else:
        logger.info(f"Processing all {len(cluster_dict):,} clusters")

    # Skip clusters without a CIF (gather_sequences_seq80_single may not have finished them)
    missing_cif = [
        cid for cid in cluster_dict
        if not (PDB_CACHE_DIR / cid / f"{cid}.cif").exists()
    ]
    if missing_cif:
        logger.warning(f"Skipping {len(missing_cif):,} clusters missing CIF in {PDB_CACHE_DIR}")
        cluster_dict = {k: v for k, v in cluster_dict.items() if k not in missing_cif}

    logger.info(f"To process: {len(cluster_dict):,} clusters (overwriting any existing .pt files)")

    n_proc = args.n_proc or len(os.sched_getaffinity(0))
    worker = partial(save_cluster_pt, protein_chain_map=protein_chain_map)

    with mp.Pool(n_proc) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker, cluster_dict.items()),
            total=len(cluster_dict),
            desc="Featurising clusters",
        ))

    n_ok = sum(1 for r in results if r is not None)
    logger.info(f"Done: {n_ok:,} / {len(cluster_dict):,} clusters saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
