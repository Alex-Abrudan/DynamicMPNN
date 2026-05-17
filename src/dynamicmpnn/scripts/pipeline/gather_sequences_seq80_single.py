"""
Single-chain version of gather_sequences_seq80.py

For each cluster80, extracts ONLY the single chain specified in pdb_auth
(e.g., "1ABC_A" -> only chain A from 1ABC), and combines them into a
multi-model CIF where each model is one chain from one PDB.

This differs from the multi-chain version which extracts ALL chains from
each PDB member.

Output:
    pdb_cache_seq80_single/{cluster80_id}/{cluster80_id}.cif
    pdb_cache_seq80_single/{cluster80_id}/{cluster80_id}.fasta
    pdb_cache_seq80_single/{cluster80_id}/{cluster80_id}_aligned.fasta
    pdb_cache_seq80_single/{cluster80_id}/{cluster80_id}_mapping.json

The mapping.json maps model number to pdb_auth (e.g., {1: "1ABC_A", 2: "2XYZ_B"})

Usage (SLURM array):
    python gather_sequences_seq80_single.py --chunk_idx $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

import argparse
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from loguru import logger
from Bio import PDB
from Bio.PDB import MMCIFIO, Select
from Bio.Align.Applications import ClustalOmegaCommandline
import subprocess
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)

# --- Paths (from environment, see .env.example) ---
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
CLUSTERING_BASE = Path(os.environ.get("CLUSTERING_BASE", PUBLIC_DB / "clustering"))
CLUSTER_CSV = CLUSTERING_BASE / "seq_clustering/output/collated/seqres_simple_clusters.csv"
REDUCED_MMCIF_DIR = PUBLIC_DB / "reduced_mmcifs"
PDB_CACHE_DIR = PUBLIC_DB / "pdb_cache_seq80_single"

from dynamicmpnn.types import STANDARD_AMINO_ACID_MAPPING_3_TO_1


class SingleChainSelect(Select):
    """Select only atoms from a specific chain."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # Only accept standard amino acids (not HETATM, water, etc.)
        return residue.id[0] == ' '


def find_mmcif_file(pdb_id: str) -> Path:
    """Find mmCIF file in reduced_mmcifs/ divided layout."""
    pdb_id_upper = pdb_id.upper()
    subdir = pdb_id[1:3].lower()
    cif_path = REDUCED_MMCIF_DIR / subdir / f"{pdb_id_upper}.cif"
    if cif_path.exists():
        return cif_path
    return None


def get_chain_sequence(structure, chain_id: str) -> str:
    """Extract amino acid sequence from a chain."""
    seq = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        resname = residue.resname.upper()
                        aa = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(resname, 'X')
                        seq.append(aa)
                return ''.join(seq)
    return ''


def combine_single_chains(cluster_id: str, cluster_members: list, output_dir: Path) -> bool:
    """
    For each pdb_auth in cluster_members, extract only that single chain
    and combine all chains into a multi-model CIF.

    Args:
        cluster_id: The cluster80 ID (integer as string)
        cluster_members: List of pdb_auth strings, e.g., ["1ABC_A", "2XYZ_B"]
        output_dir: Directory to save output files

    Returns:
        True if successful, False otherwise
    """
    parser = PDB.MMCIFParser(QUIET=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_structure = PDB.Structure.Structure(cluster_id)
    fasta_lines = []
    mapping = {}  # model_num -> pdb_auth
    model_num = 1

    for pdb_auth in cluster_members:
        parts = pdb_auth.split('_')
        if len(parts) < 2:
            logger.warning(f"Invalid pdb_auth format: {pdb_auth}, skipping")
            continue

        pdb_id = parts[0].lower()
        chain_id = parts[1]  # Keep case-sensitive

        # Handle NMR model suffix (e.g., "1ABC-1_A")
        if '-' in pdb_id:
            pdb_id_base, nmr_model = pdb_id.split('-', 1)
            nmr_model = int(nmr_model)
        else:
            pdb_id_base = pdb_id
            nmr_model = None

        cif_path = find_mmcif_file(pdb_id_base)
        if cif_path is None:
            logger.warning(f"mmCIF not found for {pdb_id_base}, skipping {pdb_auth}")
            continue

        try:
            structure = parser.get_structure(pdb_id_base, str(cif_path))
        except Exception as e:
            logger.warning(f"Failed to parse {cif_path}: {e}")
            continue

        # Select the appropriate model (for NMR structures)
        if nmr_model is not None:
            if nmr_model < len(structure):
                source_model = structure[nmr_model]
            else:
                logger.warning(f"NMR model {nmr_model} not found in {pdb_id_base}, using model 0")
                source_model = structure[0]
        else:
            source_model = structure[0]

        # Find the chain
        if chain_id not in [c.id for c in source_model]:
            logger.warning(f"Chain {chain_id} not found in {pdb_id_base}, skipping {pdb_auth}")
            continue

        source_chain = source_model[chain_id]

        # Get sequence
        seq = get_chain_sequence(structure, chain_id)
        if not seq or len(seq) < 10:
            logger.warning(f"Sequence too short for {pdb_auth} (len={len(seq)}), skipping")
            continue

        # Create new model with just this chain
        new_model = PDB.Model.Model(model_num)
        new_chain = source_chain.copy()
        new_chain.id = chain_id
        new_model.add(new_chain)
        combined_structure.add(new_model)

        # Add to FASTA
        fasta_lines.append(f">{pdb_auth.upper()}")
        fasta_lines.append(seq)

        # Add to mapping
        mapping[model_num] = pdb_auth.upper()
        model_num += 1

    if model_num < 3:  # Need at least 2 members
        logger.warning(f"Cluster {cluster_id}: only {model_num-1} valid chains, skipping")
        return False

    # Save combined CIF
    cif_out = output_dir / f"{cluster_id}.cif"
    io = MMCIFIO()
    io.set_structure(combined_structure)
    io.save(str(cif_out))

    # Save FASTA
    fasta_out = output_dir / f"{cluster_id}.fasta"
    with open(fasta_out, 'w') as f:
        f.write('\n'.join(fasta_lines) + '\n')

    # Save mapping
    mapping_out = output_dir / f"{cluster_id}_mapping.json"
    with open(mapping_out, 'w') as f:
        json.dump(mapping, f, indent=2)

    return True


def run_clustal_omega(cluster_id: str, output_dir: Path) -> bool:
    """Run ClustalOmega alignment on the cluster FASTA."""
    fasta_in = output_dir / f"{cluster_id}.fasta"
    fasta_out = output_dir / f"{cluster_id}_aligned.fasta"

    if not fasta_in.exists():
        return False

    # Check if only 1 sequence (nothing to align)
    with open(fasta_in) as f:
        n_seqs = sum(1 for line in f if line.startswith('>'))

    if n_seqs < 2:
        logger.warning(f"Cluster {cluster_id}: only {n_seqs} sequence(s), copying as-is")
        import shutil
        shutil.copy(fasta_in, fasta_out)
        return True

    try:
        cmd = ClustalOmegaCommandline(
            infile=str(fasta_in),
            outfile=str(fasta_out),
            verbose=False,
            auto=True,
            force=True
        )
        subprocess.run(str(cmd).split(), check=True, capture_output=True, timeout=300)
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"ClustalOmega timeout for cluster {cluster_id}")
        return False
    except Exception as e:
        logger.warning(f"ClustalOmega failed for cluster {cluster_id}: {e}")
        return False


def process_cluster(args):
    """Process a single cluster: combine chains + align."""
    cluster_id, cluster_members = args
    output_dir = PDB_CACHE_DIR / cluster_id

    try:
        # Stage 1: Combine single chains
        if not combine_single_chains(cluster_id, cluster_members, output_dir):
            return None

        # Stage 2: Align
        if not run_clustal_omega(cluster_id, output_dir):
            return None

        return cluster_id
    except Exception as e:
        logger.error(f"Failed to process cluster {cluster_id}: {e}")
        return None


def build_cluster_dict(csv_path: Path) -> dict:
    """Build {str(cluster80): [pdb_auth, ...]} from seqres_simple_clusters.csv."""
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
    logger.info(f"Chunk {chunk_idx}/{n_chunks}: {len(chunk_keys):,} clusters")
    return {k: cluster_dict[k] for k in chunk_keys}


def parse_args():
    p = argparse.ArgumentParser(description="Gather single-chain sequences for mmseqs2 80% clusters")
    p.add_argument("--chunk_idx", type=int, required=True,
                   help="Index of this chunk (0-based)")
    p.add_argument("--n_chunks", type=int, default=200,
                   help="Total number of chunks")
    p.add_argument("--n_proc", type=int, default=None,
                   help="Number of worker processes")
    return p.parse_args()


def main():
    args = parse_args()
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cluster_dict = build_cluster_dict(CLUSTER_CSV)
    chunk = get_chunk(cluster_dict, args.chunk_idx, args.n_chunks)

    n_proc = args.n_proc or len(os.sched_getaffinity(0))
    logger.info(f"Processing {len(chunk):,} clusters with {n_proc} workers")

    with mp.Pool(n_proc) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_cluster, chunk.items()),
            total=len(chunk),
            desc="Processing clusters"
        ))

    n_ok = sum(1 for r in results if r is not None)
    logger.info(f"Done: {n_ok:,} / {len(chunk):,} clusters processed successfully")


if __name__ == "__main__":
    main()
