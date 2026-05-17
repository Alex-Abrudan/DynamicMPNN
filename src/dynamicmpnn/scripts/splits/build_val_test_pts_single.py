"""
Build single-chain val/test .pt files from curated CoDNaS conformational pair sets.

For each pair (protein1, protein2) in val_set.csv / test_set.csv:
  1. Parse PDB ID and auth chain: '1BDT_A' -> pdb='1bdt', chain='A'
  2. Assign a unique pseudo-cluster ID:
       val pairs:  900001, 900002, ...
       test pairs: 800001, 800002, ...
  3. Stage 1 — Extract specific chains into combined CIF + mapping
  4. Stage 2 — ClustalOmega alignment
  5. Stage 3 — single-chain codnas_to_pyg → save .pt

This is the SINGLE-CHAIN version: each .pt contains exactly 2 conformations
(the two chains from protein1 and protein2) treated as the same chain.

Outputs
-------
val_pt_single_chain/{900001..}.pt
test_pt_single_chain/{800001..}.pt
data/val_pts_single.csv      (pdb_auth column for the datamodule)
data/test_pts_single.csv

Usage
-----
python build_val_test_pts_single.py [--split val|test|both] [--n_proc N]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Paths (from environment, see .env.example)
# ---------------------------------------------------------------------------
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parents[4]))

PDB_CACHE_DIR     = PUBLIC_DB / "pdb_cache_val_test_single"
VAL_OUT_DIR       = PUBLIC_DB / "val_pt_single_chain"
TEST_OUT_DIR      = PUBLIC_DB / "test_pt_single_chain"

DATA_DIR = PROJECT_PATH / "data"
VAL_SET_CSV  = DATA_DIR / "val_set.csv"
TEST_SET_CSV = DATA_DIR / "test_set.csv"

VAL_ID_OFFSET  = 900000
TEST_ID_OFFSET = 800000

# ---------------------------------------------------------------------------
# Setup sys.path so we can import from dynamicmpnn
# ---------------------------------------------------------------------------
from dynamicmpnn.datamodules.utils_mmcif_single import codnas_to_pyg

import json
import subprocess
from Bio import PDB
from Bio.PDB import MMCIFIO


def find_mmcif_file(pdb_id: str) -> Path:
    """Find mmCIF file for a PDB ID."""
    pdb_id = pdb_id.lower()
    # Try reduced_mmcifs (divided layout)
    reduced_path = PUBLIC_DB / "reduced_mmcifs" / pdb_id[1:3] / f"{pdb_id.upper()}.cif"
    if reduced_path.exists():
        return reduced_path
    # Fallback to flat mmcif_files
    flat_path = PUBLIC_DB / "mmcif_files" / f"{pdb_id}.cif"
    if flat_path.exists():
        return flat_path
    return None


def get_chain_sequence(structure, chain_id: str) -> str:
    """Extract amino acid sequence from a chain."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    seq = []
    for model in structure:
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if residue.id[0] == ' ' and residue.resname in three_to_one:
                    seq.append(three_to_one[residue.resname])
            break
    return ''.join(seq)


def process_cluster(cluster_id: str, cluster_members: list, output_dir: Path) -> bool:
    """Process a single-chain cluster: extract specific chains into combined CIF."""
    parser = PDB.MMCIFParser(QUIET=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_structure = PDB.Structure.Structure(cluster_id)
    fasta_lines = []
    mapping = {}
    model_num = 1

    for pdb_auth in cluster_members:
        parts = pdb_auth.split('_')
        if len(parts) < 2:
            logger.warning(f"Invalid pdb_auth format: {pdb_auth}, skipping")
            continue

        pdb_id = parts[0].lower()
        chain_id = parts[1]

        # Handle NMR model suffix
        if '-' in pdb_id:
            pdb_id_base, nmr_model = pdb_id.rsplit('-', 1)
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

        # Select NMR model if specified
        if nmr_model is not None:
            if nmr_model < len(structure):
                source_model = structure[nmr_model]
            else:
                source_model = structure[0]
        else:
            source_model = structure[0]

        # Find the chain
        if chain_id not in [c.id for c in source_model]:
            logger.warning(f"Chain {chain_id} not found in {pdb_id_base}, skipping {pdb_auth}")
            continue

        source_chain = source_model[chain_id]
        seq = get_chain_sequence(structure, chain_id)
        if not seq or len(seq) < 10:
            logger.warning(f"Sequence too short for {pdb_auth}, skipping")
            continue

        # Create new model with this chain
        new_model = PDB.Model.Model(model_num)
        new_chain = source_chain.copy()
        new_chain.id = chain_id
        new_model.add(new_chain)
        combined_structure.add(new_model)

        fasta_lines.append(f">{pdb_auth.upper()}")
        fasta_lines.append(seq)
        mapping[model_num] = pdb_auth.upper()
        model_num += 1

    if model_num < 3:  # Need at least 2 members
        logger.warning(f"Cluster {cluster_id}: only {model_num-1} valid chains")
        return False

    # Save CIF
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
    """Run ClustalOmega alignment."""
    fasta_in = output_dir / f"{cluster_id}.fasta"
    fasta_out = output_dir / f"{cluster_id}_aligned.fasta"

    if not fasta_in.exists():
        return False

    try:
        result = subprocess.run(
            ["/usr/local/Cluster-Apps/clustalo/1.2.4/clustalo",
             "-i", str(fasta_in), "-o", str(fasta_out), "--force"],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"ClustalOmega failed for {cluster_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_protein_id(protein_id: str):
    """
    Parse a protein ID from val_set / test_set into (pdb_auth).

    Formats:
      '1BDT_A'   -> '1bdt_A'
      '2JP1-1_A' -> '2jp1-1_A'
    """
    pdb_part, chain = protein_id.rsplit("_", 1)
    return f"{pdb_part.lower()}_{chain}"


def build_protein_chain_map(members: list) -> dict:
    """Build a minimal protein_chain_map for exactly the chains in `members`."""
    chain_map = {}
    for mem in members:
        pdb_part = mem.rsplit("_", 1)[0].upper()
        chain = mem.rsplit("_", 1)[1]
        chain_map.setdefault(pdb_part, set()).add(chain)
    return chain_map


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_set(
    set_csv: Path,
    id_offset: int,
    output_dir: Path,
) -> list:
    """
    Process one dataset split (val or test) for single-chain.

    Returns list of cluster_id strings for which a .pt was saved.
    """
    df = pd.read_csv(set_csv, usecols=["protein1", "protein2"])
    logger.info(f"\nProcessing {len(df)} pairs from {set_csv.name} (SINGLE-CHAIN)")

    output_dir.mkdir(parents=True, exist_ok=True)
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pt_ids = []

    for idx, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        cluster_id = str(id_offset + idx + 1)

        # Convert to pdb_auth format (lowercase PDB, original case chain)
        mem1 = parse_protein_id(p1)
        mem2 = parse_protein_id(p2)
        members = [mem1, mem2]

        cluster_dir = PDB_CACHE_DIR / cluster_id

        # Stage 1: Use single-chain process_cluster (creates CIF + FASTA + mapping)
        logger.info(f"Processing {cluster_id}: {p1} + {p2}")
        success = process_cluster(cluster_id, members, cluster_dir)
        if not success:
            logger.warning(f"  Stage 1 failed for {cluster_id}")
            continue

        # Stage 2: ClustalOmega alignment
        if not run_clustal_omega(cluster_id, cluster_dir):
            logger.warning(f"  Stage 2 (alignment) failed for {cluster_id}")
            continue

        # Stage 3: Create .pt file
        aligned_fasta = cluster_dir / f"{cluster_id}_aligned.fasta"
        if not aligned_fasta.exists():
            logger.warning(f"  No aligned FASTA for {cluster_id}")
            continue

        protein_chain_map = build_protein_chain_map([m.upper() for m in members])
        pyg_data = codnas_to_pyg(
            cluster_id, [m.upper() for m in members],
            str(PDB_CACHE_DIR), str(PDB_CACHE_DIR), protein_chain_map
        )

        if pyg_data is not None:
            pt_path = output_dir / f"{cluster_id}.pt"
            torch.save(pyg_data, pt_path)
            pt_ids.append(cluster_id)
            logger.info(f"  Saved {pt_path.name}")
        else:
            logger.warning(f"  codnas_to_pyg returned None for {cluster_id}")

    logger.info(f"Done: {len(pt_ids)}/{len(df)} .pt files saved to {output_dir}")
    return pt_ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    val_pts, test_pts = [], []

    if args.split in ("val", "both"):
        val_pts = process_set(VAL_SET_CSV, VAL_ID_OFFSET, VAL_OUT_DIR)
        pt_ids_df = pd.DataFrame({"pdb_auth": val_pts})
        out_val_csv = DATA_DIR / "val_pts_single.csv"
        pt_ids_df.to_csv(out_val_csv, index=False)
        logger.info(f"Val CSV -> {out_val_csv}  ({len(val_pts)} entries)")

    if args.split in ("test", "both"):
        test_pts = process_set(TEST_SET_CSV, TEST_ID_OFFSET, TEST_OUT_DIR)
        pt_ids_df = pd.DataFrame({"pdb_auth": test_pts})
        out_test_csv = DATA_DIR / "test_pts_single.csv"
        pt_ids_df.to_csv(out_test_csv, index=False)
        logger.info(f"Test CSV -> {out_test_csv}  ({len(test_pts)} entries)")

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
