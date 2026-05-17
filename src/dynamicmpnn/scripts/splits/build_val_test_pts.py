"""
Build val/test .pt files from curated CoDNaS conformational pair sets.

For each pair (protein1, protein2) in val_set.csv / test_set.csv:
  1. Parse PDB ID, model number (NMR), and auth chain:
       '2JP1-1_A' -> pdb='2jp1', model=1, chain='A'
       '1BDT_A'   -> pdb='1bdt', model=None, chain='A'
  2. Assign a unique pseudo-cluster ID:
       val pairs:  900001, 900002, ...
       test pairs: 800001, 800002, ...
  3. Stage 1 — combine the 2 CIFs into pdb_cache_val_test/{cluster_id}/
  4. Stage 2 — ClustalOmega alignment
  5. Stage 3 — codnas_to_pyg → save .pt to the output directory

Outputs
-------
val_pt_multi_chain/{900001..}.pt
test_pt_multi_chain/{800001..}.pt
data/val_pts.csv      pdb_auth column for the datamodule
data/test_pts.csv
data/train_seq80_pool_filtered.csv  train_seq80_pool.csv minus val/test cluster30 groups

Usage
-----
python build_val_test_pts.py [--split val|test|both] [--n_proc N]
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
CLUSTERING_BASE = Path(os.environ.get("CLUSTERING_BASE", PUBLIC_DB / "clustering"))

REDUCED_MMCIF_DIR = PUBLIC_DB / "reduced_mmcifs"
PDB_CACHE_DIR     = PUBLIC_DB / "pdb_cache_val_test"
PDB_SPLIT_DIR     = PUBLIC_DB / "pdb_splitting_val_test"
VAL_OUT_DIR       = PUBLIC_DB / "val_pt_multi_chain"
TEST_OUT_DIR      = PUBLIC_DB / "test_pt_multi_chain"

DATA_DIR = PROJECT_PATH / "data"
VAL_SET_CSV  = DATA_DIR / "val_set.csv"
TEST_SET_CSV = DATA_DIR / "test_set.csv"

CLUSTER_CSV = CLUSTERING_BASE / "seq_clustering/output/collated/seqres_simple_clusters.csv"
TRAIN_POOL_CSV = DATA_DIR / "train_seq80_pool.csv"
SPLIT_OUT_DIR = DATA_DIR

VAL_ID_OFFSET  = 900000
TEST_ID_OFFSET = 800000

# ---------------------------------------------------------------------------
# Setup sys.path so we can import from dynamicmpnn
# ---------------------------------------------------------------------------
from dynamicmpnn.scripts.download_codnas_mmcif import try_combine_cluster_chains
from dynamicmpnn.scripts.process_pt import align_all_pdbflex
from dynamicmpnn.datamodules.utils_mmcif import codnas_to_pyg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_protein_id(protein_id: str):
    """
    Parse a protein ID from val_set / test_set into (pdb_id, model_num, chain_id).

    Formats:
      '1BDT_A'   -> ('1bdt', None, 'A')
      '2JP1-1_A' -> ('2jp1', 1, 'A')
    """
    pdb_part, chain = protein_id.rsplit("_", 1)
    if "-" in pdb_part:
        pdb_id, model_str = pdb_part.rsplit("-", 1)
        model_num = int(model_str)
    else:
        pdb_id = pdb_part
        model_num = None
    return pdb_id.lower(), model_num, chain


def cluster_member_key(protein_id: str) -> str:
    """Cluster member format used by try_combine_cluster_chains.

    '2JP1-1_A' -> '2jp1-1_A'
    '1BDT_A'   -> '1bdt_A'
    """
    pdb_id, model_num, chain = parse_protein_id(protein_id)
    if model_num is not None:
        return f"{pdb_id}-{model_num}_{chain}"
    return f"{pdb_id}_{chain}"


def seqres_key(protein_id: str) -> str:
    """Seqres lookup key (no model number).  '2JP1-1_A' -> '2jp1_A'"""
    pdb_id, _, chain = parse_protein_id(protein_id)
    return f"{pdb_id}_{chain}"


def build_protein_chain_map(members: list) -> dict:
    """
    Build a minimal protein_chain_map for exactly the chains in `members`.

    e.g. ['2jp1-1_A', '1j9o_A'] -> {'2JP1-1': {'A'}, '1J9O': {'A'}}

    Keyed by the pdb_id part uppercased (including NMR model suffix if present),
    matching how is_protein_chain() and codnas_to_pyg() look things up.
    """
    chain_map = {}
    for mem in members:
        pdb_part = mem.rsplit("_", 1)[0].upper()  # e.g. '2JP1-1' or '1BDT'
        chain    = mem.rsplit("_", 1)[1]
        chain_map.setdefault(pdb_part, set()).add(chain)
    return chain_map


def load_seqres(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading seqres clusters from {csv_path}")
    df = pd.read_csv(csv_path, usecols=["pdb_auth", "cluster30", "cluster80"])
    logger.info(f"  {len(df):,} rows, {df['cluster30'].nunique():,} cluster30 groups")
    return df


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_set(
    set_csv: Path,
    id_offset: int,
    output_dir: Path,
    seqres_df: pd.DataFrame,
    n_proc: int,
) -> tuple:
    """
    Process one dataset split (val or test).

    Returns
    -------
    rows : list of dicts with metadata per pair
    pt_ids : list of cluster_id strings for which a .pt was saved
    """
    df = pd.read_csv(set_csv, usecols=["protein1", "protein2"])
    logger.info(f"\nProcessing {len(df)} pairs from {set_csv.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PDB_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    seqres_lookup = (
        seqres_df.set_index("pdb_auth")[["cluster30", "cluster80"]]
        .to_dict("index")
    )

    rows = []

    # -----------------------------------------------------------------------
    # Stage 1: combine CIFs
    # -----------------------------------------------------------------------
    logger.info("Stage 1: combining CIFs")
    for idx, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        cluster_id = str(id_offset + idx + 1)
        mem1 = cluster_member_key(p1)
        mem2 = cluster_member_key(p2)

        k1 = seqres_key(p1)
        k2 = seqres_key(p2)
        info1 = seqres_lookup.get(k1, {})
        info2 = seqres_lookup.get(k2, {})

        if not info1:
            logger.warning(f"  {p1} (lookup={k1}) not found in seqres CSV")
        if not info2:
            logger.warning(f"  {p2} (lookup={k2}) not found in seqres CSV")

        rows.append(dict(
            cluster_id=cluster_id,
            protein1=p1, protein2=p2,
            mem1=mem1, mem2=mem2,
            cluster30_p1=info1.get("cluster30"),
            cluster30_p2=info2.get("cluster30"),
            cluster80_p1=info1.get("cluster80"),
            cluster80_p2=info2.get("cluster80"),
        ))

        save_dir = PDB_CACHE_DIR / cluster_id
        save_dir.mkdir(parents=True, exist_ok=True)

        result = try_combine_cluster_chains(
            cluster_id,
            [mem1, mem2],
            str(REDUCED_MMCIF_DIR),
            str(save_dir),
            str(PDB_SPLIT_DIR),
            {},  # pdb_chain_dict not used inside combine_cluster_chains
        )
        if result is None:
            logger.error(f"  Stage 1 FAILED: {cluster_id} ({p1} + {p2})")
        else:
            logger.info(f"  Stage 1 OK: {cluster_id} ({p1} + {p2})")

    # -----------------------------------------------------------------------
    # Stage 2: ClustalOmega alignment
    # -----------------------------------------------------------------------
    successful_cif = [
        r["cluster_id"] for r in rows
        if (PDB_CACHE_DIR / r["cluster_id"] / f"{r['cluster_id']}.cif").exists()
    ]
    logger.info(f"Stage 2: aligning {len(successful_cif)} clusters")
    align_all_pdbflex(successful_cif, PDB_CACHE_DIR, n_proc=n_proc)

    # -----------------------------------------------------------------------
    # Stage 3: codnas_to_pyg
    # -----------------------------------------------------------------------
    logger.info("Stage 3: building .pt files")
    pt_ids = []
    for r in rows:
        cid = r["cluster_id"]
        aligned_fasta = PDB_CACHE_DIR / cid / f"{cid}_aligned.fasta"
        if not aligned_fasta.exists():
            logger.warning(f"  No aligned FASTA for {cid} — skipping")
            continue

        members = [r["mem1"], r["mem2"]]
        protein_chain_map = build_protein_chain_map(members)

        pyg_data = codnas_to_pyg(cid, members, str(PDB_CACHE_DIR), protein_chain_map)
        if pyg_data is not None:
            pt_path = output_dir / f"{cid}.pt"
            torch.save(pyg_data, pt_path)
            pt_ids.append(cid)
            logger.info(f"  Saved {pt_path.name}")
        else:
            logger.warning(f"  codnas_to_pyg returned None for {cid}")

    logger.info(f"Done: {len(pt_ids)}/{len(rows)} .pt files saved to {output_dir}")
    return rows, pt_ids


def build_train_exclusion(val_rows, test_rows):
    """
    Remove cluster30 groups containing val/test proteins from train_seq80_pool.csv.
    Saves train_seq80_pool_codnas.csv.
    """
    # Collect all cluster30 IDs from val and test
    held_out_30 = set()
    for rows in [val_rows, test_rows]:
        for r in rows:
            for c30 in [r["cluster30_p1"], r["cluster30_p2"]]:
                if c30 is not None and not pd.isna(c30):
                    held_out_30.add(int(c30))

    logger.info(f"Held-out cluster30 groups: {len(held_out_30):,}")

    train_pool = pd.read_csv(TRAIN_POOL_CSV)
    before = len(train_pool)
    train_pool_excl = train_pool[~train_pool["cluster30"].isin(held_out_30)]
    after = len(train_pool_excl)

    out_path = SPLIT_OUT_DIR / "train_seq80_pool_codnas.csv"
    train_pool_excl.to_csv(out_path, index=False)
    logger.info(
        f"Train pool: {before:,} -> {after:,} rows after removing "
        f"{before - after:,} rows in {len(held_out_30)} held-out cluster30 groups"
    )
    logger.info(f"  Saved -> {out_path}")
    return held_out_30


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--n_proc", type=int, default=4)
    args = parser.parse_args()

    seqres_df = load_seqres(CLUSTER_CSV)

    val_rows, val_pts = [], []
    test_rows, test_pts = [], []

    if args.split in ("val", "both"):
        val_rows, val_pts = process_set(
            VAL_SET_CSV, VAL_ID_OFFSET, VAL_OUT_DIR, seqres_df, args.n_proc
        )
        pt_ids_df = pd.DataFrame({"pdb_auth": val_pts})
        out_val_csv = SPLIT_OUT_DIR / "val_pts.csv"
        pt_ids_df.to_csv(out_val_csv, index=False)
        logger.info(f"Val CSV -> {out_val_csv}  ({len(val_pts)} entries)")

    if args.split in ("test", "both"):
        test_rows, test_pts = process_set(
            TEST_SET_CSV, TEST_ID_OFFSET, TEST_OUT_DIR, seqres_df, args.n_proc
        )
        pt_ids_df = pd.DataFrame({"pdb_auth": test_pts})
        out_test_csv = SPLIT_OUT_DIR / "test_pts.csv"
        pt_ids_df.to_csv(out_test_csv, index=False)
        logger.info(f"Test CSV -> {out_test_csv}  ({len(test_pts)} entries)")

    # Build exclusion train pool once we have both val and test rows
    if args.split == "both" or (val_rows and test_rows):
        build_train_exclusion(val_rows, test_rows)

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
