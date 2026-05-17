#!/usr/bin/env python
"""
Build train/val/test split CSVs for DynamicProt training.

This script:
1. Loads CoDNaS val/test pairs (val_set.csv, test_set.csv)
2. Maps their PDB IDs to mmseqs2 cluster30 groups
3. Excludes those cluster30s from the training pool
4. Outputs a clean train_seq80_pool_filtered.csv

Inputs:
  - seqres_simple_clusters.csv: mmseqs2 clustering (cluster30, cluster80)
  - val_set.csv, test_set.csv: CoDNaS pairs (protein1, protein2 columns)
  - train_seq80_pool.csv: full training pool (cluster30, clus_80)
  - failed_clus80.txt: cluster80 IDs that fail during featurization (optional)

Outputs:
  - train_seq80_pool_filtered.csv: clean training pool (excludes val/test cluster30s + bad samples)

Usage:
  python build_train_val_test_split.py [--dry-run]
"""

import argparse
import os
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (from environment, see .env.example)
# ---------------------------------------------------------------------------
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parents[4]))
CLUSTERING_BASE = Path(os.environ.get("CLUSTERING_BASE", "data/clustering"))
DPROT_DATA = PROJECT_PATH / "data"
CLUSTER_CSV = CLUSTERING_BASE / "seq_clustering/output/collated/seqres_simple_clusters.csv"

VAL_SET_CSV = DPROT_DATA / "val_set.csv"
TEST_SET_CSV = DPROT_DATA / "test_set.csv"
TRAIN_POOL_CSV = DPROT_DATA / "train_seq80_pool.csv"
FAILED_CLUS80_TXT = DPROT_DATA / "failed_clus80.txt"  # Optional

OUTPUT_CSV = DPROT_DATA / "train_seq80_pool_filtered.csv"


def load_val_test_pdb_ids():
    """Extract all PDB IDs from val_set.csv and test_set.csv."""
    pdb_ids = set()

    for csv_path in [VAL_SET_CSV, TEST_SET_CSV]:
        if not csv_path.exists():
            print(f"  Warning: {csv_path.name} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        for col in ['protein1', 'protein2']:
            if col in df.columns:
                for pdb in df[col].dropna():
                    # Handle formats like "1MBY_A" or "2JP1-1_A" (NMR models)
                    pdb_id = pdb.split('_')[0].split('-')[0].lower()
                    pdb_ids.add(pdb_id)

    return pdb_ids


def load_cluster30_for_pdbs(pdb_ids):
    """Find cluster30 groups containing the given PDB IDs."""
    clust_df = pd.read_csv(CLUSTER_CSV, usecols=['pdb_auth', 'cluster30'])
    clust_df['pdb_id'] = clust_df['pdb_auth'].str.split('_').str[0].str.lower()

    matching = clust_df[clust_df['pdb_id'].isin(pdb_ids)]
    cluster30s = set(matching['cluster30'].unique())

    return cluster30s


def load_failed_clus80():
    """Load cluster80 IDs that fail during featurization."""
    if not FAILED_CLUS80_TXT.exists():
        return set()

    failed = set()
    with open(FAILED_CLUS80_TXT) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    failed.add(int(line))
                except ValueError:
                    pass
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing')
    args = parser.parse_args()

    print("=" * 60)
    print("Building train/val/test split")
    print("=" * 60)

    # Step 1: Load val/test PDB IDs
    print("\n[1] Loading val/test PDB IDs...")
    val_test_pdbs = load_val_test_pdb_ids()
    print(f"    Found {len(val_test_pdbs)} unique PDB IDs in val/test sets")

    # Step 2: Map to cluster30 groups
    print("\n[2] Mapping PDB IDs to cluster30 groups...")
    held_out_cluster30s = load_cluster30_for_pdbs(val_test_pdbs)
    print(f"    Found {len(held_out_cluster30s)} cluster30 groups to exclude")

    # Step 3: Load failed clus80 IDs
    print("\n[3] Loading failed clus80 IDs...")
    failed_clus80 = load_failed_clus80()
    print(f"    Found {len(failed_clus80)} failed clus80 IDs")

    # Step 4: Load training pool and filter
    print("\n[4] Filtering training pool...")
    train_df = pd.read_csv(TRAIN_POOL_CSV)
    before = len(train_df)
    print(f"    Original: {before:,} rows")

    # Remove held-out cluster30s
    train_df = train_df[~train_df['cluster30'].isin(held_out_cluster30s)]
    after_cluster30 = len(train_df)
    print(f"    After cluster30 exclusion: {after_cluster30:,} rows (-{before - after_cluster30:,})")

    # Remove failed clus80s
    if failed_clus80:
        train_df = train_df[~train_df['clus_80'].isin(failed_clus80)]
        after_failed = len(train_df)
        print(f"    After failed clus80 exclusion: {after_failed:,} rows (-{after_cluster30 - after_failed:,})")

    # Step 5: Save
    print("\n[5] Summary:")
    print(f"    Held-out cluster30 groups: {len(held_out_cluster30s)}")
    print(f"    Failed clus80 IDs: {len(failed_clus80)}")
    print(f"    Final training pool: {len(train_df):,} rows")
    print(f"    Unique cluster30 groups: {train_df['cluster30'].nunique():,}")

    if args.dry_run:
        print(f"\n    [DRY RUN] Would save to: {OUTPUT_CSV}")
    else:
        # Backup existing file if it exists
        if OUTPUT_CSV.exists():
            backup = OUTPUT_CSV.with_suffix('.csv.bak')
            OUTPUT_CSV.rename(backup)
            print(f"\n    Backed up existing file to: {backup.name}")

        train_df.to_csv(OUTPUT_CSV, index=False)
        print(f"    Saved to: {OUTPUT_CSV}")

    print("\nDone!")


if __name__ == "__main__":
    main()
