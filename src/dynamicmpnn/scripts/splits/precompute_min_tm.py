"""
Annotate train_seq80_pool_filtered.csv with per-cluster80 min TM-score and
per-cluster30 min TM-score, for use with curriculum learning.

Reads:  dynamicmpnn/data/train_seq80_pool_filtered.csv
Writes: dynamicmpnn/data/train_seq80_pool_filtered_mintm.csv

New columns:
  min_tm       – minimum off-diagonal TM score among members of each clus_80
                 (NaN if tm_scores=None; filled with median before saving)
  clus30_min_tm – min(min_tm) across all clus_80 within the same cluster30

Usage
-----
python precompute_min_tm.py [--n_proc N]
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from joblib import Parallel, delayed

# Paths (from environment, see .env.example)
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parents[4]))
PT_DIR  = PUBLIC_DB / "train_pt_multi_chain"
DATA_DIR = PROJECT_PATH / "data"
POOL_IN  = DATA_DIR / "train_seq80_pool_filtered.csv"
POOL_OUT = DATA_DIR / "train_seq80_pool_filtered_mintm.csv"


def _min_tm_for_cluster(clus_80: int) -> float:
    pt_path = PT_DIR / f"{clus_80}.pt"
    if not pt_path.exists():
        return float("nan")
    try:
        pt = torch.load(pt_path, map_location="cpu")
        tm = getattr(pt, "tm_scores", None)
        if tm is None or tm.numel() <= 1:
            return float("nan")
        m = tm.numpy().copy().astype(float)
        np.fill_diagonal(m, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice")
            return float(np.nanmin(m))
    except Exception as e:
        logger.warning(f"  Failed to load {clus_80}.pt: {e}")
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_proc", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(POOL_IN)
    logger.info(f"Pool: {len(df):,} rows, {df['cluster30'].nunique():,} cluster30 groups, "
                f"{df['clus_80'].nunique():,} unique clus_80")

    unique_ids = df["clus_80"].unique()
    logger.info(f"Computing min_tm for {len(unique_ids):,} unique cluster80 IDs "
                f"using {args.n_proc} workers...")

    results = Parallel(n_jobs=args.n_proc, verbose=5)(
        delayed(_min_tm_for_cluster)(cid) for cid in unique_ids
    )

    min_tm_map = dict(zip(unique_ids, results))
    df["min_tm"] = df["clus_80"].map(min_tm_map)

    n_nan = df["min_tm"].isna().sum()
    median_val = df["min_tm"].median()
    logger.info(f"NaN min_tm: {n_nan:,} / {len(df):,} rows — filling with median {median_val:.4f}")
    df["min_tm"] = df["min_tm"].fillna(median_val)

    # Per-cluster30 difficulty ceiling: min of all clus_80 min_tms in the group
    df["clus30_min_tm"] = df.groupby("cluster30")["min_tm"].transform("min")

    df.to_csv(POOL_OUT, index=False)
    logger.info(f"Saved -> {POOL_OUT}")
    logger.info(f"min_tm range: [{df['min_tm'].min():.3f}, {df['min_tm'].max():.3f}]  "
                f"median={df['min_tm'].median():.3f}")
    logger.info(f"clus30_min_tm range: [{df['clus30_min_tm'].min():.3f}, "
                f"{df['clus30_min_tm'].max():.3f}]")


if __name__ == "__main__":
    main()
