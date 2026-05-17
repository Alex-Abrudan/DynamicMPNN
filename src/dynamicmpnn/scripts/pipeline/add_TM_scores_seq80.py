"""
Add TM-scores from foldseek_alignments to train_pt_multi_chain/*.pt files.

Adapted from add_TM_scores_foldseek.py (which used foldseek_align/ + processed_pt/).
This script:
1. Loads .pt files from train_pt_multi_chain/
2. Reads TM-score matrices from foldseek_alignments/<cluster_id>/<cluster_id>_allvsall.tsv
3. Aligns the matrix to the pt's cluster_members (case-sensitive), capped at max 10 members
4. Adds tm_scores and tm_score_representatives to data objects
5. Saves updated files back to train_pt_multi_chain/ (in-place update)

Clusters with no matching allvsall.tsv get a fallback 1x1 dummy matrix.

Usage (interactive):
    python add_TM_scores_seq80.py

For SLURM array jobs:
    sbatch slurm_add_TM_scores_seq80
"""

import os
import torch
import multiprocessing as mp
import numpy as np
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import random
import pandas as pd


# --- Configuration (from environment, see .env.example) ---
PUBLIC_DB = Path(os.environ.get("PUBLIC_DB", "data"))
PDB_CACHE_DIR = PUBLIC_DB / "train_pt_multi_chain"
OUTPUT_DIR    = PUBLIC_DB / "train_pt_multi_chain"
FOLDSEEK_DIR  = Path(os.environ.get("FOLDSEEK_DIR", PUBLIC_DB / "foldseek_alignments"))

# Columns from Foldseek convertalis output
FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "qtmscore", "ttmscore", "lddt", "rmsd"
]
# ---------------------

worker_foldseek_dir = None


def init_worker(foldseek_dir):
    """Initialize worker with shared Foldseek directory path."""
    global worker_foldseek_dir
    worker_foldseek_dir = foldseek_dir


def parse_foldseek_tsv(tsv_file: Path) -> dict:
    """
    Parse Foldseek all-vs-all TSV output to TM-score matrix.

    Returns:
        dict with "TM_avg" (np.ndarray) and "conformers" (list of chain names)
        None if the file is empty or unreadable.
    """
    if not tsv_file.exists():
        return None

    try:
        df = pd.read_csv(
            tsv_file,
            sep="\t",
            names=FOLDSEEK_COLUMNS,
            na_values=["-", "NA", "nan", ""],
            dtype={"query": str, "target": str}
        )
    except pd.errors.EmptyDataError:
        return None

    if df.empty:
        return None

    # Clean chain names (remove .pdb extension if present)
    df["query"] = df["query"].str.replace(r"\.pdb$", "", regex=True)
    df["target"] = df["target"].str.replace(r"\.pdb$", "", regex=True)

    # Drop rows with NaN chain names
    df = df.dropna(subset=["query", "target"])

    if df.empty:
        return None

    # Get unique chains (sorted for consistency)
    chains = sorted(set(df["query"].unique()) | set(df["target"].unique()))
    n = len(chains)

    if n == 0:
        return None

    chain_to_idx = {c: i for i, c in enumerate(chains)}

    # Initialize matrices
    tm1 = np.full((n, n), np.nan)  # qtmscore
    tm2 = np.full((n, n), np.nan)  # ttmscore

    # Fill matrices
    for _, row in df.iterrows():
        q, t = row["query"], row["target"]
        if q in chain_to_idx and t in chain_to_idx:
            i, j = chain_to_idx[q], chain_to_idx[t]
            if pd.notna(row["qtmscore"]):
                tm1[i, j] = row["qtmscore"]
            if pd.notna(row["ttmscore"]):
                tm2[i, j] = row["ttmscore"]

    # Average TM-scores and symmetrise
    # Suppress "Mean of empty slice" when both tm1[i,j] and tm2[i,j] are NaN
    # (pair absent from foldseek TSV in both directions — NaN filled by mean imputation below)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        tm_avg = np.nanmean([tm1, tm2], axis=0)

    np.fill_diagonal(tm_avg, 1.0)
    tm_avg = np.fmax(tm_avg, tm_avg.T)

    return {
        "TM_avg": tm_avg,
        "conformers": chains  # case-preserved
    }


def apply_fallback_and_save(data_object, filename, cluster_id, reason):
    """
    Fallback: keep ≤10 random members, set tm_scores=None so get_entries
    falls back to random sampling rather than crashing on a mismatched matrix.
    """
    try:
        original_members = data_object.cluster_members

        if len(original_members) > 10:
            selected_members = list(np.random.choice(original_members, 10, replace=False))
        else:
            selected_members = original_members

        if len(selected_members) == 0:
            return {'status': 'skipped_empty', 'cluster_id': cluster_id, 'reason': reason}

        selected_pdb_ids = {name.split('_')[0].upper() for name in selected_members}
        data_object.pyg_dict = {k: v for k, v in data_object.pyg_dict.items()
                                if k.upper() in selected_pdb_ids}
        data_object.cluster_members = selected_members

        # None signals get_entries to fall back to random pair sampling
        data_object.tm_scores = None
        data_object.tm_score_representatives = []

        output_path = OUTPUT_DIR / filename
        torch.save(data_object, output_path)

        logger.warning(f"Cluster '{cluster_id}': {reason}. Saved fallback ({len(selected_members)} members).")
        return {'status': 'saved_fallback', 'cluster_id': cluster_id, 'reason': reason}

    except Exception as e:
        logger.error(f"Cluster '{cluster_id}': Failed during fallback save: {e}")
        return {'status': 'error', 'cluster_id': cluster_id, 'reason': str(e)}


def update_single_file(filename):
    """Process a single .pt file and add TM-scores from foldseek_alignments."""
    global worker_foldseek_dir

    cluster_id = os.path.splitext(filename)[0]
    filepath = PDB_CACHE_DIR / filename
    output_path = OUTPUT_DIR / filename

    try:
        data_object = torch.load(filepath, map_location='cpu')
        pt_members = data_object.cluster_members

        # --- 1. Find Foldseek output ---
        cluster_dir = Path(worker_foldseek_dir) / cluster_id
        tsv_file = cluster_dir / f"{cluster_id}_allvsall.tsv"

        if not tsv_file.exists():
            return apply_fallback_and_save(data_object, filename, cluster_id, "Foldseek TSV not found")

        # --- 2. Parse TM-score data ---
        tm_info = parse_foldseek_tsv(tsv_file)

        if tm_info is None:
            return apply_fallback_and_save(data_object, filename, cluster_id, "Failed to parse Foldseek TSV")

        tm_matrix = tm_info["TM_avg"]
        foldseek_chains = tm_info["conformers"]

        # --- 3. Build chain mapping (CASE-SENSITIVE — auth chain IDs preserve case) ---
        # foldseek chains: lowercase pdb_id + original-case auth chain (e.g. "1abc_H", "1abc_h")
        # pt_members: same format from pdb_auth column of seqres_simple_clusters.csv
        chain_order_map = {name.strip(): i for i, name in enumerate(foldseek_chains)}

        # Find intersection: members present in BOTH .pt file AND Foldseek output
        aligned_members = []
        missing_from_foldseek = []
        for m in pt_members:
            m_clean = m.strip()
            if m_clean in chain_order_map:
                aligned_members.append((m, chain_order_map[m_clean]))
            else:
                missing_from_foldseek.append(m_clean)

        extra_in_foldseek = [c for c in foldseek_chains if c not in set(pt_members)]

        num_aligned = len(aligned_members)

        if num_aligned == 0:
            return apply_fallback_and_save(data_object, filename, cluster_id, "No aligned members found")

        # --- 4. Selection Logic (Max 10) ---
        if num_aligned > 10:
            selected_subset = random.sample(aligned_members, 10)
        else:
            selected_subset = aligned_members

        final_names = [x[0] for x in selected_subset]
        final_indices = [x[1] for x in selected_subset]

        # --- 5. Extract Sub-Matrix ---
        sub_matrix = tm_matrix[np.ix_(final_indices, final_indices)]

        # Ensure symmetry
        if not np.allclose(sub_matrix, sub_matrix.T, rtol=1e-5, atol=1e-5, equal_nan=True):
            sub_matrix = np.nanmax([sub_matrix, sub_matrix.T], axis=0)

        # Fill NaN with average off-diagonal value
        mask = ~np.eye(sub_matrix.shape[0], dtype=bool)
        valid_scores = sub_matrix[mask][~np.isnan(sub_matrix[mask])]
        if len(valid_scores) > 0:
            avg_tm = np.mean(valid_scores)
            sub_matrix = np.where(np.isnan(sub_matrix), avg_tm, sub_matrix)
        else:
            sub_matrix = np.where(np.isnan(sub_matrix), 0.9, sub_matrix)

        # Ensure diagonal is 1.0
        np.fill_diagonal(sub_matrix, 1.0)

        # --- 6. Validate dimensions ---
        expected_size = len(final_names)
        if sub_matrix.shape != (expected_size, expected_size):
            return apply_fallback_and_save(
                data_object, filename, cluster_id,
                f"Matrix size mismatch: got {sub_matrix.shape}, expected ({expected_size}, {expected_size})"
            )

        off_diag_mask = ~np.eye(expected_size, dtype=bool)
        if expected_size > 1 and np.sum(~np.isnan(sub_matrix[off_diag_mask])) == 0:
            return apply_fallback_and_save(
                data_object, filename, cluster_id,
                f"No valid TM-scores in matrix ({expected_size} members)"
            )

        # --- 7. Update Data Object ---
        data_object.cluster_members = final_names

        final_pdb_ids = {name.split('_')[0].upper() for name in final_names}
        data_object.pyg_dict = {
            k: v for k, v in data_object.pyg_dict.items()
            if k.upper() in final_pdb_ids
        }

        data_object.tm_score_representatives = final_names
        data_object.tm_scores = torch.from_numpy(sub_matrix.astype(np.float32))

        # --- 8. Save ---
        torch.save(data_object, output_path)

        return {
            'status': 'updated_filtered',
            'cluster_id': cluster_id,
            'missing_from_foldseek': missing_from_foldseek,   # pt members absent from foldseek
            'extra_in_foldseek': extra_in_foldseek,           # foldseek chains not in pt
        }

    except Exception as e:
        logger.error(f"Failed to process file {filename}: {e}")
        return {'status': 'error', 'cluster_id': cluster_id, 'reason': str(e)}


def run_multiprocessing_update():
    """Main function to run the TM-score addition with multiprocessing."""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task {task_id}/{num_tasks}: Starting TM-score addition from foldseek_alignments...")
    logger.info(f"  Input:    {PDB_CACHE_DIR}")
    logger.info(f"  Output:   {OUTPUT_DIR}")
    logger.info(f"  Foldseek: {FOLDSEEK_DIR}")

    if not PDB_CACHE_DIR.exists():
        logger.error(f"Source directory does not exist: {PDB_CACHE_DIR}")
        return
    if not FOLDSEEK_DIR.exists():
        logger.error(f"Foldseek directory does not exist: {FOLDSEEK_DIR}")
        return

    all_pt_files = sorted(f for f in os.listdir(PDB_CACHE_DIR) if f.endswith('.pt'))
    my_pt_files = all_pt_files[task_id::num_tasks]

    logger.info(f"Task {task_id}: Processing {len(my_pt_files)} / {len(all_pt_files)} files.")

    if not my_pt_files:
        return

    logger.info(f"Task {task_id}: Starting pool with {cpus_per_task} workers.")

    results = []
    with mp.Pool(processes=cpus_per_task, initializer=init_worker, initargs=(str(FOLDSEEK_DIR),)) as pool:
        for result in tqdm(pool.imap_unordered(update_single_file, my_pt_files),
                           total=len(my_pt_files), desc=f"Task {task_id}"):
            results.append(result)

    # Bucket results by status
    updated   = [r for r in results if r['status'] == 'updated_filtered']
    fallbacks = [r for r in results if r['status'] == 'saved_fallback']
    errors    = [r for r in results if r['status'] == 'error']
    skipped   = [r for r in results if r['status'] == 'skipped_empty']

    # Within updated: which had mismatches
    pt_missing = [r for r in updated if r['missing_from_foldseek']]
    extra      = [r for r in updated if r['extra_in_foldseek']]

    print("\n" + "=" * 50)
    logger.info(f"Task {task_id} Summary")
    print("=" * 50)
    logger.success(f"Updated successfully:    {len(updated)}")
    logger.info(   f"Fallback (tm_scores=None): {len(fallbacks)}")
    logger.error(  f"Errors:                  {len(errors)}")
    logger.info(   f"Skipped (empty):         {len(skipped)}")

    if pt_missing:
        total_missing_chains = sum(len(r['missing_from_foldseek']) for r in pt_missing)
        logger.warning(f"\nClusters with pt members absent from foldseek ({len(pt_missing)} clusters, {total_missing_chains} chains dropped):")
        for r in pt_missing:
            logger.warning(f"  cluster {r['cluster_id']}: dropped {r['missing_from_foldseek']}")

    if extra:
        total_extra_chains = sum(len(r['extra_in_foldseek']) for r in extra)
        logger.info(f"\nClusters with extra foldseek chains not in pt ({len(extra)} clusters, {total_extra_chains} chains ignored):")
        for r in extra:
            logger.info(f"  cluster {r['cluster_id']}: ignored {r['extra_in_foldseek']}")

    if fallbacks:
        logger.warning(f"\nFallback cluster IDs and reasons:")
        for r in fallbacks:
            logger.warning(f"  cluster {r['cluster_id']}: {r['reason']}")

    if errors:
        logger.error(f"\nError cluster IDs and reasons:")
        for r in errors:
            logger.error(f"  cluster {r['cluster_id']}: {r['reason']}")

    print("=" * 50)


if __name__ == "__main__":
    run_multiprocessing_update()
