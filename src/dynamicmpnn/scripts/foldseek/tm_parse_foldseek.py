"""
Parser for Foldseek output files to create TM-score matrices.

This module parses Foldseek all-vs-all alignment and clustering outputs
into the same format as the USalign-based tm_parse.py, ensuring compatibility
with downstream analysis.

Output format: {cluster_id: {"TM_avg": np.ndarray, "conformers": [str, ...]}}
"""

from constants import FOLDSEEK_CLUS_DIR
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FOLDSEEK OUTPUT PARSING
# ============================================================================

# Column names for Foldseek convertalis output
FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "qtmscore", "ttmscore", "lddt", "rmsd"
]


def parse_foldseek_allvsall(
    tsv_file: str,
    chains: list = None,
    fill_diagonal: bool = True,
) -> dict:
    """
    Parse Foldseek all-vs-all output TSV to TM-score matrices.
    
    Args:
        tsv_file: Path to Foldseek convertalis output
        chains: Optional list of chain IDs to include (in order)
        fill_diagonal: Fill diagonal with 1.0 (self-alignments)
    
    Returns:
        dict with keys: "TM_avg", "TM1" (qtmscore), "TM2" (ttmscore), 
                       "RMSD", "fident", "chains"
    """
    tsv_path = Path(tsv_file)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Foldseek output not found: {tsv_file}")
    
    # Read TSV
    try:
        df = pd.read_csv(
            tsv_file, 
            sep="\t", 
            names=FOLDSEEK_COLUMNS,
            na_values=["-", "NA", "nan", ""],
            dtype={"query": str, "target": str}
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty Foldseek output: {tsv_file}")
    
    if df.empty:
        raise ValueError(f"No alignments in Foldseek output: {tsv_file}")
    
    # Clean chain names (remove .pdb extension if present)
    df["query"] = df["query"].str.replace(r"\.pdb$", "", regex=True)
    df["target"] = df["target"].str.replace(r"\.pdb$", "", regex=True)
    
    # Remove rows with NaN chain names (from malformed lines)
    df = df.dropna(subset=["query", "target"])
    
    if df.empty:
        raise ValueError(f"No valid alignments in Foldseek output: {tsv_file}")
    
    # Get unique chains
    if chains is None:
        chains = sorted(set(df["query"].unique()) | set(df["target"].unique()))
    
    n = len(chains)
    if n == 0:
        raise ValueError(f"No chains found in {tsv_file}")
    
    chain_to_idx = {c: i for i, c in enumerate(chains)}
    
    # Initialize matrices
    matrices = {
        "TM1": np.full((n, n), np.nan),      # Query TM-score (qtmscore)
        "TM2": np.full((n, n), np.nan),      # Target TM-score (ttmscore)
        "TM_aln": np.full((n, n), np.nan),   # Alignment TM-score
        "RMSD": np.full((n, n), np.nan),
        "fident": np.full((n, n), np.nan),
        "lddt": np.full((n, n), np.nan),
    }
    
    # Fill matrices
    for _, row in df.iterrows():
        q, t = row["query"], row["target"]
        if q not in chain_to_idx or t not in chain_to_idx:
            continue
        
        i, j = chain_to_idx[q], chain_to_idx[t]
        
        # Handle potential NaN values
        for col, key in [("qtmscore", "TM1"), ("ttmscore", "TM2"), 
                         ("alntmscore", "TM_aln"), ("rmsd", "RMSD"),
                         ("fident", "fident"), ("lddt", "lddt")]:
            val = row.get(col)
            if pd.notna(val):
                try:
                    matrices[key][i, j] = float(val)
                except (ValueError, TypeError):
                    pass
    
    # Compute average TM-score (symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        matrices["TM_avg"] = np.nanmean([matrices["TM1"], matrices["TM2"]], axis=0)
    
    # Symmetrize matrices (take max of (i,j) and (j,i))
    for key in ["TM_avg", "TM1", "TM2", "TM_aln", "RMSD", "fident", "lddt"]:
        mat = matrices[key]
        # For RMSD, take min; for others take max
        if key == "RMSD":
            matrices[key] = np.fmin(mat, mat.T)
        else:
            matrices[key] = np.fmax(mat, mat.T)
    
    # Fill diagonal
    if fill_diagonal:
        for key in ["TM_avg", "TM1", "TM2", "TM_aln", "fident", "lddt"]:
            np.fill_diagonal(matrices[key], 1.0)
        np.fill_diagonal(matrices["RMSD"], 0.0)
    
    matrices["chains"] = chains
    
    # Validate
    if np.all(np.isnan(matrices["TM_avg"])):
        raise ValueError(f"All TM-scores are NaN in {tsv_file}")
    
    return matrices


def parse_foldseek_clusters(cluster_file: str) -> dict:
    """
    Parse Foldseek cluster TSV output.
    
    Args:
        cluster_file: Path to *_cluster.tsv file
    
    Returns:
        dict with keys: 'representatives', 'clusters' (dict: rep -> [members])
    """
    cluster_path = Path(cluster_file)
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
    
    try:
        df = pd.read_csv(cluster_file, sep="\t", names=["representative", "member"])
    except pd.errors.EmptyDataError:
        return {"representatives": [], "clusters": {}}
    
    # Clean names
    df["representative"] = df["representative"].str.replace(r"\.pdb$", "", regex=True)
    df["member"] = df["member"].str.replace(r"\.pdb$", "", regex=True)
    
    # Build clusters
    clusters = df.groupby("representative")["member"].apply(list).to_dict()
    representatives = list(clusters.keys())
    
    return {
        "representatives": representatives,
        "clusters": clusters
    }


def load_chains_list(cluster_dir: str) -> list:
    """Load list of chains from chains.txt file."""
    chains_file = Path(cluster_dir) / "chains.txt"
    if chains_file.exists():
        with open(chains_file) as f:
            return [line.strip() for line in f if line.strip()]
    return None


# ============================================================================
# SAMPLE/CLUSTER LOADING (COMPATIBLE WITH ORIGINAL tm_parse.py)
# ============================================================================

def get_sample_clusters_foldseek(sample_dir: str, max_conformers: int = None) -> dict:
    """
    Load TM-score data and conformers for a single cluster (Foldseek format).
    
    This function mimics the output of tm_parse.get_sample_clusters() for
    compatibility with downstream code.
    
    Args:
        sample_dir: Path to cluster directory
        max_conformers: Maximum number of conformers to return
    
    Returns:
        dict with keys: "TM_avg", "conformers", and optional other matrices
    """
    sample_dir = Path(sample_dir)
    clus_id = sample_dir.stem
    
    # Find the all-vs-all TSV file
    allvsall_file = sample_dir / f"{clus_id}_allvsall.tsv"
    if not allvsall_file.exists():
        raise FileNotFoundError(f"All-vs-all file not found: {allvsall_file}")
    
    # Load chain list
    chains = load_chains_list(str(sample_dir))
    
    # Parse all-vs-all results
    tm_data = parse_foldseek_allvsall(str(allvsall_file), chains=chains)
    
    # Determine conformers (representatives if clustered, else all chains)
    cluster_file = sample_dir / f"{clus_id}_clustered_cluster.tsv"
    if cluster_file.exists():
        cluster_data = parse_foldseek_clusters(str(cluster_file))
        conformers = cluster_data["representatives"]
    else:
        conformers = tm_data["chains"]
    
    # Apply max_conformers limit
    if max_conformers and len(conformers) > max_conformers:
        # Select subset - prefer diverse selection based on TM-score
        conformers = select_diverse_conformers(tm_data, conformers, max_conformers)
    
    # Get indices of conformers in the full matrix
    chain_to_idx = {c: i for i, c in enumerate(tm_data["chains"])}
    conformer_indices = [chain_to_idx[c] for c in conformers if c in chain_to_idx]
    
    # Extract sub-matrix for conformers only
    if len(conformer_indices) < len(tm_data["chains"]):
        idx = np.array(conformer_indices)
        tm_avg_subset = tm_data["TM_avg"][np.ix_(idx, idx)]
    else:
        tm_avg_subset = tm_data["TM_avg"]
    
    return {
        "TM_avg": tm_avg_subset,
        "conformers": conformers,
        "TM1": tm_data.get("TM1"),
        "TM2": tm_data.get("TM2"),
        "RMSD": tm_data.get("RMSD"),
    }


def select_diverse_conformers(tm_data: dict, conformers: list, n_select: int) -> list:
    """
    Select diverse conformers based on TM-score dissimilarity.
    
    Uses greedy selection: start with one, iteratively add the one
    most different from current selection.
    """
    if len(conformers) <= n_select:
        return conformers
    
    chains = tm_data["chains"]
    chain_to_idx = {c: i for i, c in enumerate(chains)}
    
    # Get indices of conformers
    valid_conformers = [c for c in conformers if c in chain_to_idx]
    if len(valid_conformers) <= n_select:
        return valid_conformers
    
    indices = [chain_to_idx[c] for c in valid_conformers]
    tm_matrix = tm_data["TM_avg"][np.ix_(indices, indices)]
    
    # Greedy diverse selection
    selected = [0]  # Start with first
    remaining = set(range(1, len(valid_conformers)))
    
    while len(selected) < n_select and remaining:
        # Find conformer with minimum max TM-score to selected set
        min_max_tm = float('inf')
        best_idx = None
        
        for idx in remaining:
            max_tm_to_selected = max(tm_matrix[idx, s] for s in selected 
                                     if not np.isnan(tm_matrix[idx, s]))
            if max_tm_to_selected < min_max_tm:
                min_max_tm = max_tm_to_selected
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break
    
    return [valid_conformers[i] for i in selected]


def select_random_conformers(tm_data: dict, n_select: int = 10) -> dict:
    """
    Randomly select n conformers and reduce TM_avg matrix accordingly.
    
    Compatible with original tm_parse.select_random_conformers.
    """
    conformers = tm_data["conformers"]
    n_total = len(conformers)
    
    if n_total <= n_select:
        return tm_data
    
    selected_idx = np.random.choice(n_total, n_select, replace=False)
    selected_idx = np.sort(selected_idx)
    
    tm_data = tm_data.copy()
    tm_data["conformers"] = [conformers[i] for i in selected_idx]
    tm_data["TM_avg"] = tm_data["TM_avg"][np.ix_(selected_idx, selected_idx)]
    
    return tm_data


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def _get_sample_clusters_safe(sample_dir: str, shared_dict: dict, max_conformers: int = 10):
    """Safe wrapper for multiprocessing."""
    sample_name = Path(sample_dir).stem
    try:
        allvsall_file = Path(sample_dir) / f"{sample_name}_allvsall.tsv"
        if allvsall_file.exists():
            data = get_sample_clusters_foldseek(sample_dir, max_conformers=max_conformers)
            shared_dict[sample_name] = data
        else:
            logger.debug(f"Skipping {sample_name}: no allvsall file")
    except Exception as e:
        logger.warning(f"Error processing {sample_dir}: {e}")
        shared_dict[sample_name] = None


def mp_get_sample_clusters_foldseek(
    samples_parent_dir: str, 
    n_tasks: int = 64,
    max_conformers: int = 10
) -> dict:
    """
    Load TM-score data from all cluster directories in parallel.
    
    Args:
        samples_parent_dir: Parent directory containing cluster subdirectories
        n_tasks: Number of parallel workers
        max_conformers: Maximum conformers per cluster
    
    Returns:
        dict: {cluster_id: {"TM_avg": np.ndarray, "conformers": [...]}, ...}
    """
    samples_parent_dir = Path(samples_parent_dir)
    sample_dirs = [d for d in samples_parent_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Processing {len(sample_dirs)} cluster directories with {n_tasks} workers")
    
    manager = mp.Manager()
    shared_dict = manager.dict()
    
    with mp.Pool(n_tasks) as pool:
        args = [(str(d), shared_dict, max_conformers) for d in sample_dirs]
        pool.starmap(_get_sample_clusters_safe, args)
    
    return dict(shared_dict)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Foldseek results to TM-score pickle")
    parser.add_argument("--input-dir", type=str, default=str(FOLDSEEK_CLUS_DIR),
                        help="Directory containing cluster subdirectories")
    parser.add_argument("--output", type=str, default="all_tm_data_foldseek.pkl",
                        help="Output pickle file")
    parser.add_argument("--n-proc", type=int, default=64,
                        help="Number of parallel processes")
    parser.add_argument("--max-conformers", type=int, default=10,
                        help="Maximum conformers per cluster")
    args = parser.parse_args()
    
    logger.info(f"Loading data from {args.input_dir}")
    all_data = mp_get_sample_clusters_foldseek(
        args.input_dir, 
        n_tasks=args.n_proc,
        max_conformers=args.max_conformers
    )
    
    # Filter out None values and keep only essential data
    all_data_out = {k: v for k, v in all_data.items() if v is not None}
    all_data_out = {
        k: {"TM_avg": v["TM_avg"], "conformers": v["conformers"]} 
        for k, v in all_data_out.items()
    }
    
    logger.info(f"Successfully loaded {len(all_data_out)} clusters")
    
    # Summary statistics
    clus_sizes = {k: len(v["conformers"]) for k, v in all_data_out.items()}
    logger.info(f"Cluster sizes: min={min(clus_sizes.values())}, "
                f"max={max(clus_sizes.values())}, "
                f"mean={np.mean(list(clus_sizes.values())):.1f}")
    
    # Save
    pd.to_pickle(all_data_out, args.output)
    logger.info(f"Saved to {args.output}")
