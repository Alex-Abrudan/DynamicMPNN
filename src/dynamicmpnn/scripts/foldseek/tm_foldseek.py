"""
Foldseek-based TM-score calculation and structure clustering.

This module replaces USalign (TMalign/qTMclust) with Foldseek for:
1. All-vs-all TM-score calculations between structures
2. Greedy structure clustering based on TM-score thresholds

Foldseek is faster and more robust than USalign for large-scale analysis.
"""

from constants import (
    FOLDSEEK_BIN,
    FOLDSEEK_CLUS_DIR,
    FOLDSEEK_CLUS_DIR_MMSEQS2,
    MMSEQS2_CLUSTERS_CSV,
    CHAINS_DIR,
    MMCIF_DIR,
    REDUCED_MMCIF_DIR,
    CODNAS_MMCIF_DIR,
)
from subprocess import run, CalledProcessError, TimeoutExpired
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, Structure, Model, PDBParser
import multiprocessing as mp
import shutil
import tempfile
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FOLDSEEK WRAPPER FUNCTIONS
# ============================================================================

def run_foldseek_allvsall(
    input_dir: str,
    output_prefix: str,
    tmp_dir: str = None,
    alignment_type: int = 1,  # 1 = TM-align scoring
    threads: int = 4,
    timeout: int = 3600,
    use_gpu: bool = False,
) -> dict:
    """
    Run Foldseek all-vs-all alignment to get pairwise TM-scores.
    
    Args:
        input_dir: Directory containing structure files (PDB/mmCIF)
        output_prefix: Prefix for output files
        tmp_dir: Temporary directory for Foldseek
        alignment_type: 0=3Di, 1=TM-align, 2=3Di+AA
        threads: Number of threads
        timeout: Timeout in seconds
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        dict with keys: 'success', 'output_file', 'error'
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="foldseek_")
    
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    
    # Create database from structures
    db_path = f"{output_prefix}_db"
    
    try:
        # Step 1: Create database
        cmd_createdb = [
            str(FOLDSEEK_BIN), "createdb",
            input_dir,
            db_path,
            "--threads", str(threads),
        ]
        result = run(cmd_createdb, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"createdb failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Step 2: All-vs-all search
        result_db = f"{output_prefix}_result"
        cmd_search = [
            str(FOLDSEEK_BIN), "search",
            db_path, db_path, result_db, tmp_dir,
            "--alignment-type", str(alignment_type),
            "-e", "inf",  # No E-value cutoff (get all pairs)
            "--max-seqs", "10000",  # Allow many hits
            "-s", "9.5",  # High sensitivity
            "-a",  # Enable alignment backtracing (required for TM-scores in convertalis)
            "--threads", str(threads),
        ]
        if use_gpu:
            cmd_search.extend(["--gpu", "1"])
        
        result = run(cmd_search, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"search failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Step 3: Convert to TSV with TM-scores
        output_tsv = f"{output_prefix}_allvsall.tsv"
        cmd_convert = [
            str(FOLDSEEK_BIN), "convertalis",
            db_path, db_path, result_db, output_tsv,
            "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,qtmscore,ttmscore,lddt,rmsd",
            "--threads", str(threads),
        ]
        result = run(cmd_convert, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"convertalis failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        return {"success": True, "output_file": output_tsv, "error": None}
        
    except TimeoutExpired:
        logger.error(f"Foldseek timed out after {timeout}s")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        logger.error(f"Foldseek error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup temp files (keep output)
        for suffix in ["_db", "_db.index", "_db.lookup", "_db_h", "_db_h.index", 
                       "_db_ss", "_db_ss.index", "_db_ca", "_db_ca.index",
                       "_result", "_result.index", "_result.dbtype"]:
            p = Path(f"{output_prefix}{suffix}")
            if p.exists():
                p.unlink()


def run_foldseek_cluster(
    input_dir: str,
    output_prefix: str,
    tmp_dir: str = None,
    tm_threshold: float = 0.9,
    alignment_type: int = 1,
    cluster_mode: int = 0,  # Set-cover (greedy)
    threads: int = 4,
    timeout: int = 3600,
    use_gpu: bool = False,
) -> dict:
    """
    Run Foldseek clustering based on TM-score threshold.
    
    Uses easy-cluster with --tmscore-threshold for proper TM-score based clustering.
    
    Args:
        input_dir: Directory containing structure files
        output_prefix: Prefix for output files
        tmp_dir: Temporary directory
        tm_threshold: TM-score threshold for clustering (0.0-1.0)
        alignment_type: 0=3Di, 1=TM-align, 2=3Di+AA
        cluster_mode: 0=set-cover (greedy), 1=connected-component, 2=greedy-by-length
        threads: Number of threads
        timeout: Timeout in seconds
        use_gpu: Use GPU acceleration
    
    Returns:
        dict with keys: 'success', 'cluster_file', 'rep_file', 'error'
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="foldseek_clust_")
    
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [
            str(FOLDSEEK_BIN), "easy-cluster",
            input_dir,
            output_prefix,
            tmp_dir,
            "--alignment-type", str(alignment_type),
            "-a",  # Enable alignment backtracing (required for TM-scores)
            "--tmscore-threshold", str(tm_threshold),  # TM-score threshold for clustering
            "--cluster-mode", str(cluster_mode),
            "-e", "inf",  # No E-value cutoff
            "-s", "9.5",  # High sensitivity
            "--threads", str(threads),
        ]
        if use_gpu:
            cmd.extend(["--gpu", "1"])
        
        result = run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            logger.error(f"easy-cluster failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        cluster_file = f"{output_prefix}_cluster.tsv"
        rep_file = f"{output_prefix}_rep_seq.fasta"
        
        return {
            "success": True,
            "cluster_file": cluster_file,
            "rep_file": rep_file,
            "error": None
        }
        
    except TimeoutExpired:
        logger.error(f"Foldseek clustering timed out after {timeout}s")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        logger.error(f"Foldseek clustering error: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# CHAIN EXTRACTION (IMPROVED)
# ============================================================================

def find_mmcif_file(pdb_id: str, mmcif_dir: Path = REDUCED_MMCIF_DIR) -> Path:
    """Find mmCIF file, checking reduced_mmcifs (divided layout) first, then fallback dirs.

    reduced_mmcifs uses RCSB divided layout: reduced_mmcifs/<mid2>/<PDBID>.cif
    where <mid2> = pdb_id.lower()[1:3]  (e.g. '4ye4' -> 'ye/4YE4.cif')
    Large assemblies absent from reduced_mmcifs fall back to mmcif_files/.
    """
    mid2 = pdb_id.lower()[1:3]
    candidates = [
        REDUCED_MMCIF_DIR / mid2 / f"{pdb_id.upper()}.cif",
        REDUCED_MMCIF_DIR / mid2 / f"{pdb_id.lower()}.cif",
        MMCIF_DIR / f"{pdb_id.lower()}.cif",
        MMCIF_DIR / f"{pdb_id.upper()}.cif",
        CODNAS_MMCIF_DIR / f"{pdb_id.lower()}.cif",
        CODNAS_MMCIF_DIR / f"{pdb_id.upper()}.cif",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_chain_to_pdb(
    pdb_id: str,
    auth_chain: str,
    output_path: str,
    mmcif_dir: Path = REDUCED_MMCIF_DIR,
    quiet: bool = True
) -> bool:
    """
    Extract a single chain from mmCIF to PDB format.
    
    Args:
        pdb_id: PDB identifier
        auth_chain: Author chain ID to extract
        output_path: Where to save the PDB file
        mmcif_dir: Directory containing mmCIF files
        quiet: Suppress parser warnings
    
    Returns:
        True if successful, False otherwise
    """
    mmcif_path = find_mmcif_file(pdb_id, mmcif_dir)
    if mmcif_path is None:
        logger.warning(f"mmCIF file not found for {pdb_id}")
        return False
    
    try:
        parser = MMCIFParser(QUIET=quiet)
        structure = parser.get_structure(pdb_id, mmcif_path)
        
        # Handle multi-model structures
        model = structure[0]
        
        # Try to find the chain (case-sensitive: auth chain IDs 'H' and 'h' are distinct)
        chain = None
        for c in model.get_chains():
            if c.id == auth_chain:
                chain = c
                break
        
        if chain is None:
            logger.warning(f"Chain {auth_chain} not found in {pdb_id}")
            return False
        
        # Check if chain has atoms
        if len(list(chain.get_atoms())) == 0:
            logger.warning(f"Chain {auth_chain} in {pdb_id} has no atoms")
            return False
        
        # Create new structure with just this chain
        new_structure = Structure.Structure(f"{pdb_id}_{auth_chain}")
        new_model = Model.Model(0)
        new_chain = chain.copy()
        new_chain.id = "A"  # Standardize chain ID
        new_model.add(new_chain)
        new_structure.add(new_model)
        
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {pdb_id}_{auth_chain}: {e}")
        return False


# Global variables for multiprocessing worker
_mp_output_dir = None
_mp_mmcif_dir = None

def _extract_one_worker(pdb_auth: str) -> Optional[str]:
    """Worker function for parallel chain extraction (must be at module level for pickling)."""
    global _mp_output_dir, _mp_mmcif_dir
    pdb_id, auth_chain = pdb_auth.split("_")
    output_path = Path(_mp_output_dir) / f"{pdb_auth}.pdb"
    if extract_chain_to_pdb(pdb_id, auth_chain, str(output_path), _mp_mmcif_dir):
        return pdb_auth
    return None

def _init_extract_worker(output_dir: str, mmcif_dir: Path):
    """Initialize worker process with shared directories."""
    global _mp_output_dir, _mp_mmcif_dir
    _mp_output_dir = output_dir
    _mp_mmcif_dir = mmcif_dir


def extract_chains_to_directory(
    pdb_auth_chains: list,
    output_dir: str,
    mmcif_dir: Path = REDUCED_MMCIF_DIR,
    n_workers: int = 4
) -> list:
    """
    Extract multiple chains to individual PDB files in a directory.
    
    This is more robust than multi-model PDB for Foldseek.
    
    Args:
        pdb_auth_chains: List of "pdb_chain" identifiers
        output_dir: Directory to save PDB files
        mmcif_dir: Source mmCIF directory
        n_workers: Number of parallel workers
    
    Returns:
        List of successfully extracted chain IDs
    """
    global _mp_output_dir, _mp_mmcif_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    
    if n_workers > 1:
        # Set globals for workers
        _mp_output_dir = str(output_dir)
        _mp_mmcif_dir = mmcif_dir
        with mp.Pool(n_workers, initializer=_init_extract_worker, 
                     initargs=(str(output_dir), mmcif_dir)) as pool:
            results = pool.map(_extract_one_worker, pdb_auth_chains)
        successful = [r for r in results if r is not None]
    else:
        for pdb_auth in pdb_auth_chains:
            pdb_id, auth_chain = pdb_auth.split("_")
            output_path = output_dir / f"{pdb_auth}.pdb"
            if extract_chain_to_pdb(pdb_id, auth_chain, str(output_path), mmcif_dir):
                successful.append(pdb_auth)
    
    logger.info(f"Extracted {len(successful)}/{len(pdb_auth_chains)} chains")
    return successful


# ============================================================================
# CLUSTER PROCESSING
# ============================================================================

def load_chains() -> tuple:
    """Load protein chains grouped by 80% sequence identity clusters (old RCSB entity clusters)."""
    chains_df = pd.read_csv(CHAINS_DIR / "all_protein_chains_clusters.csv")
    chains_df = chains_df[chains_df.is_protein & ~chains_df.clus_80.isna()].reset_index(drop=True)
    chains_df["pdb_auth"] = chains_df.pdb_id + "_" + chains_df.author_id
    chains_df["clus_80"] = chains_df["clus_80"].astype(int)
    clusters = chains_df.groupby("clus_80")["pdb_auth"].apply(list).to_dict()
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    return chains_df, clusters


def load_chains_mmseqs2() -> tuple:
    """Load protein chains grouped by mmseqs2 seqres 80% sequence identity clusters.

    Uses seqres_simple_clusters.csv built by build_cluster_csv.py.
    pdb_auth format: lowercase_pdb_id + "_" + author_chain (e.g. "1abc_A").
    cluster80: integer cluster ID assigned by mmseqs2.
    """
    df = pd.read_csv(MMSEQS2_CLUSTERS_CSV, usecols=["pdb_auth", "cluster80"])
    df["cluster80"] = df["cluster80"].astype(int)
    clusters = df.groupby("cluster80")["pdb_auth"].apply(list).to_dict()
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    return df, clusters


def process_cluster_foldseek(
    clus_id: int,
    clusters: dict,
    output_dir: str,
    mmcif_dir: Path = REDUCED_MMCIF_DIR,
    n_conf: int = 10,
    tm_cluster_thresh: float = 0.9,
    threads: int = 4,
    use_gpu: bool = False,
    cleanup_structures: bool = True,
) -> dict:
    """
    Process a single sequence identity cluster using Foldseek.
    
    Workflow:
    1. Extract all chains to individual PDB files
    2. Run Foldseek all-vs-all alignment
    3. If cluster size > n_conf, run Foldseek clustering
    4. Parse results to TM-score matrix
    
    Args:
        clus_id: Cluster identifier
        clusters: Dict mapping cluster IDs to lists of pdb_auth chains
        output_dir: Output directory
        mmcif_dir: Source mmCIF directory
        n_conf: Max number of conformers to keep
        tm_cluster_thresh: TM-score threshold for sub-clustering
        threads: Number of threads for Foldseek
        use_gpu: Use GPU acceleration
        cleanup_structures: Delete structure files after processing to save disk/inodes
    
    Returns:
        Dict with processing results and status
    """
    pdb_auth_chains = clusters[clus_id]
    cluster_dir = Path(output_dir) / str(clus_id)
    structures_dir = cluster_dir / "structures"
    tmp_dir = cluster_dir / "tmp"
    
    result = {
        "clus_id": clus_id,
        "n_input": len(pdb_auth_chains),
        "n_extracted": 0,
        "success": False,
        "error": None,
    }
    
    try:
        cluster_dir.mkdir(parents=True, exist_ok=True)
        structures_dir.mkdir(exist_ok=True)
        tmp_dir.mkdir(exist_ok=True)
        
        # Step 1: Extract chains
        extracted = extract_chains_to_directory(
            pdb_auth_chains, 
            str(structures_dir), 
            mmcif_dir,
            n_workers=min(4, threads)
        )
        result["n_extracted"] = len(extracted)
        
        if len(extracted) < 2:
            result["error"] = f"Only {len(extracted)} chains extracted, need at least 2"
            logger.warning(f"Cluster {clus_id}: {result['error']}")
            return result
        
        # Save extracted chain list
        with open(cluster_dir / "chains.txt", "w") as f:
            f.write("\n".join(extracted))
        
        # Step 2: Run all-vs-all alignment
        output_prefix = str(cluster_dir / str(clus_id))
        allvsall_result = run_foldseek_allvsall(
            str(structures_dir),
            output_prefix,
            str(tmp_dir),
            threads=threads,
            use_gpu=use_gpu,
        )
        
        if not allvsall_result["success"]:
            result["error"] = f"All-vs-all failed: {allvsall_result['error']}"
            logger.error(f"Cluster {clus_id}: {result['error']}")
            return result
        
        result["allvsall_file"] = allvsall_result["output_file"]
        
        # Step 3: If large cluster, run clustering to select representatives
        if len(extracted) > n_conf:
            cluster_result = run_foldseek_cluster(
                str(structures_dir),
                str(cluster_dir / f"{clus_id}_clustered"),
                str(tmp_dir),
                tm_threshold=tm_cluster_thresh,
                threads=threads,
                use_gpu=use_gpu,
            )
            
            if cluster_result["success"]:
                result["cluster_file"] = cluster_result["cluster_file"]
                result["rep_file"] = cluster_result.get("rep_file")
            else:
                logger.warning(f"Cluster {clus_id}: Sub-clustering failed ({cluster_result.get('error')}), using all {len(extracted)} structures")
        
        result["success"] = True
        logger.info(f"Cluster {clus_id}: Successfully processed {len(extracted)} structures")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Cluster {clus_id}: Exception - {e}")
    
    finally:
        # Cleanup tmp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # Cleanup structures dir after processing to save disk space/inodes
        if cleanup_structures and structures_dir.exists() and result["success"]:
            shutil.rmtree(structures_dir, ignore_errors=True)
            logger.debug(f"Cluster {clus_id}: Cleaned up structures directory")
    
    return result


def process_cluster_wrapper(args):
    """Wrapper for multiprocessing."""
    return process_cluster_foldseek(*args)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    n_to_proc = 500
    n_workers = 4  # Parallel cluster processing
    threads_per_cluster = 4
    use_gpu = False  # Set True if GPU available

    array_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if array_idx is None:
        logger.warning("SLURM_ARRAY_TASK_ID not set, using 0")
        array_idx = 0
    else:
        array_idx = int(array_idx)

    # Use mmseqs2 seqres clusters and dedicated output directory
    chains_df, clusters = load_chains_mmseqs2()
    output_dir = FOLDSEEK_CLUS_DIR_MMSEQS2
    logger.info(f"Processing array index {array_idx} with {len(clusters)} clusters total")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load already processed clusters
    processed_file = output_dir / "processed_clusters.txt"
    if processed_file.exists():
        with open(processed_file) as f:
            processed_clus = {int(line.strip()) for line in f if line.strip().isdigit()}
        logger.info(f"Already processed {len(processed_clus)} clusters")
    else:
        processed_clus = set()
    
    # Filter out already processed
    remaining_clusters = {k: v for k, v in clusters.items() if k not in processed_clus}
    logger.info(f"{len(remaining_clusters)} clusters remaining to process")
    
    # Interleaving for load balancing
    # Use SLURM_ARRAY_TASK_COUNT (fixed) instead of recalculating from remaining clusters
    # to avoid interleaving shift when jobs run at different times
    cluster_keys = list(remaining_clusters.keys())
    total_jobs = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 123))  # fallback to 123 for compatibility
    indices = [i for i in range(len(cluster_keys)) if i % total_jobs == array_idx]

    clusters_to_process = [cluster_keys[i] for i in indices if i < len(cluster_keys)]
    logger.info(f"This job will process {len(clusters_to_process)} clusters")
    
    # Process clusters
    results = []
    for clus_id in clusters_to_process:
        result = process_cluster_foldseek(
            clus_id,
            remaining_clusters,
            str(output_dir),
            mmcif_dir=REDUCED_MMCIF_DIR,
            threads=threads_per_cluster,
            use_gpu=use_gpu,
        )
        results.append(result)

        # Log progress
        if result["success"]:
            with open(processed_file, "a") as f:
                f.write(f"{clus_id}\n")

    # Summary
    n_success = sum(1 for r in results if r["success"])
    logger.info(f"Completed: {n_success}/{len(results)} clusters processed successfully")

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f"results_job_{array_idx}.csv", index=False)
