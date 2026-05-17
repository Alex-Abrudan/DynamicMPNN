##############################################################################
# This script performs the following:
# 1. Loads the CoDNaS sequence clusters together with tehir properties.
# 2. Downloads the MMCIF files for the PDB IDs in the clusters (all chains).
# 3. For each CoDNaS cluster, gather all protein chains contained in the MMCIFs of all
# members into a single MMCIF with multiple models:
#    - Each model is named after a PDB ID and contains all the ATOM lines for all the chains
#      in the PDB, with the member chains placed first.
#    - For NMR PDBs, each each NMR model is saved as a separate model in the MMCIF file with
#      the model number appended to the PDB ID as "-n".
# 4. In the process, the script also saves the sequences of the cluster member chains
# in FASTA format to later be aligned; in another fasta file, the sequences of the rest of
# the chains.
#
# Difference from download_codnas.py:
# - This script saves the combined structures in MMCIF format instead of PDB format.
# - This script saves reduced MMCIF files: it only keeps the chains within 8 Å of the member chains.
##############################################################################

import itertools
import os
import sys
import json
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict

import aiofiles
import asyncio
import hydra
import pandas as pd
import numpy as np
from aiohttp import ClientSession
from Bio import PDB
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import copy

warnings.simplefilter("ignore", PDBConstructionWarning)  # Ignore annoying discontinous chain warnings

from dynamicmpnn import constants
from dynamicmpnn.datamodules.utils import load_codnas_cluster_data
from dynamicmpnn.types import STANDARD_AMINO_ACID_MAPPING_3_TO_1


async def get_pdb_stats(session, pdb_id, chain_id=None):
    """
    Function to interface with the PDBFlex API to get PDBStats (essentially all the PDBFlex info)
    for a given PDB ID.
    """
    base_url = "https://pdbflex.org/php/api/PDBStats.php"
    url = f"{base_url}?pdbID={pdb_id}"
    if chain_id:
        url += f"&chainID={chain_id}"

    async with session.get(url, timeout=None) as response:
        if response.status == 200:
            data = await response.json()
            return data
        else:
            raise Exception(f"Failed to get PDBStats for PDB ID {pdb_id}")


async def augment_cluster_table(cluster_df):
    """
    Function to add list of all PDBFlex cluster members to the raw cluster table from (pdbflex.org/clusters.html;
    cluster_id, rmsd, n_members)
    """

    async def _get_pdb_stats(session, cluster_id, out_dict):
        # NOTE: Annoyingly, there is an issue in the PDBFLex API where querying the cluster_id is not enough to get
        # all the cluster members. We need to query the entire pdb_id and check that each chain is in the cluster.
        result = await get_pdb_stats(session, cluster_id[:-1])
        out_dict[cluster_id] = {cluster_id}
        for res in result:
            if res["parentClusterID"] == cluster_id:
                out_dict[cluster_id].update(set(res["otherClusterMembers"]))
                # out_dict[cluster_id].add(res["pdbID"] + res["chainID"])

    cluster_ids = cluster_df.cluster_id.tolist()
    thread_dict = {}
    tasks = []
    async with ClientSession() as session:
        for cluster_id in cluster_ids:
            tasks.append(_get_pdb_stats(session, cluster_id, thread_dict))
        await asyncio.gather(*tasks)
    return thread_dict


async def download_mmcif_file(session, pdb_id, mmcif_dir, n_retries=4):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    output_file = os.path.join(mmcif_dir, f"{pdb_id}.cif")
    for tries in range(n_retries):
        try:
            async with session.get(url, timeout=None) as response:
                if response.status == 200:
                    async with aiofiles.open(output_file, "wb") as f:
                        async for chunk in response.content.iter_any():
                            await f.write(chunk)
                    return
                    # logger.info(f"Successfully downloaded {pdb_id}.cif")
                else:
                    raise Exception(f"Failed to download {pdb_id}.cif")
        except Exception as e:
            if tries == n_retries - 1:
                logger.error(f"Failed to download {pdb_id}.cif after {n_retries} retries: {e}")
                raise e
            logger.info(f"Transient error when downloading {pdb_id}.cif: {e}. Retrying...")
            await asyncio.sleep(2**tries + 1)


async def download_mmcif_files(cfg: DictConfig, pdb_ids):
    """
    Download MMCIF files from RCSB for the given PDB IDs.
    Args:
        cfg: Hydra config object.
        pdb_ids: List of PDB IDs to download.
    """

    download_dir = Path(cfg.datamodule.download_dir)
    mmcif_dir = Path(os.path.join(download_dir, "mmcif_files"))
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(mmcif_dir, exist_ok=True)

    fixed_pdb_ids = []
    for pdb_id in pdb_ids:
        if "-" in pdb_id:
            fixed_pdb_ids.append(pdb_id.split("-")[0])  # Just take the base PDB ID
        else:
            fixed_pdb_ids.append(pdb_id)
    
    fixed_pdb_ids = list(set(fixed_pdb_ids))

    async with ClientSession() as session:
        tasks = []
        for pdb_id in fixed_pdb_ids:
            tasks.append(download_mmcif_file(session, pdb_id, mmcif_dir))

        await tqdm_asyncio.gather(*tasks, desc="Downloading PDBFlex MMCIF files")

def combine_cluster_chains(cluster_id, cluster_members, mmcif_dir, output_dir, pdb_split_dir, pdb_chain_dict):
    """
    Combines all chains from PDBFlex cluster members into a single PDB file with multiple models.
    Function preserves the original chain IDs and order of the cluster members.

    Additionally, the function removes HETATMs from the PDB files and saves the sequences in FASTA format.

    Args:
        cluster_id: ID of the cluster.
        cluster_members: List of cluster members in the format "{pdb_id}{chain}".
        mmcif_dir: Directory containing the MMCIF files.
        output_dir: Directory to save the combined PDB file.
    """

    parser = PDB.MMCIFParser()
    io = PDB.PDBIO()
    sequences = []
    sequences_non_members = []
    combined_structure = PDB.Structure.Structure(cluster_id)
    cluster_chains = set()
    pdb_to_cluster_chains = defaultdict(list)
    NMR_pdbs = set()

    for clus_mem in cluster_members:
        if "-" in clus_mem:

            NMR_pdbs.add(clus_mem.split("_")[0])
        cluster_chains.add(clus_mem)

    for member in cluster_chains:
        pdb = member.split("_")[0]
        chain = member.split("_")[1]
        pdb_to_cluster_chains[pdb].append(chain)

    for pdb_id in pdb_to_cluster_chains.keys():
        try:
            member_chain_ids = pdb_to_cluster_chains[pdb_id]
            true_pdb_id, nmr_frame = pdb_id.split("-") if "-" in pdb_id else (pdb_id, None)
            pdb_id_lower = true_pdb_id.lower()
            subdir_name = pdb_id_lower[1:3]
            filename = f"{true_pdb_id.upper()}.cif"
            mmcif_path = os.path.join(mmcif_dir, subdir_name, filename)
            if not os.path.exists(mmcif_path):
                # Fallback: try flat mmcif_files/ layout (same parent dir as reduced_mmcifs/)
                flat_path = os.path.join(os.path.dirname(mmcif_dir), "mmcif_files", f"{pdb_id_lower}.cif")
                if os.path.exists(flat_path):
                    logger.debug(f"Using flat mmcif fallback for {true_pdb_id}: {flat_path}")
                    mmcif_path = flat_path
                else:
                    raise FileNotFoundError(f"MMCIF file {mmcif_path} not found.")

            structure = parser.get_structure(pdb_id, mmcif_path)

            if len(structure) > 1 and pdb_id in NMR_pdbs:
                for model_idx, model in enumerate(structure):
                    # We now only save the nmr frame that corresponds to the cluster member
                    if int(nmr_frame) != (model_idx + 1):
                        continue
                    new_model = PDB.Model.Model(f"{pdb_id}")
                    for member_chain_id in member_chain_ids:
                        chain_member = model[member_chain_id]
                        new_chain = PDB.Chain.Chain(member_chain_id)  # Save as asym_label_id
                        sequences.extend([f">{pdb_id}_{member_chain_id}", ""])
                        for residue in list(chain_member):
                            if residue.id[0] == " ":  # Remove HETATMs
                                new_chain.add(residue)
                                sequences[-1] += STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(residue.resname, "X")

                        new_model.add(new_chain)

                    all_chains_in_file = [c.id for c in model]

                    for chain_id in all_chains_in_file:
                        chain = model[chain_id]

                        if chain_id in member_chain_ids:
                            # Already added in the first loop above — skip to avoid "defined twice"
                            continue

                        else:
                            # --- Process as NON-MEMBER (contacting) chain ---
                            new_chain = PDB.Chain.Chain(chain_id)
                            if model_idx == 0: # Only save non-member seq for first model
                                sequences_non_members.extend([f">{pdb_id}_{chain_id}", ""])
                            for residue in list(chain):
                                if residue.id[0] == " ":
                                    new_chain.add(residue)
                                    if model_idx == 0:
                                        sequences_non_members[-1] += STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(residue.resname, "X")
                            new_model.add(new_chain)

                    combined_structure.add(new_model)

            elif len(structure) > 1 and pdb_id not in NMR_pdbs:
                logger.warning(
                    f"MMCIF file {pdb_id} contains multiple models and is NOT measured with NMR. Only the first model will be used."
                )

            else:
                model = structure[0]
                new_model = PDB.Model.Model(pdb_id)
                
                # 1. Get ALL chains that are ACTUALLY in this reduced file
                # This is the list we will loop over. e.g., ['A', 'B']
                all_chains_in_file = [c.id for c in model]

                # 2. Use ONE loop over the chains from the file
                for chain_id in all_chains_in_file:
                    # Get the actual chain object
                    chain = model[chain_id]
                    
                    # --- START of logic from your SECOND loop ---
                    # Check if this is the special single chain to save
                    if f"{pdb_id}_{chain_id}" == cluster_id and not os.path.isfile(
                        os.path.join(pdb_split_dir, f"{cluster_id}_single_chain.cif") 
                    ):
                        single_chain_copy = copy.deepcopy(chain)
                        new_structure = PDB.Structure.Structure(f"{cluster_id}_single_chain")
                        new_model_single = PDB.Model.Model(0)
                        new_structure.add(new_model_single)
                        new_model_single.add(single_chain_copy)

                        io_single = PDB.MMCIFIO()  # Use MMCIFIO
                        io_single.set_structure(new_structure)
                        io_single.save(os.path.join(pdb_split_dir, f"{cluster_id}_single_chain.cif"))
                    # --- END of logic from your SECOND loop ---

                    
                    # --- Now, process as MEMBER or NON-MEMBER ---
                    if chain_id in member_chain_ids:
                        # --- This is logic from your FIRST loop ---
                        # This chain is a CLUSTER MEMBER (e.g., 'A')
                        new_chain = PDB.Chain.Chain(chain_id)
                        sequences.extend([f">{pdb_id}_{chain_id}", ""])
                        for residue in list(chain):
                            if residue.id[0] == " ":  # Remove HETATMs
                                new_chain.add(residue)
                                sequences[-1] += STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(residue.resname, "X")
                        new_model.add(new_chain)
                    
                    else:
                        # --- This is logic from your SECOND loop ---
                        # This chain is a NON-MEMBER (neighbor, e.g., 'B')
                        new_chain = PDB.Chain.Chain(chain_id)
                        sequences_non_members.extend([f">{pdb_id}_{chain_id}", ""])
                        for residue in list(chain):
                            if residue.id[0] == " ":  # Remove HETATMs
                                new_chain.add(residue)
                                sequences_non_members[-1] += STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(
                                    residue.resname, "X"
                                )
                        new_model.add(new_chain)

                combined_structure.add(new_model)

        except FileNotFoundError as e:
            # This is an expected failure (filtered PDB), so just warn and continue
            logger.warning(f"Skipping PDB {pdb_id} for cluster {cluster_id}: File not found (likely filtered).")
            continue  # <-- This is the key: it skips to the next PDB
    
    if not sequences:
        raise FileNotFoundError(f"Cluster {cluster_id} is empty: all member PDBs were filtered or missing.")

    io = PDB.MMCIFIO() # Use MMCIFIO, no use_model_flag needed
    io.set_structure(combined_structure)
    io.save(os.path.join(output_dir, f"{cluster_id}.cif"))

    with open(os.path.join(output_dir, f"{cluster_id}.fasta"), "w") as f:
        f.write("\n".join(sequences))

    with open(os.path.join(output_dir, f"{cluster_id}_non_members.fasta"), "w") as f:
        f.write("\n".join(sequences_non_members))


def try_combine_cluster_chains(cluster_id, cluster_members, mmcif_dir, output_dir, pdb_split_dir, pdb_chain_dict):
    """
    Wrapper function to catch exceptions when combining cluster chains when using multiprocessing.
    """
    try:
        combine_cluster_chains(cluster_id, cluster_members, mmcif_dir, output_dir, pdb_split_dir, pdb_chain_dict)
        return cluster_id
    except FileNotFoundError as e:
        # This is an expected failure (filtered PDB), so just warn
        logger.warning(f"Skipping cluster {cluster_id}: A required mmCIF file was not found (likely filtered): {e}")
        return None  # <-- FAILURE: Return None
    except Exception as e:
        logger.error(f"Failed to combine cluster chains for {cluster_id}: {e}")
        return None


def multiprocess_combine_cluster_chains(download_dir, cluster_dict, pdb_chain_dict, n_proc=None):
    """
    Apply the combine_cluster_chains function to each row in the cluster_df using multiprocessing.
    """
    mmcif_dir = Path(download_dir) / "reduced_mmcifs"
    pdb_split_dir = download_dir.parent / "data" /"codnas" / "pdb_splitting"
    cache_dir = download_dir / "pdb_cache"
    os.makedirs(pdb_split_dir, exist_ok=True)

    if n_proc is None:
        n_proc = len(os.sched_getaffinity(0))
        logger.info(f"Number of processes not specified, using all available cores: {n_proc}")
    pbar = tqdm(total=len(cluster_dict.keys()), desc="Combining cluster chains")

    def update_pbar(*args):
        pbar.update()

    successful_ids = []

    with mp.get_context("spawn").Pool(n_proc) as pool:
        results = []
        for key, val in cluster_dict.items():
            cluster_id = key
            cluster_members = val
            save_dir = os.path.join(cache_dir, cluster_id)
            os.makedirs(save_dir, exist_ok=True)
            results.append(
                pool.apply_async(
                    try_combine_cluster_chains,
                    args=(cluster_id, cluster_members, mmcif_dir, save_dir, pdb_split_dir, pdb_chain_dict),
                    callback=update_pbar,
                )
            )

        pool.close()
        for r in results:
            result_id = r.get()  # This will raise an exception if the function raised one
            if result_id:
                successful_ids.append(result_id)

        pool.join()
    
    pbar.close() # Make sure the progress bar closes

    all_ids = set(cluster_dict.keys())
    failed_ids = list(all_ids - set(successful_ids))
    
    return successful_ids, failed_ids


def save_chains_to_pdb(mmcif_path, chains, output_dir):
    """
    Load a MMCIF file and save specified chains to individual PDB files.
    Args:
        mmcif_path: Path to the MMCIF file.
        chains: List of chain IDs to save.
        output_dir: Directory to save the PDB files.
    """
    parser = PDB.MMCIFParser()
    io = PDB.PDBIO()

    mmcif_id = os.path.basename(mmcif_path).split(".")[0]
    structure = parser.get_structure(mmcif_id, mmcif_path)
    if len(structure) > 1:
        logger.warning(f"MMCIF file {mmcif_id} contains multiple models, only the first model will be used.")
    model = structure[0]
    for chain in chains:
        chain_struc = model[chain]
        io.set_structure(chain_struc)
        io.save(os.path.join(output_dir, f"{mmcif_id}{chain}.pdb"))


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH / "dataset"),
    config_name="pdb",
)
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    # TODO: Clean up this hacky/ugly code
    #cluster_dict, pdb_ids = load_codnas_cluster_data(cfg, full_info = False) # Replace this into main branch
    with open(Path(cfg.datamodule.data_index_dir) / "clust_dict.json") as f:
        cluster_dict = json.load(f)

    pdb_ids = cluster_dict.values()
    pdb_ids = list(itertools.chain(*pdb_ids))
    pdb_ids = [pdb_id.split("_")[0] for pdb_id in pdb_ids]

    asyncio.run(download_mmcif_files(cfg, pdb_ids))
    with open(os.path.join(cfg.datamodule.data_index_dir, "pdb_protein_chain_mappings.json")) as f:  
        pdb_chain_dict = json.load(f)
    multiprocess_combine_cluster_chains(Path(cfg.datamodule.download_dir), cluster_dict, pdb_chain_dict)


if __name__ == "__main__":
    _main()
