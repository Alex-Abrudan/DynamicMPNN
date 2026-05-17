import os
import csv
from typing import Dict
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from biopandas.mmcif import PandasMmcif
from biopandas.mmcif.pandas_mmcif import mmcif_col_types

from dynamicmpnn.types import (
    STANDARD_AMINO_ACIDS,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    STANDARD_AMINO_ACID_MAPPING_3_TO_1,
    FILL_VALUE,
    PROTEIN_ATOMS,
    BASE_AMINO_ACIDS,
    SEQ_SIMILARITY_THRESHOLD,
    HOMOMER_NEGATIVE,
)
from typing import List, Optional, Union

from torch_geometric.data import Data
import torch
from graphein.protein.graphs import (
    deprotonate_structure,
    filter_hetatms,
    read_pdb_to_dataframe,
    remove_insertions,
    select_chains,
    sort_dataframe,
)
from graphein.protein.tensor.io import (
    protein_df_to_tensor,
    get_sequence,
    get_residue_id,
    residue_type_tensor,
    protein_df_to_chain_tensor,
)
from os.path import join as pjoin


def get_sequence_similarity(seq1: str, seq2: str) -> float:
    """
    Calculate similarity percentage between two amino acid sequences using global alignment.

    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence

    Returns:
        float: Similarity percentage (0-100)
    """
    # Scoring parameters
    MATCH_SCORE = 1
    MISMATCH_SCORE = -1
    GAP_PENALTY = -1

    # Initialize the scoring matrix
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = np.zeros((rows, cols))

    # Initialize first row and column
    for i in range(rows):
        score_matrix[i][0] = i * GAP_PENALTY
    for j in range(cols):
        score_matrix[0][j] = j * GAP_PENALTY

    # Fill the scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):
            match = score_matrix[i - 1][j - 1] + (MATCH_SCORE if seq1[i - 1] == seq2[j - 1] else MISMATCH_SCORE)
            delete = score_matrix[i - 1][j] + GAP_PENALTY
            insert = score_matrix[i][j - 1] + GAP_PENALTY
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to get the alignment
    aligned1, aligned2 = [], []
    i, j = rows - 1, cols - 1

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and score_matrix[i][j]
            == score_matrix[i - 1][j - 1] + (MATCH_SCORE if seq1[i - 1] == seq2[j - 1] else MISMATCH_SCORE)
        ):
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i][j] == score_matrix[i - 1][j] + GAP_PENALTY:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    # Reverse the alignments
    aligned1 = "".join(reversed(aligned1))
    aligned2 = "".join(reversed(aligned2))

    # Calculate similarity percentage
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b)
    total_length = len(aligned1)
    similarity = (matches / total_length) * 100

    return similarity


def filter_alt_confs(df):
    """
    Filter alternative conformations by keeping the first alt_loc for each residue
    while preserving atoms with no alt_loc.

    Parameters:
    df (pandas.DataFrame): DataFrame containing PDB atom records

    Returns:
    pandas.DataFrame: Filtered DataFrame keeping one complete conformation set
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Group by residue
    for (chain_id, res_num), residue_df in df.groupby(["chain_id", "residue_number"]):
        # Get unique alt_locs excluding empty strings
        alt_locs = sorted(residue_df["alt_loc"][residue_df["alt_loc"].str.strip() != ""].unique())
        if alt_locs == ["B"]:
            pass
        if len(alt_locs) > 0:

            # Keep atoms that have no alt_loc (they belong to all conformations)
            no_alt_mask = residue_df["alt_loc"].str.strip() == ""

            # For atoms with alt_loc, keep only the first conformation
            first_alt = alt_locs[0]
            alt_mask = residue_df["alt_loc"] == first_alt

            # Combined mask for this residue
            residue_mask = no_alt_mask | alt_mask

            # Update the result DataFrame
            result_df.loc[residue_df[~residue_mask].index, "alt_loc"] = None

    # Final filtering: keep rows where alt_loc is either the empty string or None
    return result_df[~result_df["alt_loc"].isna()]


def protein_to_pyg(
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    chain_selection: Union[str, List[str]] = "all",
    deprotonate: bool = True,
    keep_insertions: bool = True,
    keep_hets: List[str] = [],
    model_index: int = 1,
    atom_types: List[str] = PROTEIN_ATOMS,
    remove_nonstandard: bool = True,
    store_het: bool = False,
) -> Data:
    """
    Parses a protein (from either: a PDB code, PDB file or a UniProt ID
    (via AF2 database) to a PyTorch Geometric ``Data`` object.


    .. code-block:: python

        import graphein.protein.tensor as gpt

        # From PDB code
        gpt.io.protein_to_pyg(pdb_code="3eiy")

        # From PDB Path
        gpt.io.protein_to_pyg(path="3eiy.pdb")

        # From MMTF Path
        gpt.io.protein_to_pyg(path="3eiy.mmtf")

        # From UniProt ID
        gpt.io.protein_to_pyg(uniprot_id="Q5VSL9")


    :param path: Path to PDB or MMTF file. Default is ``None``.
    :type path: Union[str, os.PathLike]
    :param pdb_code: PDB accesion code. Default is ``None``.
    :type pdb_code: str
    :param uniprot_id: UniProt ID. Default is ``None``.
    :type uniprot_id: str
    :param chain_selection: Selection of chains to include (e.g.
        ``["A", "C", "AB"]``) or ``"all"``. Default is ``"all"``.
    :type chain_selection: Union[str, List[str]]
    :param deprotonate: Whether or not to remove Hydrogens. Default is ``True``.
    :type deprotonate: bool
    :param keep_insertions: Whether or not to keep insertions. Default is
        ``True``.
    :type keep_insertions: bool
    :param keep_hets: List of heteroatoms to include. E.g. ``["HOH"]``.
    :type keep_hets: List[str]
    :param model_index: Index of model in models containing multiple structures.
    :type model_index: int
    :param atom_types: List of atom types to select. Default is:
        :const:`graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :type atom_types: List[str]
    :param remove_nonstandard: Whether or not to remove non-standard residues.
        Default is ``True``.
    :type remove_nonstandard: bool
    :param store_het: Whether or not to store heteroatoms in the ``Data``
        object. Default is ``False``.
    :type store_het: bool
    :returns: ``Data`` object with attributes: ``x`` (AtomTensor), ``residues``
        (list of 3-letter residue codes), id (ID of protein), residue_id (E.g.
        ``"A:SER:1"``), residue_type (torch.Tensor), ``chains`` (torch.Tensor).
    :rtype: torch_geometric.data.Data
    """

    # Get ID
    if path is not None:
        id = (
            os.path.splitext(path)[0].split("/")[-1] + "_" + "".join(chain_selection)
            if chain_selection != "all"
            else os.path.splitext(path)[0].split("/")[-1]
        )
    elif pdb_code is not None:
        id = pdb_code + "_" + "".join(chain_selection) if chain_selection != "all" else pdb_code
    elif uniprot_id is not None:
        id = uniprot_id + "_" + "".join(chain_selection) if chain_selection != "all" else uniprot_id
    else:
        id = None

    if df is None:
        df = read_pdb_to_dataframe(
            path=path,
            pdb_code=pdb_code,
            uniprot_id=uniprot_id,
            model_index=model_index,
        )

    df_residues = len(df.groupby(["chain_id", "residue_number"]))
    logger.info(f"Before filters: {df_residues}")
    if chain_selection != "all":
        if isinstance(chain_selection, str):
            chain_selection = [chain_selection]
        df = select_chains(df, chain_selection)

    # Filter out alternative conformations
    if (df["alt_loc"] != "").any():
        df = filter_alt_confs(df)

    if deprotonate:
        df = deprotonate_structure(df)
    if not keep_insertions:
        df = remove_insertions(df)
    # Remove hetatms
    hets = filter_hetatms(df, keep_hets=keep_hets)

    if store_het:
        hetatms = df.loc[df.record_name == "HETATM"]
        all_hets = list(set(hetatms.residue_name))
        het_coords = {}
        for het in all_hets:
            het_coords[het] = torch.tensor(
                hetatms.loc[hetatms.residue_name == het][["x_coord", "y_coord", "z_coord"]].values
            )

    df = df.loc[df.record_name == "ATOM"]
    df_residues = len(df.groupby(["chain_id", "residue_number"]))
    logger.info(f"After ATOM filter: {df_residues}")

    if remove_nonstandard:
        df = df.loc[df.residue_name.isin(STANDARD_AMINO_ACID_MAPPING_1_TO_3.values())]
        df_residues = len(df.groupby(["chain_id", "residue_number"]))
        logger.info(f"After non-standard filter: {df_residues}")

    df = pd.concat([df] + hets)
    df = sort_dataframe(df)

    df["residue_id"] = df["chain_id"] + ":" + df["residue_name"] + ":" + df["residue_number"].astype(str)
    if keep_insertions:
        df["residue_id"] = df.residue_id + ":" + df.insertion

    chain_mapping = {chain: int(idx) for idx, chain in enumerate(df["chain_id"].unique())}

    out = Data(
        coords=protein_df_to_tensor(df, atoms_to_keep=atom_types),
        residues=get_sequence(
            df,
            chains=chain_selection,
            insertions=keep_insertions,
            list_of_three=True,
        ),
        id=id,
        residue_id=get_residue_id(df),
        residue_type=residue_type_tensor(df),
        chains=protein_df_to_chain_tensor(df),
        chain_mapping=chain_mapping,
    )
    if store_het:
        out.hetatms = [het_coords]
    logger.info(f"Protein PyG Data object created with length {out.coords.shape}")
    return out


def load_codnas_cluster_data(cfg: DictConfig, full_info: bool = False):
    """
    Simple helper function to load summary of PDBFlex dataset (pdbflex_clusters.csv) and extract the unique PDB IDs
    that need to be downloaded from RCSB.
    Args:
        cfg: Hydra config object.
    Returns:
        cluster_df: DataFrame containing the PDBFlex cluster data.
        pdb_ids: Unique PDB IDs in the PDBFlex dataset.
    """
    cluster_df_path = pjoin(cfg.datamodule.data_index_dir, "codnas-2025.csv")

    df_codnas = pd.read_csv(cluster_df_path)

    if full_info:
        raise NotImplementedError("Full info not implemented yet")
        """
        cluster_members = []
        for clust_key, clust_val in cluster_dict.items():
            cluster_members.extend(clust_val["pdb1"])
        logger.info(f"Total number of unique cluster members (chains): {len(cluster_members)}")
        pdb_ids = set(
            [member.split("-")[0] if "-" in member else member.split("_")[0] for member in cluster_members]
        )
        return cluster_dict, pdb_ids
        """
    else:
        codnas_chains = []
        key_val_dict = {}
        for clust_key, clust_val2 in zip(df_codnas["pdb1"], df_codnas["pdb2"]):
            key_val_dict[clust_key] = [clust_key, clust_val2]
            codnas_chains.extend(key_val_dict[clust_key])
        pdb_ids = set(
            [member.split("_")[0] for member in codnas_chains]
        )
        logger.info(f"Total number of unique PDB IDs: {len(pdb_ids)}")
        return key_val_dict, pdb_ids


def read_fasta(filename):
    """
    Read a fasta file and return a dictionary with the sequence names as keys and the sequences as values.
    """

    seqs = {}
    with open(filename) as f:
        content = f.read().splitlines()
    for line in content:
        if line.startswith(">"):
            seqname = line[1:]
            seqs[seqname] = ""
        else:
            seqs[seqname] += line
    return seqs


def seq_diff(seqs):
    """
    Get the indices where there is a gap token in any of the sequences.

    Args:
        seqs: Dictionary of sequences.

    """
    diff = set()
    for seq in seqs.values():
        diff.update([i for i, aa in enumerate(seq) if aa == "-"])
    diff = list(diff)
    diff.sort()
    return diff


def load_and_split_mmcif(cif_path):
    """
    Load a mmCIF file with multiple models using PandasMmcif, split it into a dictionary
    of DataFrames (one for each model), and rename columns to PDB-style.
    """
    # --- Fix 1: Remove problematic keys from the biopandas schema ---
    with open(cif_path, 'r') as f:
        lines = f.readlines()
        
    # 1. Find all ATOM keys that are *actually* in the file
    # We look for lines starting with '_atom_site.'
    actual_atom_site_keys = set()
    for line in lines:
        line_s = line.strip()
        if line_s.startswith('_atom_site.'):
            # Get the key name after the dot (e.g., 'id' from '_atom_site.id')
            key_name = line_s.split(maxsplit=1)[0].split('.')[1]
            actual_atom_site_keys.add(key_name)
    
    # 2. Get all keys biopandas *expects*
    expected_keys = set(mmcif_col_types.keys())
    
    # 3. Find the keys that are expected but missing from the file
    missing_keys = expected_keys - actual_atom_site_keys
    
    # Temporarily remove keys not present in this file from biopandas schema;
    # restore after parsing to avoid permanently mutating the global for subsequent calls.
    saved_col_types = {k: mmcif_col_types[k] for k in missing_keys if k in mmcif_col_types}
    for key in saved_col_types:
        del mmcif_col_types[key]

    pmmcif = PandasMmcif()

    try:
        # --- Fix 2: Read file as text and patch the missing _entry.id line ---
        with open(cif_path, 'r') as f:
            lines = f.readlines()

        cluster_id = Path(cif_path).stem

        # --- BEGIN CORRECTED PATCH ---
        for i, line in enumerate(lines):
            if line.startswith(f"data_{cluster_id}"):
                entry_line = f"_entry.id   {cluster_id}\n"
                # SIMPLER LOGIC:
                # Check if the very next line is the entry line.
                # If it's not, add the entry line.
                if not lines[i+1].strip().startswith("_entry.id"):
                    lines.insert(i + 1, entry_line)
                break
        # --- END CORRECTED PATCH ---

        file_content_string = "".join(lines)
        mmcif_data = pmmcif.read_mmcif_from_list(file_content_string)

    except Exception as e:
        logger.error(f"Failed to read mmCIF file: {cif_path}. Error: {e}")
        return {}
    finally:
        mmcif_col_types.update(saved_col_types)

    if "ATOM" not in mmcif_data.df or mmcif_data.df["ATOM"].empty:
        logger.warning(f"No ATOM records in {cif_path}")
        return {}
        
    atom_df = mmcif_data.df["ATOM"]

    # --- 1. Define Column Mapping ---
    rename_map = {
        'id': 'atom_number',
        'auth_asym_id': 'chain_id', # Use Author ID to match your cluster dict
        'auth_seq_id': 'residue_number',
        'label_comp_id': 'residue_name',
        'label_atom_id': 'atom_name',
        'label_alt_id': 'alt_loc',
        'group_PDB': 'record_name',
        'pdbx_formal_charge': 'charge',
        'Cartn_x': 'x_coord',
        'Cartn_y': 'y_coord',
        'Cartn_z': 'z_coord',
        'pdbx_PDB_ins_code': 'insertion',
        'type_symbol': 'element_symbol'
    }
    
    existing_cols_map = {k: v for k, v in rename_map.items() if k in atom_df.columns}
    pdb_df = atom_df.rename(columns=existing_cols_map)

    # --- 2. Fix Data Types ---
    num_cols = ['x_coord', 'y_coord', 'z_coord', 'atom_number', 'residue_number']
    for col in num_cols:
        if col in pdb_df.columns:
            pdb_df[col] = pd.to_numeric(pdb_df[col], errors='coerce')
    
    pdb_df.dropna(subset=['x_coord', 'y_coord', 'z_coord', 'atom_number'], inplace=True)
            
    str_cols = ['chain_id', 'alt_loc', 'insertion', 'record_name', 'element_symbol']
    for col in str_cols:
        if col not in pdb_df.columns: pdb_df[col] = ''
        else: pdb_df[col] = pdb_df[col].fillna('').astype(str)

        if col == 'chain_id':
            pdb_df[col] = pdb_df[col].str.upper()
    
    if 'residue_name' not in pdb_df.columns: pdb_df['residue_name'] = 'UNK'
    else: pdb_df['residue_name'] = pdb_df['residue_name'].fillna('UNK').astype(str)

    # --- 3. Split by Model (Based on your debug output) ---
    model_map = getattr(mmcif_data, "models", None)
    
    if not model_map:
        if 'pdbx_PDB_model_num' not in pdb_df.columns:
             model_map = {1: Path(cif_path).stem} 
             pdb_df['pdbx_PDB_model_num'] = Path(cif_path).stem
        else:
             unique_models = pdb_df['pdbx_PDB_model_num'].unique()
             model_map = {model_name: model_name for model_name in unique_models}

    model_dict = {}
    
    for key, model_name in model_map.items():
        mask = pdb_df['pdbx_PDB_model_num'] == model_name
        model_df = pdb_df[mask].copy().reset_index(drop=True)
        model_dict[model_name] = model_df
        
    return model_dict


def align_chains_of_pdb(
    protein,
    aligned_sequences,
    sequences_non_members,
    clusters_dict,
    representation="CA_BB",
    vocab=BASE_AMINO_ACIDS,
    one_to_three_mapping=STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    fill_value_coords=FILL_VALUE,
    seq_threshold=SEQ_SIMILARITY_THRESHOLD,
    homo_fill_value=HOMOMER_NEGATIVE,
):
    """
    Insert alignment gaps into a protein PyG Data object. Meant to be used directly after protein_to_pyg.

    Args:
        protein: Protein PyG Data object containing:
                - coords: Coordinates of the protein atoms (n_res, 37, 3).
                - residues: Amino acid sequence in three-letter code.
                - residue_type: Amino acid sequence in integer code.
                - residue_id: Residue ID in the format "chain:three-letter-code:residue-number".
                - chains: List of chain indices of the residues.
        seq: Amino acid sequence with gaps ('-') in one-letter code.
        vocab: List of amino acids.
        one_to_three_mapping: Mapping from one-letter to three-letter amino acid codes.
        fill_value_coords: Value to fill the coordinates of the gap residues. This should be 1e-5.

    """
    coords_all_chains = []
    residues_all_chains = []
    residue_type_all_chains = []
    residue_index_all_chains = []
    offset = 1000
    processed_chains = set()

    for member_id, seq in aligned_sequences.items():
        chain_id = member_id.split("_")[1].upper()
        chain_idx = protein.chain_mapping[chain_id]
        processed_chains.add(chain_id)
        chain_mask = protein.chains == chain_idx
        chain_indices = torch.where(chain_mask)[0]  # Get indices where mask is True
        residues_seq = [protein.residues[i] for i in chain_indices]
        chain_coords = protein.coords[chain_mask]

        n_gaps = seq.count("-")
        n_res_in = len(residues_seq)
        n_res_out = len(seq)

        assert (
            n_res_in + n_gaps == n_res_out
        ), f"Number of residues in protein ({n_res_in}) + number of gaps ({n_gaps}) != number of residues in sequence ({n_res_out})"  # noqa

        gap_idx = [i for i, aa in enumerate(seq) if aa == "-"]
        mask = torch.ones(n_res_out, dtype=bool)
        mask[gap_idx] = False
        coords = torch.zeros((n_res_out, *chain_coords.shape[1:]), dtype=torch.float32)
        coords[mask] = chain_coords
        coords[~mask] = fill_value_coords
        base_index = chain_idx * offset
        residue_indices = torch.arange(n_res_out, dtype=torch.int64) + base_index
        residue_index_all_chains.append(residue_indices)

        residues = []
        residue_type = []

        for aa in seq:
            if aa == "-":
                residues.append("GAP")
                residue_type.append(len(vocab))  # Create new gap index
            elif aa not in vocab:
                residues.append("UNK")
                residue_type.append(len(vocab) + 1)
            else:
                residues.append(one_to_three_mapping[aa])
                residue_type.append(vocab.index(aa))

        coords_all_chains.append(coords)
        residues_all_chains.append(residues)
        residue_type_all_chains.append(residue_type)

    for chain_id, seq_non_member in sequences_non_members.items():
        chain_id = chain_id.split("_")[1].upper()
        chain_idx = protein.chain_mapping[chain_id]
        chain_mask = protein.chains == chain_idx
        chain_indices = torch.where(chain_mask)[0]  # Get indices where mask is True
        residues_seq = [protein.residues[i] for i in chain_indices]
        chain_coords = protein.coords[chain_mask]

        n_res_in = len(residues_seq)
        n_res_out = len(seq_non_member)

        assert (
            n_res_in == n_res_out
        ), f"Number of residues in protein ({n_res_in}) != number of residues in sequence ({n_res_out})"  # noqa

        base_index = chain_idx * offset
        residue_indices = torch.arange(n_res_out, dtype=torch.int64) + base_index

        residues = []
        residue_type = []

        # TODO Delete the positions with non-vocab aminoacids from all the tensors/lists in this loop before appending
        seq_non_memb_mask = torch.zeros(len(seq_non_member), dtype=bool)
        for i, aa in enumerate(seq_non_member):
            if aa not in vocab:
                pass
            else:
                seq_non_memb_mask[i] = True
                residues.append(one_to_three_mapping[aa])
                residue_type.append(vocab.index(aa))

        coords_all_chains.append(chain_coords[seq_non_memb_mask])
        residues_all_chains.append(residues)
        residue_type_all_chains.append(residue_type)
        residue_index_all_chains.append(residue_indices[seq_non_memb_mask])

    coords_all_chains = torch.cat(coords_all_chains, dim=0)
    residue_index_all_chains = torch.cat(residue_index_all_chains, dim=0)
    residue_type_all_chains = torch.tensor([x for sublist in residue_type_all_chains for x in sublist])
    residues_all_chains = [x for sublist in residues_all_chains for x in sublist]

    if len(residues_all_chains) > coords_all_chains.shape[0]:
        raise ValueError("Not all chains included!")

    out = Data(
        coords=coords_all_chains,
        residues=residues_all_chains,
        residue_type=residue_type_all_chains,
        residue_index=residue_index_all_chains,
    )
    return out

def is_protein_chain(member_str, protein_chain_map):
        try:
            # Ensure member_str is valid and can be split
            if '_' not in member_str:
                return False
            pdb_id, chain_id = member_str.upper().split('_', 1)
            # Check if PDB is in map AND chain is in that PDB's set of chains
            if pdb_id in protein_chain_map and chain_id in protein_chain_map[pdb_id]:
                return True
        except (ValueError, AttributeError): # Catch bad formatting
            pass
        return False

def codnas_to_pyg(cluster_id, cluster_info, cif_dir, fasta_dir, protein_chain_map):
    """
    ... (docstring)
    """
    # --- Use correct paths based on new args ---
    fasta_path = pjoin(fasta_dir, cluster_id, f"{cluster_id}_aligned.fasta")
    fasta_non_members_path = pjoin(fasta_dir, cluster_id, f"{cluster_id}_non_members.fasta")
    cif_path = pjoin(cif_dir, cluster_id, f"{cluster_id}.cif")
    map_path = pjoin(cif_dir, cluster_id, f"{cluster_id}_mapping.json")
    # ---

    if not os.path.exists(fasta_path) or not os.path.exists(cif_path):
        logger.error(f"Missing required FASTA ({fasta_path}) or CIF ({cif_path}) file for cluster {cluster_id}. Skipping.")
        return None

    # --- LOAD THE MAPPING ---
    if not os.path.exists(map_path):
        logger.error(f"Missing mapping file ({map_path}) for cluster {cluster_id}. Skipping.")
        return None

    with open(map_path) as f:
        raw_map = json.load(f)
    # Create reverse mapping: PDB name -> model number (as it appears in CIF)
    # The mapping file has {1: "7Q4B_A", 2: "2NAO_A", ...}
    # We need {"7Q4B_A": 1, "2NAO_A": 2, ...} but keys might be strings
    name_to_model_num = {v.upper(): k for k, v in raw_map.items()}
    # ---

    # Use MAPPING members (what was actually downloaded), not CSV members
    # Filter to only include valid protein chains
    valid_protein_members = set(
        member for member in name_to_model_num.keys()
        if is_protein_chain(member, protein_chain_map)
    )

    fasta = read_fasta(fasta_path)
    fasta_non_members = read_fasta(fasta_non_members_path) if os.path.exists(fasta_non_members_path) else {}

    # Filter fasta dicts to only include valid protein chains
    fasta_protein = {
        key.upper(): seq for key, seq in fasta.items()
        if key.upper() in valid_protein_members
    }
    
    # --- Load the multi-model CIF ---
    cif_data = load_and_split_mmcif(cif_path)
    if not cif_data:
        logger.error(f"No model data loaded from {cif_path}. Skipping cluster {cluster_id}.")
        return None

    pyg_dict = {}
    processed_members_list = []

    for pyg_key in valid_protein_members:
        
        # 1. Use mapping to get the model number/key
        model_key = name_to_model_num.get(pyg_key)
        if model_key is None:
            logger.warning(f"[{cluster_id}] {pyg_key} not found in mapping. Skipping.")
            continue
        
        # 2. Check if that model exists in the CIF data
        # model_key might be int or str depending on how load_and_split_mmcif handles it
        # CIF model_dict uses int64 keys, but mapping JSON returns string keys
        try:
            int_key = int(model_key)
        except (ValueError, TypeError):
            int_key = None

        if model_key not in cif_data and str(model_key) not in cif_data and int_key not in cif_data:
            logger.warning(f"[{cluster_id}] Model {model_key} (for {pyg_key}) not found in {cif_path}. Skipping.")
            continue

        # Get the actual key that works
        if model_key in cif_data:
            actual_key = model_key
        elif str(model_key) in cif_data:
            actual_key = str(model_key)
        else:
            actual_key = int_key
        
        # 3. Check if its aligned sequence exists
        if pyg_key not in fasta_protein:
            logger.warning(f"[{cluster_id}] Aligned sequence for {pyg_key} not in {fasta_path}. Skipping.")
            continue
            
        try:
            # Get the dataframe for this single chain using the MODEL KEY, not pyg_key
            model_df = cif_data[actual_key]

            pyg_data = protein_to_pyg(df=model_df, remove_nonstandard=False)
            pyg_data.coords = pyg_data.coords[:, :3] # Keep only C, CA, N

            # Create a dict with only this *one* member's aligned sequence
            member_aligned_sequence = {pyg_key: fasta_protein[pyg_key]}
            non_member_sequences = {} # No non-members in this file
            
            # Run align_chains_of_pdb on just this one chain
            aligned_pyg_data = align_chains_of_pdb(
                protein=pyg_data,
                aligned_sequences=member_aligned_sequence,
                sequences_non_members=non_member_sequences,
                clusters_dict=list(valid_protein_members),  # clusters_dict is unused but required
            )
            
            pyg_dict[pyg_key] = aligned_pyg_data
            processed_members_list.append(pyg_key) # Add to our list of successes

        except Exception as e:
            logger.error(f"[{cluster_id}] Failed to process {pyg_key}: {e}. Skipping.", exc_info=True)
            continue
    
    # --- END NEW LOGIC ---

    if len(processed_members_list) < 2:
        logger.warning(f"Cluster {cluster_id} has fewer than 2 valid members ({len(processed_members_list)}) after final processing. Skipping save.")
        return None

    pyg_dict_save = Data(
        pyg_dict=pyg_dict,
        cluster_members=processed_members_list,
    )

    return pyg_dict_save


def resolve_ambiguous_aa(pred_sequence_logits, i, options):
    """
    Resolve ambiguous amino acid by comparing logit values.

    Args:
        pred_sequence_logits: Tensor of logit values
        i: Current position in sequence
        options: List of two indices to compare

    Returns:
        Index of the amino acid with higher probability
    """
    # Create a tensor with just the two options we want to compare
    relevant_logits = pred_sequence_logits[i, options]
    # Get the index (0 or 1) of the higher value
    choice = torch.argmax(relevant_logits)
    # Return the actual amino acid index we want
    return options[choice]
