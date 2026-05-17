################################################
# Graphein
# Author: Eric J. Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

# Types used in the library.
##################################################

from typing import Dict, List, Literal, Tuple, NewType
import torch
from jaxtyping import Float
from torch import Tensor

# Value to fill missing coordinate entries when reading PDB files
FILL_VALUE = 1e6

SEQ_SIMILARITY_THRESHOLD = 70

HOMOMER_NEGATIVE = -1

SEQ_MASK = -1

# Small epsilon value added to distances to avoid division by zero
DISTANCE_EPS = 0.001

RESIDUE_TYPE_PAD = 22    # Padding token (after 20 AAs + GAP + UNK)

INDEX_CHAIN_PAD = -1     # Standard padding for indices


BASE_AMINO_ACIDS: List[str] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
"""Vocabulary of 20 standard amino acids."""

STANDARD_AMINO_ACIDS: List[str] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
"""Vocabulary of 24 amino acids with one-letter codes. Includes `fuzzy` standard amino acids:
``"B"`` denotes ``"ASX"`` which corresponds to ``"ASP"`` (``"D"``) **or** ``"ASN"`` (``"N"``),
``"J"`` denotes ``"XLE"`` which corresponds to ``"LEU"`` (``"L"``) **or** ``"ILE"`` (``"I"``),
and ``"Z"`` denotes ``"GLX"`` which corresponds to``"GLU"`` (``"E"``) **or** ``"GLN"`` (``"Q"``).
``"X"`` denotes unknown (``"UNK"`` or sometimes ``"XAA"``).
"""


STANDARD_AMINO_ACID_MAPPING_3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ASX": "B",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "XLE": "J",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
    "GLX": "Z",
}
"""
Mapping of 3-letter standard 24 amino acids codes to their one-letter form.
"""

STANDARD_AMINO_ACID_MAPPING_1_TO_3: Dict[str, str] = {
    v: k for k, v in STANDARD_AMINO_ACID_MAPPING_3_TO_1.items()
}
"""
Mapping of 1-letter standard amino acids codes to their three-letter form.
"""


PROTEIN_ATOMS: List[str] = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "OG",
    "CG",
    "CD1",
    "CD2",
    "CE1",
    "CE2",
    "CZ",
    "OD1",
    "ND2",
    "CG1",
    "CG2",
    "CD",
    "CE",
    "NZ",
    "OD2",
    "OE1",
    "NE2",
    "OE2",
    "OH",
    "NE",
    "NH1",
    "NH2",
    "OG1",
    "SD",
    "ND1",
    "SG",
    "NE1",
    "CE3",
    "CZ2",
    "CZ3",
    "CH2",
    "OXT",
]
"""List of standard atom types present in protein structures."""


LossType = Literal[
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "dihedral_loss"
]

ModelOutput = NewType("ModelOutput", Tuple[torch.Tensor, torch.Tensor])

Label = NewType("Label", Dict[str, List])


OrientationTensor = NewType(
    "OrientationTensor", Float[torch.Tensor, "n_nodes 2 3"]
)

ScalarNodeFeature = Literal[
    "amino_acid_one_hot",
    "alpha",
    "kappa",
    "dihedrals",
    "sidechain_torsions",
    "sequence_positional_encoding",
]
VectorNodeFeature = Literal["orientation", "virtual_cb_vector"]
ScalarEdgeFeature = Literal["edge_distance", "sequence_distance", "rbf_32+edge_distance"]
VectorEdgeFeature = Literal["edge_vectors", "pos_emb"]
