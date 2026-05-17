from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure

from dynamicmpnn.types import (
    FILL_VALUE,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    STANDARD_AMINO_ACID_MAPPING_3_TO_1,
)


BACKBONE_ATOMS = ("N", "CA", "C")


@dataclass(frozen=True)
class ChainRecord:
    chain_id: str
    sequence: str
    residues_3: tuple[str, ...]
    residue_numbers: tuple[int, ...]
    residue_insertions: tuple[str, ...]
    coords: np.ndarray


def _get_parser(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def load_structure(path: Path):
    parser = _get_parser(path)
    return parser.get_structure(path.stem, str(path))


def get_single_model(path: Path):
    structure = load_structure(path)
    models = list(structure.get_models())
    if not models:
        raise ValueError(f"No models found in {path}")
    if len(models) != 1:
        raise ValueError(
            f"{path} contains {len(models)} models; evaluation inputs must be normalized to exactly one model"
        )
    return models[0]


def _normalise_resname(resname: str) -> str:
    return resname.strip().upper()


def _one_letter_code(resname: str) -> str:
    return STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(_normalise_resname(resname), "X")


def _three_letter_code(one_letter: str) -> str:
    return STANDARD_AMINO_ACID_MAPPING_1_TO_3.get(one_letter, "UNK")


def get_chain(model, chain_id: str):
    if chain_id in model.child_dict:
        return model.child_dict[chain_id]
    available = ",".join(model.child_dict.keys())
    raise ValueError(f"Chain '{chain_id}' not found. Available chains: {available}")


def iter_protein_chain_ids(model) -> tuple[str, ...]:
    chain_ids = []
    for chain in model.get_chains():
        if any(_is_protein_residue(residue) for residue in chain.get_residues()):
            chain_ids.append(chain.id)
    return tuple(chain_ids)


def _is_protein_residue(residue) -> bool:
    return is_aa(residue, standard=False) and residue.id[0] in {" ", "H_MSE"}


def extract_chain_record(model, chain_id: str) -> ChainRecord:
    chain = get_chain(model, chain_id)

    sequence: list[str] = []
    residues_3: list[str] = []
    residue_numbers: list[int] = []
    residue_insertions: list[str] = []
    coords: list[np.ndarray] = []

    for residue in chain.get_residues():
        if not _is_protein_residue(residue):
            continue

        resname = _normalise_resname(residue.get_resname())
        aa = _one_letter_code(resname)
        sequence.append(aa)
        residues_3.append(_three_letter_code(aa))
        residue_numbers.append(int(residue.id[1]))
        residue_insertions.append(str(residue.id[2]).strip())

        backbone = np.full((len(BACKBONE_ATOMS), 3), FILL_VALUE, dtype=np.float32)
        for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name not in residue:
                continue
            atom = residue[atom_name]
            if hasattr(atom, "selected_child"):
                atom = atom.selected_child
            backbone[atom_idx] = np.asarray(atom.get_coord(), dtype=np.float32)
        coords.append(backbone)

    if not sequence:
        raise ValueError(f"No amino-acid residues found for chain {chain_id}")

    return ChainRecord(
        chain_id=chain_id,
        sequence="".join(sequence),
        residues_3=tuple(residues_3),
        residue_numbers=tuple(residue_numbers),
        residue_insertions=tuple(residue_insertions),
        coords=np.stack(coords, axis=0),
    )


def build_protein_structure(model, chain_ids: tuple[str, ...] | None = None) -> Structure:
    selected_chain_ids = set(iter_protein_chain_ids(model) if chain_ids is None else chain_ids)

    structure = Structure("normalized")
    out_model = Model(0)

    for source_chain in model.get_chains():
        if source_chain.id not in selected_chain_ids:
            continue
        out_chain = Chain(source_chain.id)
        for residue in source_chain.get_residues():
            if not _is_protein_residue(residue):
                continue
            out_chain.add(copy.deepcopy(residue))
        if len(out_chain.child_list) > 0:
            out_model.add(out_chain)

    structure.add(out_model)
    return structure
