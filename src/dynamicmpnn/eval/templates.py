from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

from dynamicmpnn.eval.inputs import PreparedInputBundle, PreparedState
from dynamicmpnn.eval.structure_utils import get_single_model


@dataclass(frozen=True)
class TemplateArtifact:
    state_name: str
    structure_kind: str
    pdb_path: Path
    mmcif_path: Path
    query_indices: tuple[int, ...]
    template_indices: tuple[int, ...]
    ca_coords: tuple[tuple[float, float, float], ...]


def _build_single_chain_structure(state: PreparedState) -> Structure:
    model = get_single_model(state.pdb_path)
    if state.chain_id not in model.child_dict:
        raise ValueError(f"Chain '{state.chain_id}' not found in {state.pdb_path}")

    source_chain = model.child_dict[state.chain_id]
    out_structure = Structure(f"{state.name}_template")
    out_model = Model(0)
    out_chain = Chain(state.chain_id)

    residue_counter = 1
    for residue in source_chain.get_residues():
        if residue.id[0] not in {" ", "H_MSE"}:
            continue
        copied = copy.deepcopy(residue)
        copied.id = (" ", residue_counter, " ")
        out_chain.add(copied)
        residue_counter += 1

    out_model.add(out_chain)
    out_structure.add(out_model)
    return out_structure


def _write_structure_files(state: PreparedState, structure_kind: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = _build_single_chain_structure(state)
    stem = f"{state.name}_{structure_kind}"

    pdb_path = output_dir / f"{stem}.pdb"
    mmcif_path = output_dir / f"{stem}.cif"

    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(str(pdb_path))

    mmcif_io = MMCIFIO()
    mmcif_io.set_structure(structure)
    mmcif_io.save(str(mmcif_path))
    # AF3 requires a template revision date, and Bio.PDB's PDB -> mmCIF
    # conversion does not populate this category for us.
    revision_block = """#
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2018-03-14
#
"""
    mmcif_text = mmcif_path.read_text(encoding="utf-8")
    if "_pdbx_audit_revision_history.revision_date" not in mmcif_text:
        mmcif_path.write_text(mmcif_text.rstrip() + "\n" + revision_block, encoding="utf-8")

    return pdb_path, mmcif_path


def _artifact_from_state(state: PreparedState, structure_kind: str, output_dir: Path) -> TemplateArtifact:
    pdb_path, mmcif_path = _write_structure_files(state, structure_kind, output_dir)
    ca_coords = tuple(tuple(float(value) for value in coord[1]) for coord in state.coords)
    return TemplateArtifact(
        state_name=state.name,
        structure_kind=structure_kind,
        pdb_path=pdb_path,
        mmcif_path=mmcif_path,
        query_indices=state.query_indices,
        template_indices=state.template_indices,
        ca_coords=ca_coords,
    )


def prepare_templates(bundle: PreparedInputBundle, output_dir: Path) -> dict[tuple[str, str], TemplateArtifact]:
    templates: dict[tuple[str, str], TemplateArtifact] = {}

    target_dir = output_dir / "targets"
    for state in bundle.targets:
        templates[(state.name, "target")] = _artifact_from_state(state, "target", target_dir)

    if bundle.decoys is not None:
        decoy_dir = output_dir / "decoys"
        for state_name, state in bundle.decoys.items():
            templates[(state_name, "decoy")] = _artifact_from_state(
                state,
                "decoy",
                decoy_dir,
            )

    return templates
