from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamicmpnn.eval.inputs import PreparedInputBundle, PreparedState
from dynamicmpnn.eval.structure_utils import _is_protein_residue, get_chain, get_single_model
from dynamicmpnn.types import FILL_VALUE


# N, CA and C come from the canonical residue list; O is read back off the
# structure and CB is built, so this order is what the runner unpacks.
GUESS_ATOMS = ("N", "CA", "C", "O", "CB")


@dataclass(frozen=True)
class GuessArtifact:
    """The initial-guess counterpart of a TemplateArtifact.

    Carries the same (state, kind) key, the same query/template index mapping
    and the same reference CA coordinates as the AF3 template for that state,
    so both metrics score the same residues against the same ground truth.
    """

    state_name: str
    structure_kind: str
    npz_path: Path
    query_indices: tuple[int, ...]
    template_indices: tuple[int, ...]
    ca_coords: tuple[tuple[float, float, float], ...]


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ideal CB from the backbone frame, independent of residue identity.

    AF2 forms its pseudo-beta from CB, so reading CB off the native residue
    would smuggle a weak side-chain signal of the native sequence into a
    metric whose whole point is that it never sees one.
    """
    b = ca - n
    c_vector = c - ca
    a = np.cross(b, c_vector)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vector + ca


def _oxygen_coords(state: PreparedState) -> np.ndarray:
    """Backbone O in the same residue order as `PreparedState.coords`.

    `extract_chain_record` keeps only N/CA/C, but the recycling embedder is
    handed all-atom slots and O is cheap to recover. The residue filter is
    imported rather than restated: the alignment indices are positions in
    THAT residue list, so a filter that drifts would silently shift the whole
    backbone against the indices that select it.
    """
    model = get_single_model(state.pdb_path)
    chain = get_chain(model, state.chain_id)
    coords = []
    for residue in chain.get_residues():
        if not _is_protein_residue(residue):
            continue
        if "O" not in residue:
            coords.append(np.full(3, FILL_VALUE, dtype=np.float32))
            continue
        atom = residue["O"]
        if hasattr(atom, "selected_child"):
            atom = atom.selected_child
        coords.append(np.asarray(atom.get_coord(), dtype=np.float32))
    return np.stack(coords, axis=0) if coords else np.zeros((0, 3), dtype=np.float32)


def build_backbone(state: PreparedState) -> tuple[np.ndarray, np.ndarray]:
    """(L, 5, 3) N/CA/C/O/CB coordinates and their (L, 5) presence mask."""
    backbone_ncac = np.asarray(state.coords, dtype=np.float32)
    length = backbone_ncac.shape[0]
    oxygen = _oxygen_coords(state)
    if oxygen.shape[0] != length:
        raise ValueError(
            f"Chain {state.chain_id} of {state.pdb_path} yielded {oxygen.shape[0]} oxygens "
            f"for {length} residues; the residue filters disagree"
        )

    present_ncac = ~np.any(np.isclose(backbone_ncac, FILL_VALUE), axis=-1)
    present_o = ~np.any(np.isclose(oxygen, FILL_VALUE), axis=-1)

    xyz = np.zeros((length, len(GUESS_ATOMS), 3), dtype=np.float32)
    mask = np.zeros((length, len(GUESS_ATOMS)), dtype=np.float32)
    xyz[:, :3] = np.where(present_ncac[..., None], backbone_ncac, 0.0)
    mask[:, :3] = present_ncac.astype(np.float32)
    xyz[:, 3] = np.where(present_o[:, None], oxygen, 0.0)
    mask[:, 3] = present_o.astype(np.float32)

    # CB needs a complete N/CA/C frame; glycine gets one too, which is what
    # AF2's own pseudo-beta does for glycine residues.
    complete_frame = present_ncac.all(axis=-1)
    virtual = _virtual_cb(backbone_ncac[:, 0], backbone_ncac[:, 1], backbone_ncac[:, 2])
    xyz[:, 4] = np.where(complete_frame[:, None], virtual, 0.0)
    mask[:, 4] = complete_frame.astype(np.float32)

    return xyz, mask


def _artifact_from_state(state: PreparedState, structure_kind: str, output_dir: Path) -> GuessArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz, mask = build_backbone(state)
    npz_path = output_dir / f"{state.name}_{structure_kind}.npz"
    # The atom order travels with the coordinates: the predictor runs in a
    # separate environment and cannot import this module to look it up.
    np.savez_compressed(npz_path, xyz=xyz, mask=mask, atoms=np.array(GUESS_ATOMS))
    ca_coords = tuple(tuple(float(value) for value in coord[1]) for coord in state.coords)
    return GuessArtifact(
        state_name=state.name,
        structure_kind=structure_kind,
        npz_path=npz_path,
        query_indices=state.query_indices,
        template_indices=state.template_indices,
        ca_coords=ca_coords,
    )


def prepare_guesses(bundle: PreparedInputBundle, output_dir: Path) -> dict[tuple[str, str], GuessArtifact]:
    """One backbone artifact per (state, structure kind), as prepare_templates."""
    artifacts: dict[tuple[str, str], GuessArtifact] = {}

    target_dir = output_dir / "targets"
    for state in bundle.targets:
        artifacts[(state.name, "target")] = _artifact_from_state(state, "target", target_dir)

    if bundle.decoys is not None:
        decoy_dir = output_dir / "decoys"
        for state_name, state in bundle.decoys.items():
            artifacts[(state_name, "decoy")] = _artifact_from_state(state, "decoy", decoy_dir)

    return artifacts
