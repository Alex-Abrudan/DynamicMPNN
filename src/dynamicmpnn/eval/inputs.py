from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from dynamicmpnn.eval.structure_utils import ChainRecord, extract_chain_record, get_single_model, iter_protein_chain_ids
from dynamicmpnn.eval.types import AlignmentSpec, EvaluationRequest
from dynamicmpnn.types import BASE_AMINO_ACIDS, HOMOMER_NEGATIVE, SEQ_SIMILARITY_THRESHOLD


@dataclass(frozen=True)
class ParsedStructure:
    name: str
    pdb_path: Path
    target_chain_id: str
    target_chain: ChainRecord
    context_chains: tuple[ChainRecord, ...]


@dataclass(frozen=True)
class PreparedState:
    name: str
    pdb_path: Path
    chain_id: str
    sequence: str
    aligned_sequence: str
    residues_3: tuple[str, ...]
    residue_numbers: tuple[int, ...]
    residue_insertions: tuple[str, ...]
    coords: np.ndarray
    query_indices: tuple[int, ...]
    template_indices: tuple[int, ...]
    context_chains: tuple[ChainRecord, ...]
    pyg_data: Optional[Data]


@dataclass(frozen=True)
class PreparedInputBundle:
    request: EvaluationRequest
    targets: tuple[PreparedState, PreparedState]
    decoys: Optional[dict[str, PreparedState]]
    multi_state_data: Data


def _sequence_similarity(seq1: str, seq2: str) -> float:
    match_score = 1
    mismatch_score = -1
    gap_penalty = -1
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = np.zeros((rows, cols), dtype=np.int32)

    score_matrix[:, 0] = np.arange(rows) * gap_penalty
    score_matrix[0, :] = np.arange(cols) * gap_penalty

    for i in range(1, rows):
        for j in range(1, cols):
            match = score_matrix[i - 1, j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
            delete = score_matrix[i - 1, j] + gap_penalty
            insert = score_matrix[i, j - 1] + gap_penalty
            score_matrix[i, j] = max(match, delete, insert)

    aligned1: list[str] = []
    aligned2: list[str] = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and score_matrix[i, j]
            == score_matrix[i - 1, j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
        ):
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + gap_penalty:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    matches = sum(1 for left, right in zip(aligned1, aligned2) if left == right)
    return (matches / len(aligned1)) * 100 if aligned1 else 0.0


def parse_structure(name: str, pdb_path: Path, chain_id: str) -> ParsedStructure:
    model = get_single_model(pdb_path)
    target_chain = extract_chain_record(model, chain_id)

    context_chains = tuple(
        extract_chain_record(model, other_chain_id)
        for other_chain_id in iter_protein_chain_ids(model)
        if other_chain_id != chain_id
    )

    return ParsedStructure(
        name=name,
        pdb_path=pdb_path,
        target_chain_id=chain_id,
        target_chain=target_chain,
        context_chains=context_chains,
    )


def resolve_alignment(
    state1_sequence: str,
    state2_sequence: str,
    state1_template_indices: Optional[tuple[int, ...]],
    state2_template_indices: Optional[tuple[int, ...]],
) -> AlignmentSpec:
    state1_template_indices = tuple(state1_template_indices or ())
    state2_template_indices = tuple(state2_template_indices or ())

    if not state1_template_indices and not state2_template_indices:
        if len(state1_sequence) != len(state2_sequence):
            raise ValueError(
                "Explicit alignment indices are required when the selected chains do not have the same length"
            )
        state1_template_indices = tuple(range(len(state1_sequence)))
        state2_template_indices = tuple(range(len(state2_sequence)))

    if (not state1_template_indices) != (not state2_template_indices):
        raise ValueError("Alignment indices must be provided for both states or for neither")
    if len(state1_template_indices) != len(state2_template_indices):
        raise ValueError("Alignment index lists must have the same length")
    if not state1_template_indices:
        raise ValueError("Alignment mappings cannot be empty")

    for state_name, sequence, template_indices in (
        ("state1", state1_sequence, state1_template_indices),
        ("state2", state2_sequence, state2_template_indices),
    ):
        if len(set(template_indices)) != len(template_indices):
            raise ValueError(f"{state_name} alignment indices must be unique")
        invalid_index = next(
            (index for index in template_indices if index < 0 or index >= len(sequence)),
            None,
        )
        if invalid_index is not None:
            raise ValueError(
                f"{state_name} alignment index {invalid_index + 1} is out of range for sequence length {len(sequence)}"
            )

    query_indices = tuple(range(len(state1_template_indices)))
    state1_aligned = "".join(state1_sequence[index] for index in state1_template_indices)
    state2_aligned = "".join(state2_sequence[index] for index in state2_template_indices)

    return AlignmentSpec(
        state1_aligned=state1_aligned,
        state2_aligned=state2_aligned,
        query_length=len(query_indices),
        state1_query_indices=query_indices,
        state2_query_indices=query_indices,
        state1_template_indices=state1_template_indices,
        state2_template_indices=state2_template_indices,
        paired_query_indices=query_indices,
        paired_state1_template_indices=state1_template_indices,
        paired_state2_template_indices=state2_template_indices,
    )


def _encode_residue_type(sequence: str) -> list[int]:
    encoded = []
    for aa in sequence:
        if aa in BASE_AMINO_ACIDS:
            encoded.append(BASE_AMINO_ACIDS.index(aa))
        else:
            encoded.append(len(BASE_AMINO_ACIDS) + 1)
    return encoded


def _build_target_only_pyg_data(parsed_structure: ParsedStructure, template_indices: tuple[int, ...]) -> Data:
    target_chain = parsed_structure.target_chain
    coords = target_chain.coords[list(template_indices)]
    residues = [target_chain.residues_3[index] for index in template_indices]
    residue_type = _encode_residue_type("".join(target_chain.sequence[index] for index in template_indices))
    residue_index = list(range(len(template_indices)))

    return Data(
        coords=torch.tensor(coords, dtype=torch.float32),
        residues=residues,
        residue_type=torch.tensor(residue_type, dtype=torch.long),
        residue_index=torch.tensor(residue_index, dtype=torch.long),
        chains=torch.zeros(len(template_indices), dtype=torch.long),
        mask_seq=torch.zeros(len(template_indices), dtype=torch.bool),
        homo_idx=torch.zeros(len(template_indices), dtype=torch.long),
        homomer_mapping={parsed_structure.target_chain_id: 0},
        chain_mapping={parsed_structure.target_chain_id: 0},
    )


def _build_multichain_pyg_data(parsed_structure: ParsedStructure, aligned_sequence: str, template_indices: tuple[int, ...]) -> Data:
    coords: list[np.ndarray] = []
    residues: list[str] = []
    residue_type: list[int] = []
    residue_index: list[int] = []
    chain_index: list[int] = []
    homo_idx: list[int] = []
    mask_seq: list[bool] = []
    chain_mapping: dict[str, int] = {}
    homomer_mapping: dict[str, int] = {}
    offset = 1000

    target_chain = parsed_structure.target_chain
    chain_mapping[target_chain.chain_id] = 0
    homomer_mapping[target_chain.chain_id] = 0

    for aligned_idx, template_idx in enumerate(template_indices):
        aa = target_chain.sequence[template_idx]
        coords.append(target_chain.coords[template_idx])
        residues.append(target_chain.residues_3[template_idx])
        residue_type.extend(_encode_residue_type(aa))
        residue_index.append(aligned_idx)
        chain_index.append(0)
        homo_idx.append(0)
        mask_seq.append(False)

    for context_idx, context_chain in enumerate(parsed_structure.context_chains, start=1):
        similarity = _sequence_similarity(context_chain.sequence, aligned_sequence)
        chain_mapping[context_chain.chain_id] = context_idx
        homomer_mapping[context_chain.chain_id] = context_idx
        for residue_idx, aa in enumerate(context_chain.sequence):
            coords.append(context_chain.coords[residue_idx])
            residues.append(context_chain.residues_3[residue_idx])
            residue_type.extend(_encode_residue_type(aa))
            residue_index.append(context_idx * offset + residue_idx)
            chain_index.append(context_idx)
            homo_idx.append(HOMOMER_NEGATIVE)
            mask_seq.append(similarity <= SEQ_SIMILARITY_THRESHOLD)

    return Data(
        coords=torch.tensor(np.stack(coords, axis=0), dtype=torch.float32),
        residues=residues,
        residue_type=torch.tensor(residue_type, dtype=torch.long),
        residue_index=torch.tensor(residue_index, dtype=torch.long),
        chains=torch.tensor(chain_index, dtype=torch.long),
        mask_seq=torch.tensor(mask_seq, dtype=torch.bool),
        homo_idx=torch.tensor(homo_idx, dtype=torch.long),
        homomer_mapping=homomer_mapping,
        chain_mapping=chain_mapping,
    )


def _prepare_target_state(
    parsed_structure: ParsedStructure,
    aligned_sequence: str,
    query_indices: tuple[int, ...],
    template_indices: tuple[int, ...],
    sampling_mode: str,
) -> PreparedState:
    if sampling_mode == "multi":
        pyg_data = _build_multichain_pyg_data(parsed_structure, aligned_sequence, template_indices)
    else:
        pyg_data = _build_target_only_pyg_data(parsed_structure, template_indices)

    target_chain = parsed_structure.target_chain
    return PreparedState(
        name=parsed_structure.name,
        pdb_path=parsed_structure.pdb_path,
        chain_id=parsed_structure.target_chain_id,
        sequence=target_chain.sequence,
        aligned_sequence=aligned_sequence,
        residues_3=target_chain.residues_3,
        residue_numbers=target_chain.residue_numbers,
        residue_insertions=target_chain.residue_insertions,
        coords=target_chain.coords,
        query_indices=query_indices,
        template_indices=template_indices,
        context_chains=parsed_structure.context_chains,
        pyg_data=pyg_data,
    )


def _prepare_decoy_state(
    parsed_structure: ParsedStructure,
    target_query_indices: tuple[int, ...],
    target_template_indices: tuple[int, ...],
) -> PreparedState:
    target_chain = parsed_structure.target_chain
    if len(target_query_indices) != len(target_template_indices):
        raise ValueError("Resolved decoy mapping must have matching query and template lengths")
    if not target_template_indices:
        raise ValueError("Resolved decoy mapping cannot be empty")

    max_template_index = max(target_template_indices)
    if max_template_index >= len(target_chain.sequence):
        raise ValueError(
            f"Decoy {parsed_structure.pdb_path} chain {parsed_structure.target_chain_id} has length {len(target_chain.sequence)} "
            f"but the matching target mapping needs residue {max_template_index + 1}"
        )

    return PreparedState(
        name=parsed_structure.name,
        pdb_path=parsed_structure.pdb_path,
        chain_id=parsed_structure.target_chain_id,
        sequence=target_chain.sequence,
        aligned_sequence="".join(target_chain.sequence[index] for index in target_template_indices),
        residues_3=target_chain.residues_3,
        residue_numbers=target_chain.residue_numbers,
        residue_insertions=target_chain.residue_insertions,
        coords=target_chain.coords,
        query_indices=target_query_indices,
        template_indices=target_template_indices,
        context_chains=parsed_structure.context_chains,
        pyg_data=None,
    )


def resolve_inputs(request: EvaluationRequest) -> PreparedInputBundle:
    parsed_target1 = parse_structure(request.targets[0].name, request.targets[0].pdb_path, request.targets[0].chain_id)
    parsed_target2 = parse_structure(request.targets[1].name, request.targets[1].pdb_path, request.targets[1].chain_id)

    alignment = resolve_alignment(
        parsed_target1.target_chain.sequence,
        parsed_target2.target_chain.sequence,
        request.alignment.state1_template_indices,
        request.alignment.state2_template_indices,
    )
    if request.sequence_recovery_target_sequences:
        for target_sequence in request.sequence_recovery_target_sequences:
            if len(target_sequence) != alignment.query_length:
                raise ValueError("Sequence recovery targets must match the resolved sampled-sequence length")
    resolved_request = replace(request, alignment=alignment)

    target_state1 = _prepare_target_state(
        parsed_target1,
        alignment.state1_aligned,
        alignment.state1_query_indices,
        alignment.state1_template_indices,
        resolved_request.sampling_mode,
    )
    target_state2 = _prepare_target_state(
        parsed_target2,
        alignment.state2_aligned,
        alignment.state2_query_indices,
        alignment.state2_template_indices,
        resolved_request.sampling_mode,
    )

    decoys = None
    if request.decoys is not None:
        decoy_alignment = {
            "state1": (alignment.state1_query_indices, alignment.state1_template_indices),
            "state2": (alignment.state2_query_indices, alignment.state2_template_indices),
        }
        decoys = {}
        for state_name, decoy_input in request.decoys.items():
            parsed_decoy = parse_structure(decoy_input.name, decoy_input.pdb_path, decoy_input.chain_id)
            query_indices, template_indices = decoy_alignment[state_name]
            decoys[state_name] = _prepare_decoy_state(parsed_decoy, query_indices, template_indices)

    multi_state_data = Data(
        pyg_dict={
            target_state1.name.upper(): target_state1.pyg_data,
            target_state2.name.upper(): target_state2.pyg_data,
        },
        cluster_members=[
            f"{target_state1.name.upper()}_{target_state1.chain_id}",
            f"{target_state2.name.upper()}_{target_state2.chain_id}",
        ],
    )

    return PreparedInputBundle(
        request=resolved_request,
        targets=(target_state1, target_state2),
        decoys=decoys,
        multi_state_data=multi_state_data,
    )
