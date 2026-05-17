from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from Bio.PDB.Polypeptide import is_aa

from dynamicmpnn.eval.af3 import AF3RunResult
from dynamicmpnn.eval.structure_utils import get_chain, get_single_model, iter_protein_chain_ids
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec, RawMetricRecord


def _is_protein_residue(residue) -> bool:
    return is_aa(residue, standard=False) and residue.id[0] in {" ", "H_MSE"}


def _extract_ca_coords(structure_path: Path, chain_id: str | None = None) -> np.ndarray:
    model = get_single_model(structure_path)
    chain_ids = (chain_id,) if chain_id is not None else iter_protein_chain_ids(model)
    coords = []
    for current_chain_id in chain_ids:
        chain = get_chain(model, current_chain_id)
        for residue in chain.get_residues():
            if not _is_protein_residue(residue):
                continue
            if "CA" not in residue:
                continue
            atom = residue["CA"]
            if hasattr(atom, "selected_child"):
                atom = atom.selected_child
            coords.append(np.asarray(atom.get_coord(), dtype=np.float64))
    if not coords:
        raise ValueError(f"No CA coordinates found in {structure_path}")
    return np.stack(coords, axis=0)


def _kabsch(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    reference_centered = reference - reference.mean(axis=0)
    mobile_centered = mobile - mobile.mean(axis=0)
    covariance = mobile_centered.T @ reference_centered
    v_matrix, _, w_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(v_matrix @ w_transpose) < 0:
        correction[-1, -1] = -1.0
    rotation = v_matrix @ correction @ w_transpose
    return mobile_centered @ rotation


def compute_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    aligned_mobile = _kabsch(reference, mobile)
    aligned_reference = reference - reference.mean(axis=0)
    squared = np.square(aligned_reference - aligned_mobile).sum(axis=1)
    return float(np.sqrt(np.mean(squared)))


def compute_tm_score(reference: np.ndarray, mobile: np.ndarray) -> float:
    aligned_mobile = _kabsch(reference, mobile)
    aligned_reference = reference - reference.mean(axis=0)
    distances = np.linalg.norm(aligned_reference - aligned_mobile, axis=1)
    length = len(distances)
    if length == 0:
        raise ValueError("TM-score requires at least one aligned residue")
    d0 = max(1.24 * max(length - 15, 1) ** (1.0 / 3.0) - 1.8, 0.5)
    return float(np.mean(1.0 / (1.0 + np.square(distances / d0))))


def compute_lddt(reference: np.ndarray, mobile: np.ndarray, cutoff: float = 15.0) -> Optional[float]:
    if len(reference) < 2:
        return None
    ref_dist = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1)
    mob_dist = np.linalg.norm(mobile[:, None, :] - mobile[None, :, :], axis=-1)
    mask = (ref_dist > 0) & (ref_dist < cutoff)
    if not np.any(mask):
        return None
    delta = np.abs(ref_dist - mob_dist)
    score = (
        (delta[mask] < 0.5).mean()
        + (delta[mask] < 1.0).mean()
        + (delta[mask] < 2.0).mean()
        + (delta[mask] < 4.0).mean()
    ) / 4.0
    return float(score)


def _flatten_numeric_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return []


def extract_mean_plddt(confidences_path: Path) -> Optional[float]:
    payload = json.loads(confidences_path.read_text(encoding="utf-8"))
    candidates: list[list[float]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if "plddt" in key.lower():
                    numeric = _flatten_numeric_values(value)
                    if numeric:
                        candidates.append(numeric)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    if not candidates:
        return None
    best = max(candidates, key=len)
    return float(sum(best) / len(best))


def extract_ranking_score(summary_confidences_path: Path) -> Optional[float]:
    payload = json.loads(summary_confidences_path.read_text(encoding="utf-8"))
    ranking_score = payload.get("ranking_score")
    if isinstance(ranking_score, (int, float)):
        return float(ranking_score)
    return None


def discover_prediction_artifacts(output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    structure_candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
    )
    confidence_candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".json" and "confid" in path.name.lower()
    )
    return (
        structure_candidates[0] if structure_candidates else None,
        confidence_candidates[0] if confidence_candidates else None,
    )


def _iter_prediction_candidates(output_dir: Path) -> list[tuple[Path, Optional[int], Optional[int]]]:
    import re

    seed_sample_dir_pattern = re.compile(r"^seed-(\d+)_sample-(\d+)$")
    candidates = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_dir():
            continue
        match = seed_sample_dir_pattern.fullmatch(path.name)
        if match is None:
            continue
        candidates.append((path, int(match.group(1)), int(match.group(2))))
    if candidates:
        return candidates
    return [(output_dir, None, None)]


def _get_target_protein(job: AF3JobSpec) -> AF3ProteinSpec:
    for protein in job.proteins:
        if protein.entity_id == job.predicted_target_chain_id:
            return protein
    if not job.proteins:
        raise ValueError(f"AF3 job {job.job_name} does not define any protein entities")
    return job.proteins[0]


def score_jobs(
    jobs: Sequence[AF3JobSpec],
    templates: dict[tuple[str, str], TemplateArtifact],
    run_results: dict[str, AF3RunResult],
) -> tuple[RawMetricRecord, ...]:
    records = []
    for job in jobs:
        run_result = run_results[job.job_name]
        if run_result.status != "completed":
            records.append(
                RawMetricRecord(
                    sequence_id=job.sequence_id,
                    state_name=job.state_name,
                    structure_kind=job.structure_kind,
                    af3_status=run_result.status,
                    error=run_result.error,
                )
            )
            continue

        template = templates[(job.state_name, job.structure_kind)]
        reference_ca = np.asarray(template.ca_coords, dtype=np.float64)
        target_protein = _get_target_protein(job)
        query_indices = np.asarray(target_protein.query_indices, dtype=np.int64)
        template_indices = np.asarray(target_protein.template_indices, dtype=np.int64)

        for candidate_dir, seed, af3_sample in _iter_prediction_candidates(run_result.output_dir):
            structure_path, confidence_path = discover_prediction_artifacts(candidate_dir)
            summary_confidences_path = next(
                (
                    path
                    for path in sorted(candidate_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() == ".json" and "summary_confid" in path.name.lower()
                ),
                None,
            )
            if structure_path is None:
                records.append(
                    RawMetricRecord(
                        sequence_id=job.sequence_id,
                        state_name=job.state_name,
                        structure_kind=job.structure_kind,
                        af3_status="failed",
                        error="No predicted structure file found in AF3 output directory",
                        seed=seed,
                        af3_sample=af3_sample,
                    )
                )
                continue

            predicted_ca = _extract_ca_coords(structure_path, chain_id=job.predicted_target_chain_id)
            if query_indices.size == 0:
                records.append(
                    RawMetricRecord(
                        sequence_id=job.sequence_id,
                        state_name=job.state_name,
                        structure_kind=job.structure_kind,
                        af3_status="failed",
                        error="Alignment mapping produced no query indices",
                        seed=seed,
                        af3_sample=af3_sample,
                    )
                )
                continue

            if predicted_ca.shape[0] <= int(query_indices.max()):
                records.append(
                    RawMetricRecord(
                        sequence_id=job.sequence_id,
                        state_name=job.state_name,
                        structure_kind=job.structure_kind,
                        af3_status="failed",
                        error=(
                            f"Predicted structure has {predicted_ca.shape[0]} residues, "
                            f"but query mapping needs index {int(query_indices.max())}"
                        ),
                        seed=seed,
                        af3_sample=af3_sample,
                    )
                )
                continue

            reference_subset = reference_ca[template_indices]
            predicted_subset = predicted_ca[query_indices]
            if reference_subset.shape != predicted_subset.shape:
                records.append(
                    RawMetricRecord(
                        sequence_id=job.sequence_id,
                        state_name=job.state_name,
                        structure_kind=job.structure_kind,
                        af3_status="failed",
                        error="Reference and predicted coordinate subsets do not match in shape",
                        seed=seed,
                        af3_sample=af3_sample,
                    )
                )
                continue

            mean_plddt = extract_mean_plddt(confidence_path) if confidence_path is not None else None
            ranking_score = (
                extract_ranking_score(summary_confidences_path) if summary_confidences_path is not None else None
            )
            records.append(
                RawMetricRecord(
                    sequence_id=job.sequence_id,
                    state_name=job.state_name,
                    structure_kind=job.structure_kind,
                    af3_status="completed",
                    predicted_structure_path=structure_path,
                    confidences_path=confidence_path,
                    tm_score=compute_tm_score(reference_subset, predicted_subset),
                    lddt=compute_lddt(reference_subset, predicted_subset),
                    rmsd=compute_rmsd(reference_subset, predicted_subset),
                    mean_plddt=mean_plddt,
                    seed=seed,
                    af3_sample=af3_sample,
                    ranking_score=ranking_score,
                )
            )

    return tuple(records)
