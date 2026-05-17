import json
from pathlib import Path

import numpy as np

from dynamicmpnn.eval.af3 import AF3RunResult
from dynamicmpnn.eval.scoring import compute_rmsd, compute_tm_score, extract_mean_plddt, score_jobs
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec


def _write_pdb(path: Path, chains: tuple[tuple[str, tuple[str, ...], float], ...]) -> None:
    atom_serial = 1
    lines: list[str] = []
    for chain_id, residues, x_offset in chains:
        for residue_index, residue_name in enumerate(residues, start=1):
            base_x = x_offset + float((residue_index - 1) * 4)
            lines.extend(
                [
                    f"ATOM  {atom_serial:5d}  N   {residue_name:>3} {chain_id}{residue_index:4d}    {base_x + 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           N\n",
                    f"ATOM  {atom_serial + 1:5d}  CA  {residue_name:>3} {chain_id}{residue_index:4d}    {base_x + 2.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n",
                    f"ATOM  {atom_serial + 2:5d}  C   {residue_name:>3} {chain_id}{residue_index:4d}    {base_x + 3.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n",
                ]
            )
            atom_serial += 3
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _target_job(tmp_path: Path, output_dir: Path, *, include_context: bool = False) -> AF3JobSpec:
    proteins = [
        AF3ProteinSpec(
            entity_id="A",
            sequence="AC",
            template_mmcif=tmp_path / "template.cif",
            query_indices=(0, 1),
            template_indices=(0, 1),
        )
    ]
    if include_context:
        proteins.append(AF3ProteinSpec(entity_id="B", sequence="GGG"))

    return AF3JobSpec(
        job_name="sample_000__state1__target",
        sequence_id="sample_000",
        state_name="state1",
        structure_kind="target",
        input_dir=tmp_path / "job_input",
        json_path=tmp_path / "job.json",
        output_dir=output_dir,
        proteins=tuple(proteins),
        model_seeds=(0,),
    )


def test_extract_mean_plddt_from_nested_json(tmp_path):
    payload = {"candidates": [{"plddt": [80.0, 90.0]}, {"other": {"atom_plddts": [70.0]}}]}
    confidence_path = tmp_path / "sample_confidences.json"
    confidence_path.write_text(json.dumps(payload), encoding="utf-8")

    assert extract_mean_plddt(confidence_path) == 85.0


def test_score_jobs_on_identical_structure(tmp_path):
    output_dir = tmp_path / "af3_job"
    output_dir.mkdir()
    predicted_path = output_dir / "prediction.pdb"
    confidence_path = output_dir / "sample_confidences.json"
    _write_pdb(predicted_path, (("A", ("ALA", "CYS"), 0.0),))
    confidence_path.write_text(json.dumps({"plddt": [88.0, 92.0]}), encoding="utf-8")

    template = TemplateArtifact(
        state_name="state1",
        structure_kind="target",
        pdb_path=tmp_path / "template.pdb",
        mmcif_path=tmp_path / "template.cif",
        query_indices=(0, 1),
        template_indices=(0, 1),
        ca_coords=((2.0, 0.0, 0.0), (6.0, 0.0, 0.0)),
    )
    job = _target_job(tmp_path, output_dir)
    run_result = AF3RunResult(
        job_name=job.job_name,
        status="completed",
        returncode=0,
        stdout_path=output_dir / "stdout.txt",
        stderr_path=output_dir / "stderr.txt",
        output_dir=output_dir,
    )

    records = score_jobs([job], {("state1", "target"): template}, {job.job_name: run_result})
    record = records[0]

    assert record.af3_status == "completed"
    assert record.tm_score is not None and abs(record.tm_score - 1.0) < 1e-6
    assert record.rmsd is not None and abs(record.rmsd) < 1e-6
    assert record.mean_plddt == 90.0
    assert compute_tm_score(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ) == 1.0
    assert compute_rmsd(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ) == 0.0


def test_score_jobs_ignores_context_chains_when_extracting_metrics(tmp_path):
    output_dir = tmp_path / "af3_job_multi"
    output_dir.mkdir()
    predicted_path = output_dir / "prediction.pdb"
    confidence_path = output_dir / "sample_confidences.json"
    _write_pdb(
        predicted_path,
        (
            ("A", ("ALA", "CYS"), 0.0),
            ("B", ("GLY", "GLY", "GLY"), 100.0),
        ),
    )
    confidence_path.write_text(json.dumps({"plddt": [88.0, 92.0, 50.0, 50.0, 50.0]}), encoding="utf-8")

    template = TemplateArtifact(
        state_name="state1",
        structure_kind="target",
        pdb_path=tmp_path / "template.pdb",
        mmcif_path=tmp_path / "template.cif",
        query_indices=(0, 1),
        template_indices=(0, 1),
        ca_coords=((2.0, 0.0, 0.0), (6.0, 0.0, 0.0)),
    )
    job = _target_job(tmp_path, output_dir, include_context=True)
    run_result = AF3RunResult(
        job_name=job.job_name,
        status="completed",
        returncode=0,
        stdout_path=output_dir / "stdout.txt",
        stderr_path=output_dir / "stderr.txt",
        output_dir=output_dir,
    )

    record = score_jobs([job], {("state1", "target"): template}, {job.job_name: run_result})[0]

    assert record.af3_status == "completed"
    assert record.tm_score is not None and abs(record.tm_score - 1.0) < 1e-6
    assert record.rmsd is not None and abs(record.rmsd) < 1e-6
