from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from loguru import logger

from dynamicmpnn.eval.inputs import PreparedInputBundle, PreparedState
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec


@dataclass(frozen=True)
class AF3RunResult:
    job_name: str
    status: str
    returncode: int | None
    stdout_path: Path
    stderr_path: Path
    output_dir: Path
    error: str | None = None


_PREFOLD_WARNING_RESIDUE_THRESHOLD = 2048


def _build_msa_payload(protein: AF3ProteinSpec) -> dict[str, str]:
    if not protein.use_msa:
        return {
            "unpairedMsa": "",
            "pairedMsa": "",
        }

    payload: dict[str, str] = {}
    if protein.unpaired_msa is not None:
        payload["unpairedMsa"] = protein.unpaired_msa
    elif protein.unpaired_msa_path is not None:
        payload["unpairedMsaPath"] = protein.unpaired_msa_path

    if protein.paired_msa is not None:
        payload["pairedMsa"] = protein.paired_msa
    elif protein.paired_msa_path is not None:
        payload["pairedMsaPath"] = protein.paired_msa_path

    has_unpaired = "unpairedMsa" in payload or "unpairedMsaPath" in payload
    has_paired = "pairedMsa" in payload or "pairedMsaPath" in payload

    if has_unpaired and not has_paired:
        payload["pairedMsa"] = ""
    if has_paired and not has_unpaired:
        payload["unpairedMsa"] = ""

    return payload


def build_af3_payload(job: AF3JobSpec) -> dict:
    run_data_pipeline = _job_requires_data_pipeline(job)
    sequences = []
    for protein in job.proteins:
        protein_payload = {
            "id": protein.entity_id,
            "sequence": protein.sequence,
            **_build_msa_payload(protein),
        }
        if protein.template_mmcif is not None:
            protein_payload["templates"] = [
                {
                    "mmcifPath": str(protein.template_mmcif),
                    "queryIndices": list(protein.query_indices),
                    "templateIndices": list(protein.template_indices),
                }
            ]
        elif not run_data_pipeline:
            # AF3 requires explicit empty templates when the data pipeline is skipped.
            protein_payload["templates"] = []
        sequences.append({"protein": protein_payload})
    return {
        "name": job.job_name,
        "modelSeeds": list(job.model_seeds),
        "dialect": "alphafold3",
        "version": 4,
        "sequences": sequences,
    }


def _job_requires_data_pipeline(job: AF3JobSpec) -> bool:
    for protein in job.proteins:
        if not protein.use_msa:
            continue
        if all(
            value is None
            for value in (
                protein.unpaired_msa,
                protein.unpaired_msa_path,
                protein.paired_msa,
                protein.paired_msa_path,
            )
        ):
            return True
    return False


def _af3_chain_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _build_target_protein(
    sample: SampleRecord,
    template: TemplateArtifact,
    *,
    use_msa: bool,
    unpaired_msa: str | None,
    unpaired_msa_path: str | None,
    paired_msa: str | None,
    paired_msa_path: str | None,
) -> AF3ProteinSpec:
    return AF3ProteinSpec(
        entity_id="A",
        sequence=sample.sequence,
        template_mmcif=template.mmcif_path,
        query_indices=template.query_indices,
        template_indices=template.template_indices,
        use_msa=use_msa,
        unpaired_msa=unpaired_msa,
        unpaired_msa_path=unpaired_msa_path,
        paired_msa=paired_msa,
        paired_msa_path=paired_msa_path,
    )


def _build_context_proteins(
    state: PreparedState,
    *,
    use_msa: bool,
    has_custom_msa: bool,
) -> tuple[AF3ProteinSpec, ...]:
    context_proteins = []
    for context_index, context_chain in enumerate(state.context_chains, start=1):
        context_proteins.append(
            AF3ProteinSpec(
                entity_id=_af3_chain_label(context_index),
                sequence=context_chain.sequence,
                use_msa=use_msa and not has_custom_msa,
            )
    )
    return tuple(context_proteins)


def _warn_if_large_prefold(job_name: str, proteins: Sequence[AF3ProteinSpec]) -> None:
    total_residues = sum(len(protein.sequence) for protein in proteins)
    if total_residues <= _PREFOLD_WARNING_RESIDUE_THRESHOLD:
        return
    logger.warning(
        "AF3 prefold job {} contains {} residues across {} chains; inputs above {} residues often OOM.",
        job_name,
        total_residues,
        len(proteins),
        _PREFOLD_WARNING_RESIDUE_THRESHOLD,
    )


def build_af3_jobs(
    samples: Sequence[SampleRecord],
    bundle: PreparedInputBundle,
    templates: dict[tuple[str, str], TemplateArtifact],
    json_dir: Path,
    raw_dir: Path,
    model_seeds: Sequence[int],
    refold_mode: str = "single",
    use_msa: bool = False,
    unpaired_msa: str | None = None,
    unpaired_msa_path: str | None = None,
    paired_msa: str | None = None,
    paired_msa_path: str | None = None,
) -> tuple[AF3JobSpec, ...]:
    jobs = []
    json_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    has_custom_msa = any(
        value is not None
        for value in (unpaired_msa, unpaired_msa_path, paired_msa, paired_msa_path)
    )
    state_lookup: dict[tuple[str, str], PreparedState] = {
        (state.name, "target"): state for state in bundle.targets
    }
    if bundle.decoys is not None:
        state_lookup.update(
            {
                (state_name, "decoy"): state
                for state_name, state in bundle.decoys.items()
            }
        )

    for sample in samples:
        for (state_name, structure_kind), template in templates.items():
            job_name = f"{sample.sequence_id}__{state_name}__{structure_kind}"
            proteins = [
                _build_target_protein(
                    sample,
                    template,
                    use_msa=use_msa,
                    unpaired_msa=unpaired_msa,
                    unpaired_msa_path=unpaired_msa_path,
                    paired_msa=paired_msa,
                    paired_msa_path=paired_msa_path,
                )
            ]
            if refold_mode == "multi":
                proteins.extend(
                    _build_context_proteins(
                        state_lookup[(state_name, structure_kind)],
                        use_msa=use_msa,
                        has_custom_msa=has_custom_msa,
                    )
                )
            _warn_if_large_prefold(job_name, proteins)
            jobs.append(
                AF3JobSpec(
                    job_name=job_name,
                    sequence_id=sample.sequence_id,
                    state_name=state_name,
                    structure_kind=structure_kind,
                    input_dir=json_dir,
                    json_path=json_dir / f"{job_name}.json",
                    output_dir=raw_dir / job_name,
                    proteins=tuple(proteins),
                    model_seeds=tuple(int(seed) for seed in model_seeds),
                )
            )

    return tuple(jobs)


def write_af3_jsons(jobs: Sequence[AF3JobSpec]) -> None:
    for job in jobs:
        job.input_dir.mkdir(parents=True, exist_ok=True)
        payload = build_af3_payload(job)
        job.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_legacy_command(executable: str, job: AF3JobSpec) -> list[str]:
    return [*shlex.split(executable), str(job.json_path), str(job.output_dir)]


def _run_command(
    command: list[str],
    env: Mapping[str, str],
    env_script: str | None,
) -> subprocess.CompletedProcess[str]:
    if env_script is None:
        return subprocess.run(command, capture_output=True, text=True, check=False, env=dict(env))

    shell_command = f"source {shlex.quote(env_script)} && " + " ".join(
        shlex.quote(part) for part in command
    )
    return subprocess.run(
        ["bash", "-lc", shell_command],
        capture_output=True,
        text=True,
        check=False,
        env=dict(env),
    )


def _run_af3_batch(
    executable: str,
    jobs: Sequence[AF3JobSpec],
    script: str,
    model_dir: str,
    db_dir: str | None,
    command_env: Mapping[str, str],
    env_script: str | None,
) -> dict[str, AF3RunResult]:
    batch_input_dir = jobs[0].input_dir
    batch_output_dir = jobs[0].output_dir.parent
    run_data_pipeline = _job_requires_data_pipeline(jobs[0])
    if any(job.input_dir != batch_input_dir for job in jobs):
        raise ValueError("Native AF3 batching requires a shared input directory")
    if any(job.output_dir.parent != batch_output_dir for job in jobs):
        raise ValueError("Native AF3 batching requires a shared output root")
    if any(_job_requires_data_pipeline(job) != run_data_pipeline for job in jobs):
        raise ValueError("Native AF3 batching requires a shared data-pipeline mode")
    if run_data_pipeline and db_dir is None:
        raise ValueError("AF3 native CLI mode requires db_dir when the data pipeline is enabled")
    batch_input_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *shlex.split(executable),
        script,
        f"--input_dir={batch_input_dir}",
        f"--model_dir={model_dir}",
        f"--output_dir={batch_output_dir}",
        f"--run_data_pipeline={'true' if run_data_pipeline else 'false'}",
    ]
    if db_dir is not None:
        command.insert(4, f"--db_dir={db_dir}")

    results: dict[str, AF3RunResult] = {}
    try:
        completed = _run_command(command, command_env, env_script)
        status = "completed" if completed.returncode == 0 else "failed"
        error = None if completed.returncode == 0 else f"AF3 subprocess returned {completed.returncode}"
        for job in jobs:
            job_output_dir = batch_output_dir / job.job_name
            job_output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = job_output_dir / "stdout.txt"
            stderr_path = job_output_dir / "stderr.txt"
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            results[job.job_name] = AF3RunResult(
                job_name=job.job_name,
                status=status,
                returncode=completed.returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_dir=job_output_dir,
                error=error,
            )
    except Exception as exc:
        for job in jobs:
            job_output_dir = batch_output_dir / job.job_name
            job_output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = job_output_dir / "stdout.txt"
            stderr_path = job_output_dir / "stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            results[job.job_name] = AF3RunResult(
                job_name=job.job_name,
                status="failed",
                returncode=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_dir=job_output_dir,
                error=str(exc),
            )
    return results


def run_af3_jobs(
    executable: str,
    jobs: Sequence[AF3JobSpec],
    *,
    script: str | None = None,
    model_dir: str | None = None,
    db_dir: str | None = None,
    env_script: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, AF3RunResult]:
    results: dict[str, AF3RunResult] = {}
    command_env = os.environ.copy()
    if env is not None:
        command_env.update({key: str(value) for key, value in env.items()})
    use_native_cli = any(value is not None for value in (script, model_dir, db_dir))

    if use_native_cli:
        if script is None or model_dir is None:
            raise ValueError("AF3 native CLI mode requires script and model_dir")
        results.update(
            _run_af3_batch(
                executable,
                jobs,
                script,
                model_dir,
                db_dir,
                command_env,
                env_script,
            )
        )
        return results

    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job.output_dir / "stdout.txt"
        stderr_path = job.output_dir / "stderr.txt"
        command = _build_legacy_command(executable, job)

        try:
            completed = _run_command(command, command_env, env_script)
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            status = "completed" if completed.returncode == 0 else "failed"
            error = None if completed.returncode == 0 else f"AF3 subprocess returned {completed.returncode}"
            results[job.job_name] = AF3RunResult(
                job_name=job.job_name,
                status=status,
                returncode=completed.returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_dir=job.output_dir,
                error=error,
            )
        except Exception as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            results[job.job_name] = AF3RunResult(
                job_name=job.job_name,
                status="failed",
                returncode=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_dir=job.output_dir,
                error=str(exc),
            )

    return results
