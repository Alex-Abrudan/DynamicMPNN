from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from loguru import logger

from dynamicmpnn.eval.guesses import GuessArtifact
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.types import AF2IGJobSpec


@dataclass(frozen=True)
class AF2IGRunResult:
    job_name: str
    status: str
    returncode: int | None
    stdout_path: Path
    stderr_path: Path
    output_dir: Path
    error: str | None = None


def build_af2ig_jobs(
    samples: Sequence[SampleRecord],
    guesses: dict[tuple[str, str], GuessArtifact],
    raw_dir: Path,
    *,
    num_recycles: int = 1,
    template_backbone: bool = True,
) -> tuple[AF2IGJobSpec, ...]:
    """One job per (sequence, state, structure kind), as build_af3_jobs.

    The structure kinds come from the guess artifacts rather than a setting,
    so a config with decoys produces a decoy arm here for the same reason it
    does for AF3, and the decoy-normalised columns downstream need no special
    casing.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for sample in samples:
        for (state_name, structure_kind), guess in guesses.items():
            job_name = f"{sample.sequence_id}__{state_name}__{structure_kind}"
            jobs.append(
                AF2IGJobSpec(
                    job_name=job_name,
                    sequence_id=sample.sequence_id,
                    state_name=state_name,
                    structure_kind=structure_kind,
                    sequence=sample.sequence,
                    guess_npz=guess.npz_path,
                    query_indices=guess.query_indices,
                    template_indices=guess.template_indices,
                    num_recycles=num_recycles,
                    template_backbone=template_backbone,
                    output_dir=raw_dir / job_name,
                )
            )
    return tuple(jobs)


def write_af2ig_manifest(jobs: Sequence[AF2IGJobSpec], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": [job.to_manifest() for job in jobs]}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


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


def run_af2ig_jobs(
    executable: str,
    jobs: Sequence[AF2IGJobSpec],
    *,
    script: str,
    params_dir: str,
    initial_guess_dir: str | None = None,
    env_script: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, AF2IGRunResult]:
    """Fold every job in one subprocess.

    AlphaFold2 needs a jax old enough to conflict with this package's torch
    stack, so the predictor lives behind `executable` in its own environment,
    the same arrangement `run_af3_jobs` uses for AF3. The whole batch goes in
    one call because loading the parameters and tracing the model costs far
    more than a single prediction.
    """
    if not jobs:
        return {}

    command_env = os.environ.copy()
    if env is not None:
        command_env.update({key: str(value) for key, value in env.items()})
    if initial_guess_dir is not None:
        command_env["AF2IG_INITIAL_GUESS_DIR"] = str(initial_guess_dir)

    batch_output_dir = jobs[0].output_dir.parent
    if any(job.output_dir.parent != batch_output_dir for job in jobs):
        raise ValueError("AF2 initial-guess batching requires a shared output root")
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = write_af2ig_manifest(jobs, batch_output_dir.parent / "manifest.json")
    command = [
        *shlex.split(executable),
        script,
        f"--manifest={manifest_path}",
        f"--params_dir={params_dir}",
        f"--output_dir={batch_output_dir}",
    ]

    results: dict[str, AF2IGRunResult] = {}
    try:
        completed = _run_command(command, command_env, env_script)
        status = "completed" if completed.returncode == 0 else "failed"
        error = None if completed.returncode == 0 else f"AF2 subprocess returned {completed.returncode}"
        if status == "failed":
            logger.warning("AF2 initial-guess batch failed: {}", (completed.stderr or "")[-2000:])
        stdout_text, stderr_text, returncode = completed.stdout or "", completed.stderr or "", completed.returncode
    except Exception as exc:  # noqa: BLE001 - recorded per job, as the AF3 path does
        status, error = "failed", str(exc)
        stdout_text, stderr_text, returncode = "", str(exc), None

    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job.output_dir / "stdout.txt"
        stderr_path = job.output_dir / "stderr.txt"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        # A batch that exited cleanly can still have skipped one job, so the
        # per-job verdict follows the artifact on disk rather than the
        # process exit code alone.
        job_status = status
        job_error = error
        if job_status == "completed" and not (job.output_dir / "prediction.pdb").exists():
            job_status = "failed"
            job_error = "AF2 initial-guess produced no prediction for this job"
        results[job.job_name] = AF2IGRunResult(
            job_name=job.job_name,
            status=job_status,
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_dir=job.output_dir,
            error=job_error,
        )
    return results
