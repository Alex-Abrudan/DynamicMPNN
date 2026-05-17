from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dynamicmpnn import constants
from dynamicmpnn.eval.af3 import build_af3_jobs, run_af3_jobs, write_af3_jsons
from dynamicmpnn.eval.aggregation import build_all_af3_results_dataframe, build_per_sequence_dataframe, write_reports
from dynamicmpnn.eval.inputs import resolve_inputs
from dynamicmpnn.eval.sampling import sample_sequences, write_sample_artifacts
from dynamicmpnn.eval.scoring import score_jobs
from dynamicmpnn.eval.templates import prepare_templates
from dynamicmpnn.eval.types import AlignmentSpec, DecoyInput, EvaluationRequest, StateInput


_ALIGNMENT_RANGE_PATTERN = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PACKAGE_PATH_PREFIX = "package://"
_VALID_EVAL_MODES = {"single", "multi"}


def _resolve_model_path(cfg: DictConfig) -> Path:
    model_ref = str(cfg.eval.model_ref)
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate.resolve()

    registry = cfg.eval.get("model_registry")
    if registry and model_ref in registry:
        registry_path = Path(str(registry[model_ref])).expanduser()
        if registry_path.exists():
            return registry_path.resolve()
        raise FileNotFoundError(f"Model registry entry '{model_ref}' resolved to a missing path: {registry_path}")

    raise FileNotFoundError(
        f"Could not resolve eval.model_ref='{model_ref}' as a path or model_registry entry"
    )


def _parse_alignment_indices(value: object) -> tuple[int, ...]:
    if value is None:
        return tuple()

    if isinstance(value, str):
        match = _ALIGNMENT_RANGE_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "Alignment ranges must use the form '(start, end)' with 1-indexed inclusive bounds"
            )
        start = int(match.group(1))
        end = int(match.group(2))
        if start <= 0 or end <= 0:
            raise ValueError("Alignment indices must be 1-indexed positive integers")
        if end < start:
            raise ValueError("Alignment range end must be greater than or equal to the start")
        return tuple(range(start - 1, end))

    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        indices = tuple(int(item) for item in value)
        if not indices:
            raise ValueError("Alignment index lists cannot be empty")
        if any(index <= 0 for index in indices):
            raise ValueError("Alignment indices must be 1-indexed positive integers")
        return tuple(index - 1 for index in indices)

    raise ValueError("Alignment entries must be either a 1-indexed list or a '(start, end)' range")


def _build_alignment(cfg: DictConfig) -> AlignmentSpec:
    alignment_cfg = cfg.eval.get("alignment")
    if alignment_cfg is None:
        state1_indices = tuple()
        state2_indices = tuple()
    else:
        if alignment_cfg.get("state1_aligned") is not None or alignment_cfg.get("state2_aligned") is not None:
            raise ValueError("eval.alignment now accepts AF3-style indices only: use eval.alignment.state1/state2")
        state1_indices = _parse_alignment_indices(alignment_cfg.get("state1"))
        state2_indices = _parse_alignment_indices(alignment_cfg.get("state2"))

    return AlignmentSpec(
        state1_aligned="",
        state2_aligned="",
        query_length=0,
        state1_query_indices=tuple(),
        state2_query_indices=tuple(),
        state1_template_indices=state1_indices,
        state2_template_indices=state2_indices,
        paired_query_indices=tuple(),
        paired_state1_template_indices=tuple(),
        paired_state2_template_indices=tuple(),
    )


def _build_sequence_recovery_targets(cfg: DictConfig) -> tuple[str, ...]:
    recovery_cfg = cfg.eval.get("sequence_recovery")
    if not recovery_cfg:
        return tuple()

    target_sequences = recovery_cfg.get("target_sequences")
    if target_sequences is None:
        return tuple()
    if isinstance(target_sequences, str):
        return (target_sequences,)
    if OmegaConf.is_list(target_sequences) or isinstance(target_sequences, (list, tuple)):
        return tuple(str(sequence) for sequence in target_sequences)

    raise ValueError("eval.sequence_recovery.target_sequences must be a string or a list of strings")


def _optional_inline(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_path(value: object) -> str | None:
    if value is None:
        return None
    path_text = str(value)
    if path_text == "":
        return None
    return str(_resolve_input_path(path_text))


def _optional_env_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if OmegaConf.is_dict(value) or isinstance(value, dict):
        return {str(key): str(val) for key, val in dict(value).items()}
    raise ValueError("eval.af3.env must be a mapping of environment variable names to string values")


def _resolve_input_path(path_value: object) -> Path:
    path_text = str(path_value)
    if path_text.startswith(_PACKAGE_PATH_PREFIX):
        relative_path = path_text.removeprefix(_PACKAGE_PATH_PREFIX).lstrip("/")
        candidate = constants.PACKAGE_RESOURCE_ROOT / relative_path
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Packaged input path '{path_text}' resolved to a missing file: {candidate}"
        )
    return Path(path_text).expanduser()


def _validate_single_source(inline_value: str | None, path_value: str | None, inline_name: str, path_name: str) -> None:
    if inline_value is not None and path_value is not None:
        raise ValueError(f"{inline_name} and {path_name} are mutually exclusive")


def _build_af3_settings(cfg: DictConfig) -> dict[str, object]:
    af3_cfg = cfg.eval.get("af3")
    if not af3_cfg:
        return {
            "executable": None,
            "script": None,
            "model_dir": None,
            "db_dir": None,
            "env_script": None,
            "env": {},
            "model_seeds": (0, 1, 2),
            "use_msa": False,
            "unpaired_msa": None,
            "unpaired_msa_path": None,
            "paired_msa": None,
            "paired_msa_path": None,
        }

    executable = _optional_inline(af3_cfg.get("executable"))
    script = _optional_path(af3_cfg.get("script"))
    model_dir = _optional_path(af3_cfg.get("model_dir"))
    db_dir = _optional_path(af3_cfg.get("db_dir"))
    env_script = _optional_path(af3_cfg.get("env_script"))
    env = _optional_env_map(af3_cfg.get("env"))
    unpaired_msa = _optional_inline(af3_cfg.get("unpaired_msa"))
    unpaired_msa_path = _optional_path(af3_cfg.get("unpaired_msa_path"))
    paired_msa = _optional_inline(af3_cfg.get("paired_msa"))
    paired_msa_path = _optional_path(af3_cfg.get("paired_msa_path"))

    _validate_single_source(
        unpaired_msa,
        unpaired_msa_path,
        "eval.af3.unpaired_msa",
        "eval.af3.unpaired_msa_path",
    )
    _validate_single_source(
        paired_msa,
        paired_msa_path,
        "eval.af3.paired_msa",
        "eval.af3.paired_msa_path",
    )

    use_msa = bool(af3_cfg.get("use_msa", False))
    has_custom_msa = any(
        (
            unpaired_msa not in (None, ""),
            unpaired_msa_path is not None,
            paired_msa not in (None, ""),
            paired_msa_path is not None,
        )
    )
    if has_custom_msa and not use_msa:
        raise ValueError("Custom AF3 MSA input requires eval.af3.use_msa=true")

    native_cli_values = (script, model_dir, db_dir)
    native_cli_enabled = any(value is not None for value in native_cli_values)
    if native_cli_enabled and (script is None or model_dir is None):
        raise ValueError("eval.af3.script and eval.af3.model_dir must be set together")
    if native_cli_enabled and use_msa and db_dir is None:
        raise ValueError("eval.af3.db_dir is required when eval.af3.use_msa=true")

    return {
        "executable": executable,
        "script": script,
        "model_dir": model_dir,
        "db_dir": db_dir,
        "env_script": env_script,
        "env": env,
        "model_seeds": tuple(int(seed) for seed in af3_cfg.get("model_seeds", [0, 1, 2])),
        "use_msa": use_msa,
        "unpaired_msa": unpaired_msa,
        "unpaired_msa_path": unpaired_msa_path,
        "paired_msa": paired_msa,
        "paired_msa_path": paired_msa_path,
    }


def _resolve_eval_mode(cfg: DictConfig, key: str, default: str = "multi") -> str:
    value = str(cfg.eval.get(key, default)).strip().lower()
    if value not in _VALID_EVAL_MODES:
        expected = ", ".join(sorted(_VALID_EVAL_MODES))
        raise ValueError(f"eval.{key} must be one of: {expected}")
    return value


def build_request(cfg: DictConfig) -> EvaluationRequest:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = (
        StateInput(
            name="state1",
            pdb_path=_resolve_input_path(cfg.eval.targets.state1.pdb_path),
            chain_id=str(cfg.eval.targets.state1.chain_id),
        ),
        StateInput(
            name="state2",
            pdb_path=_resolve_input_path(cfg.eval.targets.state2.pdb_path),
            chain_id=str(cfg.eval.targets.state2.chain_id),
        ),
    )

    decoys = None
    if cfg.eval.get("decoys"):
        decoys = {}
        for state_name in ("state1", "state2"):
            state_cfg = cfg.eval.decoys.get(state_name)
            if not state_cfg:
                continue
            decoys[state_name] = DecoyInput(
                name=f"{state_name}_decoy",
                pdb_path=_resolve_input_path(state_cfg.pdb_path),
                chain_id=str(state_cfg.chain_id),
            )
        if not decoys:
            decoys = None

    af3_settings = _build_af3_settings(cfg)
    af3_evaluate = bool(cfg.eval.get("af3_evaluate", False))
    af3_executable = af3_settings["executable"]
    if af3_evaluate and not af3_executable:
        raise ValueError("eval.af3.executable is required when eval.af3_evaluate=true")

    return EvaluationRequest(
        model_ref=str(cfg.eval.model_ref),
        resolved_model_path=_resolve_model_path(cfg),
        num_samples=int(cfg.eval.get("num_samples", 25)),
        device=str(cfg.eval.get("device", "auto")),
        sampling_mode=_resolve_eval_mode(cfg, "sampling_mode"),
        refold_mode=_resolve_eval_mode(cfg, "refold_mode"),
        targets=targets,
        alignment=_build_alignment(cfg),
        decoys=decoys,
        af3_evaluate=af3_evaluate,
        af3_executable=None if af3_executable is None else str(af3_executable),
        af3_script=af3_settings["script"],
        af3_model_dir=af3_settings["model_dir"],
        af3_db_dir=af3_settings["db_dir"],
        af3_env_script=af3_settings["env_script"],
        af3_env=af3_settings["env"],
        af3_model_seeds=af3_settings["model_seeds"],
        output_dir=output_dir,
        sequence_recovery_target_sequences=_build_sequence_recovery_targets(cfg),
        af3_use_msa=bool(af3_settings["use_msa"]),
        af3_unpaired_msa=af3_settings["unpaired_msa"],
        af3_unpaired_msa_path=af3_settings["unpaired_msa_path"],
        af3_paired_msa=af3_settings["paired_msa"],
        af3_paired_msa_path=af3_settings["paired_msa_path"],
    )


def write_manifest(request: EvaluationRequest, cfg: DictConfig) -> Path:
    manifest_path = request.output_dir / "manifest.json"
    payload = {
        "request": request.to_manifest(),
        "runtime": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hydra_output_dir": str(request.output_dir),
        },
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def run_evaluation(cfg: DictConfig) -> dict[str, str]:
    request = build_request(cfg)
    bundle = resolve_inputs(request)
    manifest_path = write_manifest(bundle.request, cfg)

    sample_result = sample_sequences(cfg, bundle)
    sample_paths = write_sample_artifacts(sample_result.samples, bundle.request.output_dir / "samples")

    raw_records = ()
    if bundle.request.af3_evaluate:
        templates = prepare_templates(bundle, bundle.request.output_dir / "templates")
        af3_jobs = build_af3_jobs(
            sample_result.samples,
            bundle,
            templates,
            bundle.request.output_dir / "af3" / "json",
            bundle.request.output_dir / "af3" / "raw",
            bundle.request.af3_model_seeds,
            refold_mode=bundle.request.refold_mode,
            use_msa=bundle.request.af3_use_msa,
            unpaired_msa=bundle.request.af3_unpaired_msa,
            unpaired_msa_path=bundle.request.af3_unpaired_msa_path,
            paired_msa=bundle.request.af3_paired_msa,
            paired_msa_path=bundle.request.af3_paired_msa_path,
        )
        write_af3_jsons(af3_jobs)
        run_results = run_af3_jobs(
            bundle.request.af3_executable,
            af3_jobs,
            script=bundle.request.af3_script,
            model_dir=bundle.request.af3_model_dir,
            db_dir=bundle.request.af3_db_dir,
            env_script=bundle.request.af3_env_script,
            env=bundle.request.af3_env,
        )
        raw_records = score_jobs(af3_jobs, templates, run_results)

    per_sequence_df = build_per_sequence_dataframe(
        sample_result.samples,
        raw_records,
        include_decoys=bool(bundle.decoys),
        af3_enabled=bundle.request.af3_evaluate,
        sequence_recovery_target_sequences=bundle.request.sequence_recovery_target_sequences,
    )
    all_af3_results_df = build_all_af3_results_dataframe(sample_result.samples, raw_records)
    report_paths = write_reports(all_af3_results_df, per_sequence_df, bundle.request.output_dir)

    output_paths = {
        "manifest": str(manifest_path),
        "output_dir": str(bundle.request.output_dir),
        **sample_paths,
        **report_paths,
    }
    logger.info("Evaluation artifacts: {}", output_paths)
    return output_paths
