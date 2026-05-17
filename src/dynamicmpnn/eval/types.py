from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class StateInput:
    name: str
    pdb_path: Path
    chain_id: str

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pdb_path"] = str(self.pdb_path)
        return payload


@dataclass(frozen=True)
class DecoyInput(StateInput):
    pass


@dataclass(frozen=True)
class AlignmentSpec:
    state1_aligned: str
    state2_aligned: str
    query_length: int
    state1_query_indices: tuple[int, ...]
    state2_query_indices: tuple[int, ...]
    state1_template_indices: tuple[int, ...]
    state2_template_indices: tuple[int, ...]
    paired_query_indices: tuple[int, ...]
    paired_state1_template_indices: tuple[int, ...]
    paired_state2_template_indices: tuple[int, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "state1_aligned": self.state1_aligned,
            "state2_aligned": self.state2_aligned,
            "query_length": self.query_length,
            "state1_query_indices": list(self.state1_query_indices),
            "state2_query_indices": list(self.state2_query_indices),
            "state1_template_indices": list(self.state1_template_indices),
            "state2_template_indices": list(self.state2_template_indices),
            "paired_query_indices": list(self.paired_query_indices),
            "paired_state1_template_indices": list(self.paired_state1_template_indices),
            "paired_state2_template_indices": list(self.paired_state2_template_indices),
        }


@dataclass(frozen=True)
class EvaluationRequest:
    model_ref: str
    resolved_model_path: Path
    num_samples: int
    device: str
    sampling_mode: str
    refold_mode: str
    targets: tuple[StateInput, StateInput]
    alignment: AlignmentSpec
    decoys: Optional[dict[str, DecoyInput]]
    af3_evaluate: bool
    af3_executable: Optional[str]
    af3_script: Optional[str]
    af3_model_dir: Optional[str]
    af3_db_dir: Optional[str]
    af3_env_script: Optional[str]
    af3_env: dict[str, str]
    af3_model_seeds: tuple[int, ...]
    output_dir: Path
    sequence_recovery_target_sequences: tuple[str, ...] = field(default_factory=tuple)
    af3_use_msa: bool = False
    af3_unpaired_msa: Optional[str] = None
    af3_unpaired_msa_path: Optional[str] = None
    af3_paired_msa: Optional[str] = None
    af3_paired_msa_path: Optional[str] = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "resolved_model_path": str(self.resolved_model_path),
            "num_samples": self.num_samples,
            "device": self.device,
            "sampling_mode": self.sampling_mode,
            "refold_mode": self.refold_mode,
            "targets": [target.to_manifest() for target in self.targets],
            "alignment": self.alignment.to_manifest(),
            "decoys": None
            if self.decoys is None
            else {
                state_name: decoy.to_manifest()
                for state_name, decoy in self.decoys.items()
            },
            "af3_evaluate": self.af3_evaluate,
            "af3_executable": self.af3_executable,
            "af3_script": self.af3_script,
            "af3_model_dir": self.af3_model_dir,
            "af3_db_dir": self.af3_db_dir,
            "af3_env_script": self.af3_env_script,
            "af3_env": dict(self.af3_env),
            "af3_model_seeds": list(self.af3_model_seeds),
            "sequence_recovery_target_sequences": list(self.sequence_recovery_target_sequences),
            "af3_use_msa": self.af3_use_msa,
            "af3_unpaired_msa": self.af3_unpaired_msa,
            "af3_unpaired_msa_path": self.af3_unpaired_msa_path,
            "af3_paired_msa": self.af3_paired_msa,
            "af3_paired_msa_path": self.af3_paired_msa_path,
            "output_dir": str(self.output_dir),
        }


@dataclass(frozen=True)
class AF3ProteinSpec:
    entity_id: str
    sequence: str
    template_mmcif: Optional[Path] = None
    query_indices: tuple[int, ...] = field(default_factory=tuple)
    template_indices: tuple[int, ...] = field(default_factory=tuple)
    use_msa: bool = False
    unpaired_msa: Optional[str] = None
    unpaired_msa_path: Optional[str] = None
    paired_msa: Optional[str] = None
    paired_msa_path: Optional[str] = None


@dataclass(frozen=True)
class AF3JobSpec:
    job_name: str
    sequence_id: str
    state_name: str
    structure_kind: str
    input_dir: Path
    json_path: Path
    output_dir: Path
    proteins: tuple[AF3ProteinSpec, ...]
    model_seeds: tuple[int, ...]
    predicted_target_chain_id: str = "A"


@dataclass(frozen=True)
class RawMetricRecord:
    sequence_id: str
    state_name: str
    structure_kind: str
    af3_status: str
    predicted_structure_path: Optional[Path] = None
    confidences_path: Optional[Path] = None
    tm_score: Optional[float] = None
    lddt: Optional[float] = None
    rmsd: Optional[float] = None
    mean_plddt: Optional[float] = None
    error: Optional[str] = None
    seed: Optional[int] = None
    af3_sample: Optional[int] = None
    ranking_score: Optional[float] = None


@dataclass(frozen=True)
class NormalizedMetricRecord:
    sequence_id: str
    state_name: str
    metric_name: str
    raw_target: Optional[float]
    raw_decoy: Optional[float]
    normalized_value: Optional[float]


@dataclass(frozen=True)
class SequenceSummary:
    sequence_id: str
    sequence: str
    af3_status: str
    metrics: dict[str, Any] = field(default_factory=dict)
