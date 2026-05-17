import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from torch_geometric.data import Data

from dynamicmpnn.eval.af3 import build_af3_jobs, build_af3_payload, run_af3_jobs, write_af3_jsons
from dynamicmpnn.eval.inputs import PreparedInputBundle, PreparedState
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.structure_utils import ChainRecord
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AlignmentSpec, DecoyInput, EvaluationRequest, StateInput


_RESIDUE_NAMES = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "G": "GLY",
    "S": "SER",
}


def _coords(length: int) -> np.ndarray:
    rows = []
    for residue_index in range(length):
        base_x = float(residue_index * 4)
        rows.append(
            [
                [base_x + 1.0, 0.0, 0.0],
                [base_x + 2.0, 0.0, 0.0],
                [base_x + 3.0, 0.0, 0.0],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _chain_record(chain_id: str, sequence: str) -> ChainRecord:
    return ChainRecord(
        chain_id=chain_id,
        sequence=sequence,
        residues_3=tuple(_RESIDUE_NAMES[aa] for aa in sequence),
        residue_numbers=tuple(range(1, len(sequence) + 1)),
        residue_insertions=tuple("" for _ in sequence),
        coords=_coords(len(sequence)),
    )


def _prepared_state(tmp_path: Path, name: str, *, context_sequences: tuple[str, ...] = ("GS",)) -> PreparedState:
    target_chain = _chain_record("A", "ACD")
    return PreparedState(
        name=name,
        pdb_path=tmp_path / f"{name}.pdb",
        chain_id="A",
        sequence=target_chain.sequence,
        aligned_sequence=target_chain.sequence,
        residues_3=target_chain.residues_3,
        residue_numbers=target_chain.residue_numbers,
        residue_insertions=target_chain.residue_insertions,
        coords=target_chain.coords,
        query_indices=(0, 1, 2),
        template_indices=(0, 1, 2),
        context_chains=tuple(
            _chain_record(chr(ord("B") + index), sequence)
            for index, sequence in enumerate(context_sequences)
        ),
        pyg_data=None,
    )


def _request(tmp_path: Path) -> EvaluationRequest:
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")
    return EvaluationRequest(
        model_ref=str(model_path),
        resolved_model_path=model_path,
        num_samples=1,
        device="cpu",
        sampling_mode="multi",
        refold_mode="multi",
        targets=(
            StateInput(name="state1", pdb_path=tmp_path / "state1.pdb", chain_id="A"),
            StateInput(name="state2", pdb_path=tmp_path / "state2.pdb", chain_id="A"),
        ),
        alignment=AlignmentSpec(
            state1_aligned="ACD",
            state2_aligned="ACD",
            query_length=3,
            state1_query_indices=(0, 1, 2),
            state2_query_indices=(0, 1, 2),
            state1_template_indices=(0, 1, 2),
            state2_template_indices=(0, 1, 2),
            paired_query_indices=(0, 1, 2),
            paired_state1_template_indices=(0, 1, 2),
            paired_state2_template_indices=(0, 1, 2),
        ),
        decoys={
            "state1": DecoyInput(name="state1_decoy", pdb_path=tmp_path / "state1_decoy.pdb", chain_id="A"),
            "state2": DecoyInput(name="state2_decoy", pdb_path=tmp_path / "state2_decoy.pdb", chain_id="A"),
        },
        af3_evaluate=True,
        af3_executable="/bin/true",
        af3_script=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_env_script=None,
        af3_env={},
        af3_model_seeds=(0,),
        output_dir=tmp_path / "out",
    )


def _bundle(tmp_path: Path, *, context_sequences: tuple[str, ...] = ("GS",)) -> PreparedInputBundle:
    return PreparedInputBundle(
        request=_request(tmp_path),
        targets=(
            _prepared_state(tmp_path, "state1", context_sequences=context_sequences),
            _prepared_state(tmp_path, "state2", context_sequences=context_sequences),
        ),
        decoys={
            "state1": _prepared_state(tmp_path, "state1_decoy", context_sequences=context_sequences),
            "state2": _prepared_state(tmp_path, "state2_decoy", context_sequences=context_sequences),
        },
        multi_state_data=Data(cluster_members=["STATE1_A", "STATE2_A"]),
    )


def _artifact(tmp_path: Path, state_name: str, structure_kind: str) -> TemplateArtifact:
    template_path = tmp_path / f"{state_name}_{structure_kind}.cif"
    template_path.write_text("data_test\n#\n", encoding="utf-8")
    pdb_path = tmp_path / f"{state_name}_{structure_kind}.pdb"
    pdb_path.write_text("END\n", encoding="utf-8")
    return TemplateArtifact(
        state_name=state_name,
        structure_kind=structure_kind,
        pdb_path=pdb_path,
        mmcif_path=template_path,
        query_indices=(0, 1, 2),
        template_indices=(0, 1, 2),
        ca_coords=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )


def _templates(tmp_path: Path) -> dict[tuple[str, str], TemplateArtifact]:
    return {
        ("state1", "target"): _artifact(tmp_path, "state1", "target"),
        ("state2", "target"): _artifact(tmp_path, "state2", "target"),
        ("state1", "decoy"): _artifact(tmp_path, "state1", "decoy"),
        ("state2", "decoy"): _artifact(tmp_path, "state2", "decoy"),
    }


def test_af3_jobs_and_blank_msa_json_shape(tmp_path):
    samples = (SampleRecord(sequence_id="sample_000", sequence="ACD"),)
    jobs = build_af3_jobs(samples, _bundle(tmp_path), _templates(tmp_path), tmp_path / "json", tmp_path / "raw", [0, 1])
    write_af3_jsons(jobs)

    assert len(jobs) == 4
    payload = build_af3_payload(jobs[0])
    proteins = payload["sequences"]
    protein = proteins[0]["protein"]

    assert payload["name"] == jobs[0].job_name
    assert payload["modelSeeds"] == [0, 1]
    assert payload["dialect"] == "alphafold3"
    assert payload["version"] == 4
    assert len(proteins) == 1
    assert protein["sequence"] == "ACD"
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert jobs[0].json_path.parent == tmp_path / "json"
    assert protein["templates"][0]["queryIndices"] == [0, 1, 2]
    assert protein["templates"][0]["templateIndices"] == [0, 1, 2]

    json_payload = json.loads(jobs[0].json_path.read_text(encoding="utf-8"))
    assert json_payload == payload


def test_af3_jobs_support_partial_decoys(tmp_path):
    samples = (SampleRecord(sequence_id="sample_000", sequence="ACD"),)
    bundle = _bundle(tmp_path)
    bundle = PreparedInputBundle(
        request=bundle.request,
        targets=bundle.targets,
        decoys={"state1": bundle.decoys["state1"]},
        multi_state_data=bundle.multi_state_data,
    )
    templates = {
        ("state1", "target"): _artifact(tmp_path, "state1", "target"),
        ("state2", "target"): _artifact(tmp_path, "state2", "target"),
        ("state1", "decoy"): _artifact(tmp_path, "state1", "decoy"),
    }

    jobs = build_af3_jobs(samples, bundle, templates, tmp_path / "json", tmp_path / "raw", [0])

    assert [(job.state_name, job.structure_kind) for job in jobs] == [
        ("state1", "target"),
        ("state2", "target"),
        ("state1", "decoy"),
    ]


def test_af3_payload_omits_msa_fields_when_auto_mode_enabled(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        use_msa=True,
    )

    protein = build_af3_payload(jobs[0])["sequences"][0]["protein"]

    assert "unpairedMsa" not in protein
    assert "pairedMsa" not in protein
    assert "unpairedMsaPath" not in protein
    assert "pairedMsaPath" not in protein


def test_af3_payload_allows_custom_msa_inputs(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        use_msa=True,
        unpaired_msa=">query\nACD\n",
        paired_msa_path="paired.a3m",
    )

    protein = build_af3_payload(jobs[0])["sequences"][0]["protein"]

    assert protein["unpairedMsa"] == ">query\nACD\n"
    assert protein["pairedMsaPath"] == "paired.a3m"


def test_af3_payload_backfills_missing_partner_msa_with_blank_string(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        use_msa=True,
        unpaired_msa_path="unpaired.a3m",
    )

    protein = build_af3_payload(jobs[0])["sequences"][0]["protein"]

    assert protein["unpairedMsaPath"] == "unpaired.a3m"
    assert protein["pairedMsa"] == ""


def test_af3_payload_multi_mode_adds_untemplated_context_chains(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path, context_sequences=("GS",)),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        refold_mode="multi",
    )

    proteins = [entry["protein"] for entry in build_af3_payload(jobs[0])["sequences"]]

    assert [protein["id"] for protein in proteins] == ["A", "B"]
    assert "templates" in proteins[0]
    assert proteins[1]["templates"] == []
    assert proteins[1]["sequence"] == "GS"


def test_af3_payload_omits_empty_templates_when_data_pipeline_is_enabled(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path, context_sequences=("GS",)),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        refold_mode="multi",
        use_msa=True,
    )

    proteins = [entry["protein"] for entry in build_af3_payload(jobs[0])["sequences"]]

    assert [protein["id"] for protein in proteins] == ["A", "B"]
    assert "templates" in proteins[0]
    assert "templates" not in proteins[1]


def test_build_af3_jobs_warns_when_prefold_input_exceeds_2048_residues(tmp_path, monkeypatch):
    warnings = []

    def _capture_warning(message, *args):
        warnings.append(message.format(*args))

    monkeypatch.setattr("dynamicmpnn.eval.af3.logger.warning", _capture_warning)

    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path, context_sequences=("A" * 2050,)),
        {("state1", "target"): _artifact(tmp_path, "state1", "target")},
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        refold_mode="multi",
    )

    assert len(jobs) == 1
    assert len(warnings) == 1
    assert "sample_000__state1__target" in warnings[0]
    assert "2053 residues" in warnings[0]


def test_run_af3_jobs_uses_native_cli_without_data_pipeline_for_precomputed_inputs(tmp_path, monkeypatch):
    jobs = build_af3_jobs(
        (
            SampleRecord(sequence_id="sample_000", sequence="ACD"),
            SampleRecord(sequence_id="sample_001", sequence="ACE"),
        ),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
    )
    write_af3_jsons(jobs)
    captured: dict[str, object] = {"commands": []}

    def _fake_run(command, **kwargs):
        captured["commands"].append(command)
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("dynamicmpnn.eval.af3.subprocess.run", _fake_run)

    result = run_af3_jobs(
        "/path/to/python",
        jobs,
        script="/path/to/run_alphafold.py",
        model_dir="/path/to/models",
        env={"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"},
    )

    command = captured["commands"][0]

    assert result[jobs[0].job_name].status == "completed"
    assert len(captured["commands"]) == 1
    assert command[:2] == ["/path/to/python", "/path/to/run_alphafold.py"]
    assert f"--input_dir={tmp_path / 'json'}" in command
    assert f"--output_dir={tmp_path / 'raw'}" in command
    assert "--db_dir=/path/to/databases" not in command
    assert "--run_data_pipeline=false" in command
    assert captured["env"]["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"
    assert (tmp_path / "json" / "sample_000__state1__target.json").exists()
    assert (tmp_path / "json" / "sample_001__state2__decoy.json").exists()


def test_run_af3_jobs_enables_data_pipeline_when_af3_should_search_msa(tmp_path, monkeypatch):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        use_msa=True,
    )
    write_af3_jsons(jobs)
    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("dynamicmpnn.eval.af3.subprocess.run", _fake_run)

    run_af3_jobs(
        "/path/to/python",
        jobs[:1],
        script="/path/to/run_alphafold.py",
        model_dir="/path/to/models",
        db_dir="/path/to/databases",
        env_script="/path/to/af3_env.sh",
    )

    assert captured["command"][0:2] == ["bash", "-lc"]
    assert "source /path/to/af3_env.sh" in captured["command"][2]
    assert "--run_data_pipeline=true" in captured["command"][2]


def test_run_af3_jobs_requires_db_dir_when_data_pipeline_is_enabled(tmp_path):
    jobs = build_af3_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="ACD"),),
        _bundle(tmp_path),
        _templates(tmp_path),
        tmp_path / "json",
        tmp_path / "raw",
        [0],
        use_msa=True,
    )
    write_af3_jsons(jobs)

    with pytest.raises(ValueError, match="requires db_dir"):
        run_af3_jobs(
            "/path/to/python",
            jobs[:1],
            script="/path/to/run_alphafold.py",
            model_dir="/path/to/models",
        )
