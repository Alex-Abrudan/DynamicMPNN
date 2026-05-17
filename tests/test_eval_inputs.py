from pathlib import Path
import re

import pytest
from omegaconf import OmegaConf

from dynamicmpnn import constants
from dynamicmpnn.eval.inputs import resolve_alignment, resolve_inputs
from dynamicmpnn.eval.pipeline import build_request
from dynamicmpnn.eval.templates import prepare_templates
from dynamicmpnn.eval.types import AlignmentSpec, EvaluationRequest, StateInput


def _write_pdb(path: Path, residues: list[str] | dict[str, list[str]], chain_id: str = "A") -> None:
    chain_residues = residues if isinstance(residues, dict) else {chain_id: residues}
    atom_serial = 1
    residue_lines = []
    for current_chain_id, current_residues in chain_residues.items():
        for res_idx, resname in enumerate(current_residues, start=1):
            base_x = float(res_idx * 3)
            residue_lines.extend(
                [
                    f"ATOM  {atom_serial:5d}  N   {resname:>3} {current_chain_id}{res_idx:4d}    {base_x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           N\n",
                    f"ATOM  {atom_serial + 1:5d}  CA  {resname:>3} {current_chain_id}{res_idx:4d}    {base_x + 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n",
                    f"ATOM  {atom_serial + 2:5d}  C   {resname:>3} {current_chain_id}{res_idx:4d}    {base_x + 2.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n",
                ]
            )
            atom_serial += 3
        residue_lines.append("TER\n")
    residue_lines.append("END\n")
    path.write_text("".join(residue_lines), encoding="utf-8")


def _build_cfg(tmp_path: Path):
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path / "out"),
            "eval": {
                "model_ref": str(model_path),
                "model_registry": {},
                "num_samples": 2,
                "device": "cpu",
                "af3_evaluate": True,
                "targets": {
                    "state1": {"pdb_path": str(tmp_path / "state1.pdb"), "chain_id": "A"},
                    "state2": {"pdb_path": str(tmp_path / "state2.pdb"), "chain_id": "A"},
                },
                "alignment": {"state1": None, "state2": None},
                "decoys": {"state1": None, "state2": None},
                "af3": {
                    "executable": "/path/to/af3_runner",
                    "script": None,
                    "model_dir": None,
                    "db_dir": None,
                    "env_script": None,
                    "env": {},
                    "use_msa": False,
                    "unpaired_msa": None,
                    "unpaired_msa_path": None,
                    "paired_msa": None,
                    "paired_msa_path": None,
                    "model_seeds": [0, 1],
                },
            },
        }
    )


def test_resolve_alignment_maps_af3_style_indices():
    alignment = resolve_alignment("AC", "ADC", (0, 1), (0, 2))

    assert alignment.query_length == 2
    assert alignment.state1_aligned == "AC"
    assert alignment.state2_aligned == "AC"
    assert alignment.state1_query_indices == (0, 1)
    assert alignment.state2_query_indices == (0, 1)
    assert alignment.state1_template_indices == (0, 1)
    assert alignment.state2_template_indices == (0, 2)
    assert alignment.paired_query_indices == (0, 1)
    assert alignment.paired_state1_template_indices == (0, 1)
    assert alignment.paired_state2_template_indices == (0, 2)


def test_build_request_parses_alignment_range(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.alignment.state1 = "(1, 2)"
    cfg.eval.alignment.state2 = [1, 3]

    request = build_request(cfg)

    assert request.alignment.state1_template_indices == (0, 1)
    assert request.alignment.state2_template_indices == (0, 2)


def test_build_request_defaults_to_blank_af3_msa(tmp_path):
    request = build_request(_build_cfg(tmp_path))

    assert request.sampling_mode == "multi"
    assert request.refold_mode == "multi"
    assert request.af3_executable == "/path/to/af3_runner"
    assert request.af3_script is None
    assert request.af3_model_dir is None
    assert request.af3_db_dir is None
    assert request.af3_env_script is None
    assert request.af3_env == {}
    assert request.af3_use_msa is False
    assert request.af3_unpaired_msa is None
    assert request.af3_unpaired_msa_path is None
    assert request.af3_paired_msa is None
    assert request.af3_paired_msa_path is None


def test_build_request_allows_partial_decoys(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.decoys.state1 = {
        "pdb_path": str(tmp_path / "state2.pdb"),
        "chain_id": "A",
    }

    request = build_request(cfg)

    assert request.decoys is not None
    assert set(request.decoys) == {"state1"}
    assert request.decoys["state1"].name == "state1_decoy"
    assert request.decoys["state1"].pdb_path == (tmp_path / "state2.pdb")


def test_build_request_resolves_packaged_example_paths(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.targets.state1.pdb_path = "package://benchmarks/pdb/1BDT.pdb"
    cfg.eval.targets.state2.pdb_path = "package://benchmarks/pdb/1QTG.pdb"

    request = build_request(cfg)

    assert request.targets[0].pdb_path == (constants.BENCHMARK_PDB_DIR / "1BDT.pdb").resolve()
    assert request.targets[1].pdb_path == (constants.BENCHMARK_PDB_DIR / "1QTG.pdb").resolve()


def test_build_request_rejects_conflicting_af3_msa_sources(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.af3.use_msa = True
    cfg.eval.af3.unpaired_msa = ">query\nACD\n"
    cfg.eval.af3.unpaired_msa_path = str(tmp_path / "unpaired.a3m")

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_request(cfg)


def test_build_request_rejects_custom_af3_msa_without_toggle(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.af3.unpaired_msa = ">query\nACD\n"

    with pytest.raises(ValueError, match="use_msa=true"):
        build_request(cfg)


def test_build_request_accepts_native_af3_cli_settings(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.af3.executable = "/path/to/python"
    cfg.eval.af3.script = str(tmp_path / "run_alphafold.py")
    cfg.eval.af3.model_dir = str(tmp_path / "models")
    cfg.eval.af3.env_script = str(tmp_path / "af3_env.sh")
    cfg.eval.af3.env = {
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    }

    request = build_request(cfg)

    assert request.af3_executable == "/path/to/python"
    assert request.af3_script == str(tmp_path / "run_alphafold.py")
    assert request.af3_model_dir == str(tmp_path / "models")
    assert request.af3_db_dir is None
    assert request.af3_env_script == str(tmp_path / "af3_env.sh")
    assert request.af3_env == {
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    }


def test_build_request_requires_db_dir_for_native_af3_cli_when_msa_enabled(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.af3.script = str(tmp_path / "run_alphafold.py")
    cfg.eval.af3.model_dir = str(tmp_path / "models")
    cfg.eval.af3.use_msa = True

    with pytest.raises(ValueError, match="db_dir is required"):
        build_request(cfg)


def test_build_request_rejects_partial_native_af3_cli_settings(tmp_path):
    cfg = _build_cfg(tmp_path)
    cfg.eval.af3.script = str(tmp_path / "run_alphafold.py")

    with pytest.raises(ValueError, match="script and eval.af3.model_dir must be set together"):
        build_request(cfg)


def test_resolve_inputs_and_prepare_templates(tmp_path):
    state1_path = tmp_path / "state1.pdb"
    state2_path = tmp_path / "state2.pdb"
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")
    _write_pdb(state1_path, ["ALA", "CYS"])
    _write_pdb(state2_path, ["ALA", "ASP", "CYS"])

    request = EvaluationRequest(
        model_ref=str(model_path),
        resolved_model_path=model_path,
        num_samples=2,
        device="cpu",
        sampling_mode="single",
        refold_mode="single",
        targets=(
            StateInput(name="state1", pdb_path=state1_path, chain_id="A"),
            StateInput(name="state2", pdb_path=state2_path, chain_id="A"),
        ),
        alignment=AlignmentSpec(
            state1_aligned="",
            state2_aligned="",
            query_length=0,
            state1_query_indices=tuple(),
            state2_query_indices=tuple(),
            state1_template_indices=(0, 1),
            state2_template_indices=(0, 2),
            paired_query_indices=tuple(),
            paired_state1_template_indices=tuple(),
            paired_state2_template_indices=tuple(),
        ),
        decoys=None,
        af3_evaluate=False,
        af3_executable=None,
        af3_script=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_env_script=None,
        af3_env={},
        af3_model_seeds=(0, 1, 2),
        output_dir=tmp_path / "out",
    )

    bundle = resolve_inputs(request)
    templates = prepare_templates(bundle, tmp_path / "templates")

    assert bundle.request.alignment.query_length == 2
    assert bundle.targets[0].aligned_sequence == "AC"
    assert bundle.targets[1].aligned_sequence == "AC"
    assert bundle.multi_state_data.cluster_members == ["STATE1_A", "STATE2_A"]
    assert set(templates.keys()) == {("state1", "target"), ("state2", "target")}
    assert templates[("state1", "target")].pdb_path.exists()
    assert templates[("state1", "target")].mmcif_path.exists()
    mmcif_text = templates[("state1", "target")].mmcif_path.read_text(encoding="utf-8")
    assert "_pdbx_audit_revision_history.revision_date" in mmcif_text
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", mmcif_text)


def test_resolve_inputs_supports_partial_decoys(tmp_path):
    state1_path = tmp_path / "state1.pdb"
    state2_path = tmp_path / "state2.pdb"
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")
    _write_pdb(state1_path, ["ALA", "CYS"])
    _write_pdb(state2_path, ["ALA", "ASP", "CYS"])

    request = EvaluationRequest(
        model_ref=str(model_path),
        resolved_model_path=model_path,
        num_samples=2,
        device="cpu",
        sampling_mode="single",
        refold_mode="single",
        targets=(
            StateInput(name="state1", pdb_path=state1_path, chain_id="A"),
            StateInput(name="state2", pdb_path=state2_path, chain_id="A"),
        ),
        alignment=AlignmentSpec(
            state1_aligned="",
            state2_aligned="",
            query_length=0,
            state1_query_indices=tuple(),
            state2_query_indices=tuple(),
            state1_template_indices=(0, 1),
            state2_template_indices=(0, 2),
            paired_query_indices=tuple(),
            paired_state1_template_indices=tuple(),
            paired_state2_template_indices=tuple(),
        ),
        decoys={
            "state1": StateInput(name="state1_decoy", pdb_path=state2_path, chain_id="A"),
        },
        af3_evaluate=False,
        af3_executable=None,
        af3_script=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_env_script=None,
        af3_env={},
        af3_model_seeds=(0, 1, 2),
        output_dir=tmp_path / "out",
    )

    bundle = resolve_inputs(request)

    assert bundle.decoys is not None
    assert set(bundle.decoys) == {"state1"}
    assert bundle.decoys["state1"].aligned_sequence == "AD"


def test_resolve_inputs_builds_multichain_sampling_graph(tmp_path):
    state1_path = tmp_path / "state1_full.pdb"
    state2_path = tmp_path / "state2_full.pdb"
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")

    _write_pdb(state1_path, {"A": ["ALA", "CYS"], "B": ["GLY", "SER"]})
    _write_pdb(state2_path, {"A": ["ALA", "ASP", "CYS"], "B": ["GLY", "SER"]})

    request = EvaluationRequest(
        model_ref=str(model_path),
        resolved_model_path=model_path,
        num_samples=2,
        device="cpu",
        sampling_mode="multi",
        refold_mode="single",
        targets=(
            StateInput(name="state1", pdb_path=state1_path, chain_id="A"),
            StateInput(name="state2", pdb_path=state2_path, chain_id="A"),
        ),
        alignment=AlignmentSpec(
            state1_aligned="",
            state2_aligned="",
            query_length=0,
            state1_query_indices=tuple(),
            state2_query_indices=tuple(),
            state1_template_indices=(0, 1),
            state2_template_indices=(0, 2),
            paired_query_indices=tuple(),
            paired_state1_template_indices=tuple(),
            paired_state2_template_indices=tuple(),
        ),
        decoys=None,
        af3_evaluate=False,
        af3_executable=None,
        af3_script=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_env_script=None,
        af3_env={},
        af3_model_seeds=(0, 1, 2),
        output_dir=tmp_path / "out",
    )

    bundle = resolve_inputs(request)
    state1 = bundle.targets[0]

    assert len(state1.context_chains) == 1
    assert state1.context_chains[0].chain_id == "B"
    assert state1.pyg_data is not None
    assert state1.pyg_data.chain_mapping == {"A": 0, "B": 1}
    assert tuple(state1.pyg_data.chains.tolist()) == (0, 0, 1, 1)
    assert tuple(state1.pyg_data.homo_idx.tolist()) == (0, 0, -1, -1)
    assert tuple(state1.pyg_data.mask_seq.tolist()) == (False, False, True, True)
