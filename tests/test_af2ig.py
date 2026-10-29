"""The AF2 initial-guess arm, which mirrors the AF3 template-based one.

What these tests pin down is the mirroring itself: the same (state, kind)
keys, the same query/template index mapping, and the same record type, so
the decoy-normalised columns are the AF3 arithmetic rather than a second
implementation of it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from torch_geometric.data import Data

from dynamicmpnn.eval.af2ig import build_af2ig_jobs, write_af2ig_manifest
from dynamicmpnn.eval.af2ig_runner import seed_from_guess
from dynamicmpnn.eval.aggregation import build_per_sequence_dataframe
from dynamicmpnn.eval.guesses import GUESS_ATOMS, build_backbone, prepare_guesses
from dynamicmpnn.eval.inputs import PreparedInputBundle, PreparedState
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.scoring import score_af2ig_jobs
from dynamicmpnn.eval.structure_utils import extract_chain_record, get_single_model
from dynamicmpnn.eval.types import AlignmentSpec, DecoyInput, EvaluationRequest, RawMetricRecord, StateInput


# One extended residue repeated along x, so every backbone frame is complete
# and a virtual CB is always well defined.
_RESIDUE_TEMPLATE = (
    ("N", (0.000, 0.000, 0.000)),
    ("CA", (1.458, 0.000, 0.000)),
    ("C", (2.009, 1.420, 0.000)),
    ("O", (1.251, 2.390, 0.000)),
)


def _write_pdb(path: Path, length: int, *, drop_oxygen_at: int | None = None) -> Path:
    lines = []
    serial = 1
    for residue_index in range(length):
        offset = residue_index * 4.0
        for atom_name, (x, y, z) in _RESIDUE_TEMPLATE:
            if atom_name == "O" and residue_index == drop_oxygen_at:
                continue
            lines.append(
                f"ATOM  {serial:>5} {atom_name:<4} ALA A{residue_index + 1:>4}    "
                f"{x + offset:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atom_name[0]}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _state_from_pdb(path: Path, name: str, *, query_indices, template_indices) -> PreparedState:
    record = extract_chain_record(get_single_model(path), "A")
    return PreparedState(
        name=name,
        pdb_path=path,
        chain_id="A",
        sequence=record.sequence,
        aligned_sequence="".join(record.sequence[index] for index in template_indices),
        residues_3=record.residues_3,
        residue_numbers=record.residue_numbers,
        residue_insertions=record.residue_insertions,
        coords=record.coords,
        query_indices=tuple(query_indices),
        template_indices=tuple(template_indices),
        context_chains=(),
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
            state1_aligned="",
            state2_aligned="",
            query_length=0,
            state1_query_indices=(),
            state2_query_indices=(),
            state1_template_indices=(),
            state2_template_indices=(),
            paired_query_indices=(),
            paired_state1_template_indices=(),
            paired_state2_template_indices=(),
        ),
        decoys={
            "state1": DecoyInput(name="state1_decoy", pdb_path=tmp_path / "state1_decoy.pdb", chain_id="A"),
            "state2": DecoyInput(name="state2_decoy", pdb_path=tmp_path / "state2_decoy.pdb", chain_id="A"),
        },
        af3_evaluate=False,
        af3_executable=None,
        af3_script=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_env_script=None,
        af3_env={},
        af3_model_seeds=(0,),
        output_dir=tmp_path / "out",
        af2ig_evaluate=True,
    )


def _bundle(tmp_path: Path, length: int = 4) -> PreparedInputBundle:
    indices = tuple(range(length))
    states = {}
    for name in ("state1", "state2", "state1_decoy", "state2_decoy"):
        path = _write_pdb(tmp_path / f"{name}.pdb", length)
        states[name] = _state_from_pdb(path, name, query_indices=indices, template_indices=indices)
    return PreparedInputBundle(
        request=_request(tmp_path),
        targets=(states["state1"], states["state2"]),
        decoys={"state1": states["state1_decoy"], "state2": states["state2_decoy"]},
        multi_state_data=Data(cluster_members=["STATE1_A", "STATE2_A"]),
    )


class _StubResidueConstants:
    """Only what seed_from_guess touches, in AlphaFold's own slot order."""

    atom_type_num = 37
    atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4}


def test_build_backbone_fills_five_atoms_and_builds_cb(tmp_path):
    state = _state_from_pdb(
        _write_pdb(tmp_path / "s.pdb", 3), "state1", query_indices=(0, 1, 2), template_indices=(0, 1, 2)
    )
    xyz, mask = build_backbone(state)

    assert xyz.shape == (3, len(GUESS_ATOMS), 3)
    assert mask.shape == (3, len(GUESS_ATOMS))
    assert mask.all(), "every atom including the virtual CB should be present"
    # CB is built, not read: it must sit off the N/CA/C plane that defines it.
    n, ca, c, _o, cb = xyz[0]
    assert np.linalg.norm(cb - ca) == pytest.approx(1.53, abs=0.1)
    assert not np.allclose(cb, n) and not np.allclose(cb, c)


def test_build_backbone_masks_a_missing_oxygen(tmp_path):
    path = _write_pdb(tmp_path / "s.pdb", 3, drop_oxygen_at=1)
    state = _state_from_pdb(path, "state1", query_indices=(0, 1, 2), template_indices=(0, 1, 2))
    _xyz, mask = build_backbone(state)

    oxygen_slot = GUESS_ATOMS.index("O")
    assert mask[1, oxygen_slot] == 0.0
    assert mask[0, oxygen_slot] == 1.0
    # A missing O must not cost the residue its CB, which needs N/CA/C only.
    assert mask[1, GUESS_ATOMS.index("CB")] == 1.0


def test_prepare_guesses_covers_targets_and_decoys(tmp_path):
    guesses = prepare_guesses(_bundle(tmp_path), tmp_path / "guesses")

    assert set(guesses) == {
        ("state1", "target"),
        ("state2", "target"),
        ("state1", "decoy"),
        ("state2", "decoy"),
    }
    for key, artifact in guesses.items():
        assert artifact.npz_path.exists(), key
        stored = np.load(artifact.npz_path)
        assert [name for name in stored["atoms"]] == list(GUESS_ATOMS)
        assert len(artifact.ca_coords) == stored["xyz"].shape[0]


def test_build_af2ig_jobs_spans_every_sequence_state_and_kind(tmp_path):
    guesses = prepare_guesses(_bundle(tmp_path), tmp_path / "guesses")
    samples = (SampleRecord(sequence_id="sample_000", sequence="AAAA"),
               SampleRecord(sequence_id="sample_001", sequence="AAAA"))

    jobs = build_af2ig_jobs(samples, guesses, tmp_path / "raw", num_recycles=1)

    assert len(jobs) == len(samples) * len(guesses)
    assert {(job.state_name, job.structure_kind) for job in jobs} == set(guesses)
    # The decoy job seeds on the decoy shape, which is what makes the
    # decoy-normalised column mean the same thing it means for AF3.
    decoy_job = next(job for job in jobs if job.structure_kind == "decoy")
    assert decoy_job.guess_npz == guesses[(decoy_job.state_name, "decoy")].npz_path

    manifest_path = write_af2ig_manifest(jobs, tmp_path / "manifest.json")
    assert manifest_path.exists()


def test_seed_from_guess_scatters_through_the_index_mapping(tmp_path):
    xyz = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
    mask = np.ones((3, 5), dtype=np.float32)
    npz_path = tmp_path / "guess.npz"
    np.savez_compressed(npz_path, xyz=xyz, mask=mask, atoms=np.array(GUESS_ATOMS))

    # Residue 2 of the structure is residue 0 of the design, and residue 0 of
    # the structure is residue 3 - the same remapping AF3 does with
    # queryIndices/templateIndices.
    job = {
        "job_name": "j",
        "sequence": "AAAA",
        "guess_npz": str(npz_path),
        "query_indices": [0, 3],
        "template_indices": [2, 0],
    }
    positions, masks = seed_from_guess(job, _StubResidueConstants)

    assert positions.shape == (4, 37, 3)
    ca_slot = _StubResidueConstants.atom_order["CA"]
    assert np.allclose(positions[0, ca_slot], xyz[2, GUESS_ATOMS.index("CA")])
    assert np.allclose(positions[3, ca_slot], xyz[0, GUESS_ATOMS.index("CA")])
    assert masks[1].sum() == 0.0, "unmapped residues stay masked out"
    assert masks[0, ca_slot] == 1.0


def test_seed_from_guess_rejects_a_mapping_past_the_sequence(tmp_path):
    npz_path = tmp_path / "guess.npz"
    np.savez_compressed(
        npz_path,
        xyz=np.zeros((2, 5, 3), dtype=np.float32),
        mask=np.ones((2, 5), dtype=np.float32),
        atoms=np.array(GUESS_ATOMS),
    )
    job = {
        "job_name": "j",
        "sequence": "AA",
        "guess_npz": str(npz_path),
        "query_indices": [0, 9],
        "template_indices": [0, 1],
    }
    with pytest.raises(ValueError, match="query mapping needs index 9"):
        seed_from_guess(job, _StubResidueConstants)


def test_score_af2ig_jobs_scores_a_perfect_prediction(tmp_path):
    bundle = _bundle(tmp_path)
    guesses = prepare_guesses(bundle, tmp_path / "guesses")
    samples = (SampleRecord(sequence_id="sample_000", sequence="AAAA"),)
    jobs = build_af2ig_jobs(samples, guesses, tmp_path / "raw")

    class _Run:
        def __init__(self, output_dir):
            self.status, self.error, self.output_dir = "completed", None, output_dir

    run_results = {}
    for job in jobs:
        # Hand back the very structure the job was seeded on.
        job.output_dir.mkdir(parents=True, exist_ok=True)
        _write_pdb(job.output_dir / "prediction.pdb", 4)
        run_results[job.job_name] = _Run(job.output_dir)

    records = score_af2ig_jobs(jobs, guesses, run_results)

    assert len(records) == len(jobs)
    assert {record.af3_status for record in records} == {"completed"}
    for record in records:
        assert record.rmsd == pytest.approx(0.0, abs=1e-6)
        assert record.tm_score == pytest.approx(1.0, abs=1e-6)
        assert record.mean_plddt is None, "no confidences file was written"


def test_score_af2ig_jobs_records_a_failed_run(tmp_path):
    bundle = _bundle(tmp_path)
    guesses = prepare_guesses(bundle, tmp_path / "guesses")
    jobs = build_af2ig_jobs((SampleRecord(sequence_id="sample_000", sequence="AAAA"),), guesses, tmp_path / "raw")

    class _Run:
        status, error, output_dir = "failed", "boom", Path("/nonexistent")

    records = score_af2ig_jobs(jobs, guesses, {job.job_name: _Run() for job in jobs})

    assert {record.af3_status for record in records} == {"failed"}
    assert {record.error for record in records} == {"boom"}


def test_af2ig_backend_names_its_own_columns_and_normalizes_by_decoy():
    samples = (SampleRecord(sequence_id="sample_000", sequence="AC"),)
    records = (
        RawMetricRecord("sample_000", "state1", "target", "completed", tm_score=0.8, lddt=0.7, rmsd=1.0, mean_plddt=85.0),
        RawMetricRecord("sample_000", "state2", "target", "completed", tm_score=0.6, lddt=0.5, rmsd=2.0, mean_plddt=75.0),
        RawMetricRecord("sample_000", "state1", "decoy", "completed", tm_score=0.4, lddt=0.2, rmsd=4.0, mean_plddt=50.0),
        RawMetricRecord("sample_000", "state2", "decoy", "completed", tm_score=0.2, lddt=0.1, rmsd=5.0, mean_plddt=40.0),
    )

    df = build_per_sequence_dataframe(
        samples, records, include_decoys=True, af3_enabled=True, backend="af2ig"
    )
    row = df.iloc[0]

    assert row["af2ig_status"] == "completed"
    assert row["state1_target_af2ig_status"] == "completed"
    assert "af3_status" not in df.columns, "the AF3 arm keeps its own column namespace"
    # Same target/decoy arithmetic the AF3 arm uses.
    assert row["state1_tm_score_decoy_normalized_mean"] == pytest.approx(0.8 / 0.4)
    assert row["state2_rmsd_decoy_normalized_mean"] == pytest.approx(2.0 / 5.0)


_STUB_PREDICTOR = '''
import json, sys
from pathlib import Path

args = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
jobs = json.loads(Path(args["--manifest"]).read_text())["jobs"]
assert args["--params_dir"] and args["--output_dir"], "runner needs both paths"
for job in jobs:
    # Everything the real runner reads must be present in the manifest.
    for key in ("sequence", "guess_npz", "query_indices", "template_indices",
                "num_recycles", "template_backbone", "output_dir", "chain_id"):
        assert key in job, key
    out = Path(job["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lines, serial = [], 1
    for index in range(len(job["sequence"])):
        offset = index * 4.0
        for name, x, y, z in (("N", 0.0, 0.0, 0.0), ("CA", 1.458, 0.0, 0.0), ("C", 2.009, 1.42, 0.0)):
            lines.append(
                f"ATOM  {serial:>5} {name:<4} ALA {job['chain_id']}{index + 1:>4}    "
                f"{x + offset:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {name[0]}")
            serial += 1
    (out / "prediction.pdb").write_text("\\n".join(lines) + "\\nEND\\n")
    (out / "confidences.json").write_text(json.dumps({"plddt": [90.0] * len(job["sequence"])}))
'''


def test_af2ig_round_trip_through_a_stub_predictor(tmp_path):
    """Pins the manifest contract between af2ig.py and the runner.

    The stub stands in for AlphaFold2, which cannot be imported here: it
    asserts on every manifest key the real runner reads and writes back the
    two artifacts the real one writes, so a change to either side that breaks
    the other fails here rather than after a GPU batch.
    """
    from dynamicmpnn.eval.af2ig import run_af2ig_jobs
    from dynamicmpnn.eval.aggregation import build_all_af3_results_dataframe, write_reports

    predictor = tmp_path / "stub_predictor.py"
    predictor.write_text(_STUB_PREDICTOR, encoding="utf-8")

    bundle = _bundle(tmp_path)
    guesses = prepare_guesses(bundle, tmp_path / "af2ig" / "guesses")
    samples = (SampleRecord(sequence_id="sample_000", sequence="AAAA"),)
    jobs = build_af2ig_jobs(samples, guesses, tmp_path / "af2ig" / "raw")

    run_results = run_af2ig_jobs(
        sys.executable, jobs, script=str(predictor), params_dir=str(tmp_path), initial_guess_dir=None
    )
    assert {result.status for result in run_results.values()} == {"completed"}

    records = score_af2ig_jobs(jobs, guesses, run_results)
    assert {record.af3_status for record in records} == {"completed"}
    assert all(record.mean_plddt == pytest.approx(90.0) for record in records)

    summary = build_per_sequence_dataframe(
        samples, records, include_decoys=True, af3_enabled=True, backend="af2ig"
    )
    paths = write_reports(
        build_all_af3_results_dataframe(samples, records, backend="af2ig"),
        summary,
        tmp_path,
        backend="af2ig",
    )
    assert Path(paths["summary_af2ig_results"]).exists()
    assert summary.iloc[0]["af2ig_status"] == "completed"
    assert summary.iloc[0]["state1_tm_score_decoy_normalized_mean"] is not None


def test_af2ig_marks_a_job_whose_prediction_never_appeared(tmp_path):
    """A batch can exit 0 and still skip a job; the artifact is the verdict."""
    from dynamicmpnn.eval.af2ig import run_af2ig_jobs

    bundle = _bundle(tmp_path)
    guesses = prepare_guesses(bundle, tmp_path / "af2ig" / "guesses")
    jobs = build_af2ig_jobs(
        (SampleRecord(sequence_id="sample_000", sequence="AAAA"),), guesses, tmp_path / "af2ig" / "raw"
    )

    run_results = run_af2ig_jobs(
        "/bin/true", jobs, script="ignored", params_dir=str(tmp_path), initial_guess_dir=None
    )

    assert {result.status for result in run_results.values()} == {"failed"}
    assert all("no prediction" in result.error for result in run_results.values())
