import pandas as pd
import pytest

from dynamicmpnn.eval.aggregation import build_per_sequence_dataframe
from dynamicmpnn.eval.inputs import resolve_inputs
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.types import AlignmentSpec, EvaluationRequest, RawMetricRecord, StateInput


def test_decoy_normalization_columns():
    samples = (SampleRecord(sequence_id="sample_000", sequence="AC"),)
    records = (
        RawMetricRecord("sample_000", "state1", "target", "completed", tm_score=0.8, lddt=0.7, rmsd=1.0, mean_plddt=85.0, seed=0),
        RawMetricRecord("sample_000", "state1", "target", "completed", tm_score=0.7, lddt=0.8, rmsd=1.5, mean_plddt=82.0, seed=1),
        RawMetricRecord("sample_000", "state2", "target", "completed", tm_score=0.6, lddt=0.5, rmsd=2.0, mean_plddt=75.0, seed=0),
        RawMetricRecord("sample_000", "state2", "target", "completed", tm_score=0.4, lddt=0.6, rmsd=3.0, mean_plddt=70.0, seed=1),
        RawMetricRecord("sample_000", "state1", "decoy", "completed", tm_score=0.3, lddt=0.2, rmsd=3.0, mean_plddt=50.0, seed=0),
        RawMetricRecord("sample_000", "state1", "decoy", "completed", tm_score=0.4, lddt=0.3, rmsd=2.5, mean_plddt=55.0, seed=1),
        RawMetricRecord("sample_000", "state2", "decoy", "completed", tm_score=0.1, lddt=0.1, rmsd=4.0, mean_plddt=40.0, seed=0),
        RawMetricRecord("sample_000", "state2", "decoy", "completed", tm_score=0.2, lddt=0.2, rmsd=5.0, mean_plddt=45.0, seed=1),
    )

    df = build_per_sequence_dataframe(samples, records, include_decoys=True, af3_enabled=True)
    row = df.iloc[0]

    assert row["af3_status"] == "completed"
    assert row["state1_target_tm_score_mean"] == pytest.approx(0.75)
    assert row["state1_target_tm_score_best"] == pytest.approx(0.8)
    assert row["state1_tm_score_decoy_normalized_mean"] == pytest.approx(0.75 / 0.35)
    assert row["state2_tm_score_decoy_normalized_best"] == pytest.approx(0.6 / 0.2)
    assert row["state1_rmsd_decoy_normalized_mean"] == pytest.approx(1.25 / 2.75)
    assert row["state2_rmsd_decoy_normalized_best"] == pytest.approx(2.0 / 4.0)
    assert row["pair_tm_score_decoy_normalized_mean"] == pytest.approx(((0.75 / 0.35) + (0.5 / 0.15)) / 2.0)
    assert row["pair_rmsd_decoy_normalized_best"] == pytest.approx(((1.0 / 2.5) + (2.0 / 4.0)) / 2.0)


def test_skipped_af3_still_emits_structural_columns():
    samples = (SampleRecord(sequence_id="sample_000", sequence="AC"),)
    df = build_per_sequence_dataframe(samples, (), include_decoys=False, af3_enabled=False)
    row = df.iloc[0]

    assert row["af3_status"] == "skipped"
    assert row["state1_target_af3_status"] == "skipped"
    assert "state1_target_tm_score_mean" in df.columns
    assert "pair_tm_score_mean" in df.columns


def test_partial_decoys_only_contribute_present_state_ratios():
    samples = (SampleRecord(sequence_id="sample_000", sequence="AC"),)
    records = (
        RawMetricRecord("sample_000", "state1", "target", "completed", tm_score=0.8, rmsd=1.0, seed=0),
        RawMetricRecord("sample_000", "state2", "target", "completed", tm_score=0.6, rmsd=2.0, seed=0),
        RawMetricRecord("sample_000", "state1", "decoy", "completed", tm_score=0.3, rmsd=3.5, seed=0),
    )

    df = build_per_sequence_dataframe(samples, records, include_decoys=True, af3_enabled=True)
    row = df.iloc[0]

    assert row["state1_tm_score_decoy_normalized_mean"] == pytest.approx(0.8 / 0.3)
    assert pd.isna(row["state2_decoy_af3_status"])
    assert row["pair_tm_score_decoy_normalized_mean"] == pytest.approx(0.8 / 0.3)
    assert row["pair_rmsd_decoy_normalized_mean"] == pytest.approx(1.0 / 3.5)


def test_sequence_recovery_adds_one_column_per_target_sequence():
    samples = (SampleRecord(sequence_id="sample_000", sequence="AC"),)

    df = build_per_sequence_dataframe(
        samples,
        (),
        include_decoys=False,
        af3_enabled=False,
        sequence_recovery_target_sequences=("AC", "AA"),
    )
    row = df.iloc[0]

    assert row["sequence_recovery_target_000"] == 1.0
    assert row["sequence_recovery_target_001"] == 0.5


def test_sequence_recovery_ignores_gap_positions_when_all_non_gap_positions_match():
    samples = (SampleRecord(sequence_id="sample_000", sequence="ABCD"),)

    df = build_per_sequence_dataframe(
        samples,
        (),
        include_decoys=False,
        af3_enabled=False,
        sequence_recovery_target_sequences=("A-C-",),
    )

    assert df.iloc[0]["sequence_recovery_target_000"] == 1.0


def test_sequence_recovery_excludes_gap_positions_from_denominator():
    samples = (SampleRecord(sequence_id="sample_000", sequence="ABXD"),)

    df = build_per_sequence_dataframe(
        samples,
        (),
        include_decoys=False,
        af3_enabled=False,
        sequence_recovery_target_sequences=("ABCD", "A-C-"),
    )
    row = df.iloc[0]

    assert row["sequence_recovery_target_000"] == 0.75
    assert row["sequence_recovery_target_001"] == 0.5


def test_sequence_recovery_returns_missing_for_all_gap_target():
    samples = (SampleRecord(sequence_id="sample_000", sequence="ABCD"),)

    df = build_per_sequence_dataframe(
        samples,
        (),
        include_decoys=False,
        af3_enabled=False,
        sequence_recovery_target_sequences=("----",),
    )

    assert pd.isna(df.iloc[0]["sequence_recovery_target_000"])


def test_resolve_inputs_rejects_mismatched_sequence_recovery_target_length(tmp_path):
    state1_path = tmp_path / "state1.pdb"
    state2_path = tmp_path / "state2.pdb"
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("dummy", encoding="utf-8")
    state1_path.write_text(
        "".join(
            [
                "ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00 20.00           N\n",
                "ATOM      2  CA  ALA A   1       2.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      3  C   ALA A   1       3.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      4  N   CYS A   2       4.000   0.000   0.000  1.00 20.00           N\n",
                "ATOM      5  CA  CYS A   2       5.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      6  C   CYS A   2       6.000   0.000   0.000  1.00 20.00           C\n",
                "TER\nEND\n",
            ]
        ),
        encoding="utf-8",
    )
    state2_path.write_text(
        "".join(
            [
                "ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00 20.00           N\n",
                "ATOM      2  CA  ALA A   1       2.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      3  C   ALA A   1       3.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      4  N   CYS A   2       4.000   0.000   0.000  1.00 20.00           N\n",
                "ATOM      5  CA  CYS A   2       5.000   0.000   0.000  1.00 20.00           C\n",
                "ATOM      6  C   CYS A   2       6.000   0.000   0.000  1.00 20.00           C\n",
                "TER\nEND\n",
            ]
        ),
        encoding="utf-8",
    )

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
            state2_template_indices=(0, 1),
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
        sequence_recovery_target_sequences=("A",),
    )

    with pytest.raises(ValueError, match="sampled-sequence length"):
        resolve_inputs(request)
