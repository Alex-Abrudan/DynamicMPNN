from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.types import RawMetricRecord


RAW_METRICS = ("tm_score", "lddt", "rmsd", "mean_plddt")
_MAX_METRICS = {"tm_score", "lddt", "mean_plddt"}


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None and pd.notna(value)]
    if not present:
        return None
    return float(sum(present) / len(present))


def _safe_ratio(numerator: object, denominator: object) -> Optional[float]:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(float(numerator) / denominator_value)


def _sequence_recovery(sequence: str, target_sequence: str) -> object:
    non_gap_positions = [index for index, residue in enumerate(target_sequence) if residue != "-"]
    if not non_gap_positions:
        return pd.NA
    # Gap residues in the target are excluded from the denominator.
    matches = sum(sequence[index] == target_sequence[index] for index in non_gap_positions)
    return float(matches / len(non_gap_positions))


def _build_record_groups(
    records: Sequence[RawMetricRecord],
) -> dict[tuple[str, str, str], list[RawMetricRecord]]:
    grouped: dict[tuple[str, str, str], list[RawMetricRecord]] = {}
    for record in records:
        grouped.setdefault((record.sequence_id, record.state_name, record.structure_kind), []).append(record)
    return grouped


def _aggregate_metric(records: Sequence[RawMetricRecord], metric_name: str) -> tuple[Optional[float], Optional[float]]:
    values = [
        getattr(record, metric_name)
        for record in records
        if getattr(record, metric_name) is not None and pd.notna(getattr(record, metric_name))
    ]
    if not values:
        return None, None
    mean_value = float(sum(values) / len(values))
    best_value = max(values) if metric_name in _MAX_METRICS else min(values)
    return mean_value, float(best_value)


def _aggregate_status(records: Sequence[RawMetricRecord], af3_enabled: bool) -> object:
    if not af3_enabled:
        return "skipped"
    if not records:
        return pd.NA
    statuses = {record.af3_status for record in records}
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"failed"}:
        return "failed"
    return "partial"


def build_all_af3_results_dataframe(
    samples: Sequence[SampleRecord],
    records: Sequence[RawMetricRecord],
) -> pd.DataFrame:
    sequence_map = {sample.sequence_id: sample.sequence for sample in samples}
    rows = []
    for record in records:
        rows.append(
            {
                "sequence_id": record.sequence_id,
                "sequence": sequence_map.get(record.sequence_id),
                "state_name": record.state_name,
                "structure_kind": record.structure_kind,
                "seed": record.seed,
                "af3_sample": record.af3_sample,
                "af3_status": record.af3_status,
                "ranking_score": record.ranking_score,
                "tm_score": record.tm_score,
                "lddt": record.lddt,
                "rmsd": record.rmsd,
                "mean_plddt": record.mean_plddt,
                "error": record.error,
                "predicted_structure_path": None
                if record.predicted_structure_path is None
                else str(record.predicted_structure_path),
                "confidences_path": None if record.confidences_path is None else str(record.confidences_path),
            }
        )
    return pd.DataFrame(rows)


def build_per_sequence_dataframe(
    samples: Sequence[SampleRecord],
    records: Sequence[RawMetricRecord],
    include_decoys: bool,
    af3_enabled: bool,
    sequence_recovery_target_sequences: Sequence[str] = (),
) -> pd.DataFrame:
    record_groups = _build_record_groups(records)
    rows = []
    states = ("state1", "state2")
    structure_kinds = ("target", "decoy") if include_decoys else ("target",)

    for sample in samples:
        row: dict[str, object] = {
            "sequence_id": sample.sequence_id,
            "sequence": sample.sequence,
        }
        for target_index, target_sequence in enumerate(sequence_recovery_target_sequences):
            row[f"sequence_recovery_target_{target_index:03d}"] = _sequence_recovery(sample.sequence, target_sequence)
        target_statuses = []

        for state_name in states:
            for structure_kind in structure_kinds:
                prefix = f"{state_name}_{structure_kind}"
                group = record_groups.get((sample.sequence_id, state_name, structure_kind), [])
                status = _aggregate_status(group, af3_enabled)
                row[f"{prefix}_af3_status"] = status
                row[f"{prefix}_error"] = None if not group else "; ".join(
                    sorted({record.error for record in group if record.error})
                ) or None
                for metric_name in RAW_METRICS:
                    mean_value, best_value = _aggregate_metric(group, metric_name)
                    row[f"{prefix}_{metric_name}_mean"] = mean_value
                    row[f"{prefix}_{metric_name}_best"] = best_value
                if structure_kind == "target":
                    target_statuses.append(status)

        if not af3_enabled:
            row["af3_status"] = "skipped"
        elif any(status == "failed" for status in target_statuses):
            row["af3_status"] = "failed"
        elif target_statuses and all(status == "completed" for status in target_statuses):
            row["af3_status"] = "completed"
        else:
            row["af3_status"] = "partial"

        for metric_name in RAW_METRICS:
            row[f"pair_{metric_name}_mean"] = _mean(
                row.get(f"{state_name}_target_{metric_name}_mean") for state_name in states
            )
            row[f"pair_{metric_name}_best"] = _mean(
                row.get(f"{state_name}_target_{metric_name}_best") for state_name in states
            )

        if include_decoys:
            # Paper protocol: decoy-normalized metrics are always target / decoy.
            for state_name in states:
                for aggregate_name in ("mean", "best"):
                    tm_target = row.get(f"{state_name}_target_tm_score_{aggregate_name}")
                    tm_decoy = row.get(f"{state_name}_decoy_tm_score_{aggregate_name}")
                    lddt_target = row.get(f"{state_name}_target_lddt_{aggregate_name}")
                    lddt_decoy = row.get(f"{state_name}_decoy_lddt_{aggregate_name}")
                    rmsd_target = row.get(f"{state_name}_target_rmsd_{aggregate_name}")
                    rmsd_decoy = row.get(f"{state_name}_decoy_rmsd_{aggregate_name}")
                    plddt_target = row.get(f"{state_name}_target_mean_plddt_{aggregate_name}")
                    plddt_decoy = row.get(f"{state_name}_decoy_mean_plddt_{aggregate_name}")

                    row[f"{state_name}_tm_score_decoy_normalized_{aggregate_name}"] = _safe_ratio(
                        tm_target, tm_decoy
                    )
                    row[f"{state_name}_lddt_decoy_normalized_{aggregate_name}"] = _safe_ratio(
                        lddt_target, lddt_decoy
                    )
                    row[f"{state_name}_rmsd_decoy_normalized_{aggregate_name}"] = _safe_ratio(
                        rmsd_target, rmsd_decoy
                    )
                    row[f"{state_name}_mean_plddt_decoy_normalized_{aggregate_name}"] = _safe_ratio(
                        plddt_target, plddt_decoy
                    )

            for metric_name in RAW_METRICS:
                row[f"pair_{metric_name}_decoy_normalized_mean"] = _mean(
                    row.get(f"{state_name}_{metric_name}_decoy_normalized_mean") for state_name in states
                )
                row[f"pair_{metric_name}_decoy_normalized_best"] = _mean(
                    row.get(f"{state_name}_{metric_name}_decoy_normalized_best") for state_name in states
                )

        rows.append(row)

    return pd.DataFrame(rows)


def write_reports(all_af3_results_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    all_results_path = output_dir / "all_af3_results.csv"
    summary_path = output_dir / "summary_af3_results.csv"
    all_af3_results_df.to_csv(all_results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return {
        "all_af3_results": str(all_results_path),
        "summary_af3_results": str(summary_path),
    }
