from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from dynamicmpnn.eval.inputs import PreparedInputBundle
from dynamicmpnn.features.featurizer import ProteinGraphFeaturiser
from dynamicmpnn.types import BASE_AMINO_ACIDS, DISTANCE_EPS


@dataclass(frozen=True)
class SampleRecord:
    sequence_id: str
    sequence: str


@dataclass(frozen=True)
class SampleResult:
    samples: tuple[SampleRecord, ...]
    featurized_graph: Data


def _checkpoint_model_config(checkpoint: dict[object, object]):
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict) or "cfg" not in hyper_parameters:
        raise KeyError("Checkpoint is missing hyper_parameters.cfg")

    cfg = hyper_parameters["cfg"]
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise KeyError("Checkpoint cfg is missing model")

    return OmegaConf.create(cfg["model"])


def _load_model_for_sampling(checkpoint_path: str | Path, cfg, num_samples: int):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_cfg = _checkpoint_model_config(checkpoint)
    OmegaConf.set_struct(model_cfg, False)

    # Remap old module paths to new ones
    if "_target_" in model_cfg:
        old_target = model_cfg["_target_"]
        if old_target.startswith("dynamicprot.src."):
            new_target = old_target.replace("dynamicprot.src.", "dynamicmpnn.")
            new_target = new_target.replace("AutoregressiveMultiGNNv1", "DynamicMPNN")
            model_cfg["_target_"] = new_target

    model_cfg.n_samples = num_samples
    if cfg.model.get("refresh_interval") is not None:
        model_cfg.refresh_interval = cfg.model.refresh_interval
    if cfg.model.get("temperature") is not None:
        model_cfg.temperature = cfg.model.temperature

    model = hydra.utils.instantiate(model_cfg)
    state_dict = {
        key.removeprefix("GNN_model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("GNN_model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _create_featuriser(cfg) -> ProteinGraphFeaturiser:
    return ProteinGraphFeaturiser(
        representation=cfg.features.representation,
        scalar_node_features=cfg.features.scalar_node_features,
        vector_node_features=cfg.features.vector_node_features,
        edge_types=cfg.features.edge_types,
        scalar_edge_features=cfg.features.scalar_edge_features,
        vector_edge_features=cfg.features.vector_edge_features,
        split="test",
        noise_scale=cfg.features.noise_scale,
        distance_eps=DISTANCE_EPS,
        device="cpu",
        pair_tm_tau=getattr(cfg.features, "pair_tm_tau", 0.0),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def decode_samples(sample_tensor: torch.Tensor) -> tuple[str, ...]:
    sequences = []
    for sample in sample_tensor:
        sequence = "".join(BASE_AMINO_ACIDS[int(index)] for index in sample.cpu().tolist())
        sequences.append(sequence)
    return tuple(sequences)


def load_model_and_sample_batch(
    checkpoint_path: str | Path,
    cfg,
    batch: Data,
    num_samples: int,
    device: str = "auto",
) -> tuple[str, ...]:
    torch_device = resolve_device(device)
    model = _load_model_for_sampling(checkpoint_path, cfg, num_samples)
    model = model.to(torch_device)

    batch = batch.to(torch_device)
    with torch.inference_mode():
        sample_tensor, _, _, _ = model.sample(batch, return_logits=True)
    return decode_samples(sample_tensor)


def sample_sequences(cfg, bundle: PreparedInputBundle) -> SampleResult:
    featuriser = _create_featuriser(cfg)
    featurized_graph = featuriser(bundle.multi_state_data, pdb_code="evaluation_pair")
    if featurized_graph is None:
        raise ValueError("Featuriser returned None for the resolved evaluation inputs")

    batch = Batch.from_data_list([featurized_graph])
    sequences = load_model_and_sample_batch(
        checkpoint_path=bundle.request.resolved_model_path,
        cfg=cfg,
        batch=batch,
        num_samples=bundle.request.num_samples,
        device=bundle.request.device,
    )
    samples = tuple(
        SampleRecord(sequence_id=f"sample_{sample_idx:03d}", sequence=sequence)
        for sample_idx, sequence in enumerate(sequences)
    )
    return SampleResult(samples=samples, featurized_graph=featurized_graph)


def write_sample_artifacts(samples: Sequence[SampleRecord], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"
    fasta_path = output_dir / "samples.fasta"

    dataframe = pd.DataFrame(
        {
            "sequence_id": [sample.sequence_id for sample in samples],
            "sequence": [sample.sequence for sample in samples],
            "length": [len(sample.sequence) for sample in samples],
        }
    )
    dataframe.to_csv(csv_path, index=False)

    with fasta_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f">{sample.sequence_id}\n{sample.sequence}\n")

    return {
        "samples_csv": str(csv_path),
        "samples_fasta": str(fasta_path),
    }
