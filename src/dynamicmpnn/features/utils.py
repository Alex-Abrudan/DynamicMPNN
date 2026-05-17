from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from torch import Tensor
from torch_geometric.data import Batch, Data

from dynamicmpnn.types import OrientationTensor, SEQ_MASK


@jaxtyped(typechecker=typechecker)
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safely normalize a Tensor."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


@jaxtyped(typechecker=typechecker)
def compute_edge_distance(pos: CoordTensor, edge_index: EdgeTensor) -> torch.Tensor:
    """Compute euclidean distance between nodes connected by edges."""
    return torch.pairwise_distance(pos[edge_index[0, :]], pos[edge_index[1, :]])


@jaxtyped(typechecker=typechecker)
def pos_emb(edge_index: EdgeTensor, num_pos_emb: int = 16):
    """Positional embedding for edges based on sequence distance."""
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_pos_emb)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


@typechecker
def rbf_expansion(
    h: torch.Tensor,
    value_min: float = 0.0,
    value_max: float = 30.0,
    num_rbf: int = 32,
) -> torch.Tensor:
    """Expand distances using radial basis functions."""
    rbf_centers = torch.linspace(value_min, value_max, num_rbf, device=h.device)
    std = (rbf_centers[1] - rbf_centers[0]).item()
    h_expanded = h.unsqueeze(-1)
    rbf_values = torch.exp(-((h_expanded - rbf_centers) / std) ** 2)
    return rbf_values


@jaxtyped(typechecker=typechecker)
def orientations(coords: torch.Tensor, ca_idx: int = 1) -> OrientationTensor:
    """Calculate orientation vectors for a protein."""
    if coords.ndim == 3:
        coords = coords[:, ca_idx, :]

    forward = coords[1:] - coords[:-1]
    backward = coords[:-1] - coords[1:]

    forward = F.normalize(forward, dim=-1)
    backward = F.normalize(backward, dim=-1)

    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])

    return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)


@typechecker
def amino_acid_one_hot(x: Union[Batch, Data], num_classes: int = 23) -> torch.Tensor:
    """Returns one-hot encoding of amino acid sequence."""
    return F.one_hot(x.residue_type, num_classes=num_classes).float()


@typechecker
def get_masked_sequence(seq: Union[Tensor, Data], mask: torch.Tensor, fill_value=SEQ_MASK) -> torch.Tensor:
    """Return sequence with masked positions filled."""
    return torch.where(mask == 1, seq, fill_value)
