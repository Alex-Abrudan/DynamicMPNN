from typing import List, Union

import torch
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

from dynamicmpnn.features.utils import (
    _normalize,
    compute_edge_distance,
    pos_emb,
    rbf_expansion,
)
from dynamicmpnn.types import SEQ_MASK

EDGE_FEATURES: List[str] = [
    "edge_distance",
    "node_features",
    "edge_type",
    "sequence_distance",
]
"""List of edge features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_edge_features(x: Union[Data, Batch], features: Union[List[str], ListConfig], idx: int=0) -> torch.Tensor:
    """
    Computes scalar edge features from a :class:`~torch_geometric.data.Data` or 
    :class:`~torch_geometric.data.Batch` object.
    """
    feats = []
    pos = x.coords[:, idx][:, 1]
    
    # Get edge_index for this conformation
    edge_index = x.edge_index[idx]  # Shape: [2, max_edges] or List of 2
    
    # Filter out padding (assuming SEQ_MASK is used for padding)
    valid_mask = (edge_index[0] != SEQ_MASK) & (edge_index[1] != SEQ_MASK)
    edge_index = edge_index[:, valid_mask]  # Only keep valid edges
    
    distances = compute_edge_distance(pos, edge_index)
    
    for feature in features:
        if feature == "edge_distance":
            feats.append(distances)
        elif feature == "rbf_16":
            rbf_features = rbf_expansion(distances, num_rbf=16)
            feats.append(rbf_features)
        elif feature == "node_features":
            n1, n2 = x.x[edge_index[0]], x.x[edge_index[1]]
            feats.append(torch.cat([n1, n2], dim=1))
        elif feature == "edge_type":
            # Also need to filter edge_type
            edge_type = x.edge_type[idx][valid_mask] if valid_mask.any() else x.edge_type[idx]
            feats.append(edge_type.unsqueeze(-1))
        elif feature == "sequence_distance":
            seq_dist = edge_index[1] - edge_index[0]
            feats.append(seq_dist.unsqueeze(-1))
        elif feature == "pos_emb":
            feats.append(pos_emb(edge_index))
        else:
            raise ValueError(f"Unknown edge feature {feature}")
            
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1) if feats else torch.empty((0, 0), dtype=torch.float, device=pos.device)

@jaxtyped(typechecker=typechecker)
def compute_vector_edge_features(x: Union[Data, Batch], features: Union[List[str], ListConfig], idx: int=0) -> torch.Tensor:
    vector_edge_features = []
    pos = x.coords[:, idx][:, 1]
    
    # Get edge_index for this conformation and filter padding
    edge_index = x.edge_index[idx]
    valid_mask = (edge_index[0] != SEQ_MASK) & (edge_index[1] != SEQ_MASK)
    if valid_mask.any():
        edge_index = edge_index[:, valid_mask]
    else:
        return torch.empty((0, 0, 3), dtype=torch.float, device=pos.device)
    
    for feature in features:
        if feature == "edge_vectors":
            E_vectors = pos[edge_index[0]] - pos[edge_index[1]]
            vector_edge_features.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    
    return torch.cat(vector_edge_features, dim=0) if vector_edge_features else torch.empty((0, 0, 3), dtype=torch.float, device=pos.device)
