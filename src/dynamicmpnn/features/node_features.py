# Inspired from https://github.com/a-r-j/ProteinWorkshop

from typing import List, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.angles import (
    alpha,
    dihedrals,
    kappa,
    sidechain_torsion,
)
from graphein.protein.tensor.data import Protein, ProteinBatch
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

from dynamicmpnn.modules.models.utils import flatten_list
from dynamicmpnn.types import ScalarNodeFeature
from dynamicmpnn.features.utils import get_masked_sequence, orientations


@jaxtyped(typechecker=typechecker)
def compute_scalar_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
    idx: int = 0,
) -> torch.Tensor:
    """
    Factory function for node features.

    .. seealso::
        :py:class:`proteinworkshop.types.ScalarNodeFeature` for a list of node
        features that can be computed.

    This function operates on a :py:class:`torch_geometric.data.Data` or
    :py:class:`torch_geometric.data.Batch` object and computes the requested
    node features.

    :param x: :py:class:`~torch_geometric.data.Data` or
        :py:class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param node_features: List of node features to compute.
    :type node_features: Union[List[str], ListConfig]
    :return: Tensor of node features of shape (``N x F``), where ``N`` is the
        number of nodes and ``F`` is the number of features.
    :rtype: torch.Tensor
    """
    feats = []
    for feature in node_features:
        if feature == "sequence":
            feats.append(get_masked_sequence(x.residue_type[:, idx], x.mask_seq[:, idx]))
        elif feature == "alpha":
            feats.append(alpha(x.coords[:, idx], rad=True, embed=True))
        elif feature == "kappa":
            feats.append(kappa(x.coords[:, idx], rad=True, embed=True))
        elif feature == "dihedrals":
            feats.append(dihedrals(x.coords[:, idx], rad=True, embed=True))
        elif feature == "sidechain_torsions":
            feats.append(
                sidechain_torsion(
                    x.coords[:, idx],
                    res_types=flatten_list(x.residues),
                    rad=True,
                    embed=True,
                )
            )
        elif feature == "sequence_positional_encoding":
            continue
        else:
            raise ValueError(f"Node feature {feature} not recognised.")
    feats = [feat.unsqueeze(-1) if feat.ndim == 1 else feat for feat in feats]
    # Return concatenated features or original features if no features were computed
    return torch.cat(feats, dim=1) if feats else x.x

@jaxtyped(typechecker=typechecker)
def compute_vector_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    vector_features: Union[ListConfig, List[str]],
    idx: int = 0,
) -> torch.Tensor:
    """Factory function for vector features.

    Currently implemented vector features are:

        - ``orientation``: Orientation of each node in the protein backbone
        - ``virtual_cb_vector``: Virtual CB vector for each node in the protein
        backbone
    """
    vector_node_features = []
    for feature in vector_features:
        if feature == "orientation":
            vector_node_features.append(orientations(x.coords[:, idx]))
        elif feature == "virtual_cb_vector":
            raise NotImplementedError("Virtual CB vector not implemented yet.")
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    # x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    # return x
    return torch.cat(vector_node_features, dim=1)
