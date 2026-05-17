# Inspired from https://github.com/a-r-j/ProteinWorkshop

import functools
from typing import List, Literal, Optional, Tuple, Union

from dynamicmpnn.types import HOMOMER_NEGATIVE
import graphein.protein.tensor.edges as gp
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
from torch_geometric.utils import coalesce, to_undirected


@typechecker
def compute_edges(
    x: Union[Data, Batch, Protein, ProteinBatch],
    edge_types: Union[ListConfig, List[str]],
    idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orchestrates the computation of edges for a given data object.

    This function returns a tuple of tensors, where the first tensor is a
    tensor indicating the edge type of shape (``|E|``) and the second are the
    edge indices of shape (``2 x |E|``).

    The edge type tensor can be used to mask out edges of a particular type
    downstream.

    .. warning::

        For spatial edges, (e.g. ``knn_``, ``eps_``), the input data/batch
        object must have a ``pos`` attribute of shape (``N x 3``).

    :param x: The input data object to compute edges for
    :type x: Union[Data, Batch, Protein, ProteinBatch]
    :param edge_types: List of edge types to compute. Must be a sequence of
        ``knn_{x}``, ``eps_{x}``, (where ``{x}`` should be replaced by a
        numerical value) ``seq_forward``, ``seq_backward``.
    :type edge_types: Union[ListConfig, List[str]]
    :raises ValueError: Raised if ``x`` is not a ``torch_geometric`` Data or
        Batch object
    :raises NotImplementedError: Raised if an edge type is not implemented
    :return: Tuple of tensors, where the first tensor is a tensor indicating
        the edge type of shape (``|E|``) and the second are the edge indices of
        shape (``2 x |E|``).
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    # Handle batch
    if isinstance(x, Batch):
        edge_fn = functools.partial(gp.compute_edges, batch=x.batch)
    elif isinstance(x, Data):
        edge_fn = gp.compute_edges
    else:
        raise ValueError("x must be a torch_geometric Data or Batch object")
    # Iterate over edge types
    edges = []
    for edge_type in edge_types:
        if edge_type.startswith("knn") or edge_type.startswith("eps"):
            pos = x.coords[:, idx][:, 1]
            edges.append(edge_fn(pos, edge_type))
        elif edge_type.startswith("random_log"):
            parts = edge_type.split("_")
            k_total = int(parts[2])
            k_local = int(parts[3])
            edges.append(local_first_random_log_edges(x, k_total, k_local, idx))
        elif edge_type == "seq_forward":
            edges.append(sequence_edges(x, chains=x.chains[:, idx], direction="forward"))
        elif edge_type == "seq_backward":
            edges.append(sequence_edges(x, chains=x.chains[:, idx], direction="backward"))
        else:
            raise NotImplementedError(f"Edge type {edge_type} not implemented")

    # Compute edge types
    edge_types = []
    undirected_edges = []
    for e_idx in edges:
        src_nodes = e_idx[0, :]
        tgt_nodes = e_idx[1, :]

        src_chains = x.chains[:, idx][src_nodes]
        tgt_chains = x.chains[:, idx][tgt_nodes]
        src_homo_ids = x.homo_idx[:, idx][src_nodes]
        tgt_homo_ids = x.homo_idx[:, idx][tgt_nodes]

        is_inter_chain = (src_chains != tgt_chains)
        src_is_homomer = (src_homo_ids != HOMOMER_NEGATIVE)
        tgt_is_homomer = (tgt_homo_ids != HOMOMER_NEGATIVE)

        new_edge_type = torch.full_like(src_nodes, fill_value=2, dtype=torch.long)

        # Set type 0: Intra-chain edge within a homomer chain
        # For an intra-chain edge, we only need to check if the source is a homomer.
        mask_type_0 = (~is_inter_chain) & src_is_homomer
        new_edge_type[mask_type_0] = 0

        # Set type 1: Inter-chain edge involving at least one homomer chain
        mask_type_1 = is_inter_chain & (src_is_homomer | tgt_is_homomer)
        new_edge_type[mask_type_1] = 1

        # Coalesce and convert to undirected in one step
        undirected_edge_index, undirected_edge_type = to_undirected(coalesce(e_idx), edge_attr=new_edge_type)
        edge_types.append(undirected_edge_type)
        undirected_edges.append(undirected_edge_index)

    indxs = torch.cat(edge_types, dim=0) # In case there are more types of edges
    edges = torch.cat(undirected_edges, dim=1) # In case there are more types of edges

    return edges, indxs

@typechecker
def local_first_random_log_edges(
    x: Union[Data, Protein],
    k_total: int,
    k_local: int,
    idx: int = 0,
) -> torch.Tensor:
    """
    Computes edges using a two-step "local-first" approach.
    
    1. Deterministically selects the `k_local` nearest neighbors (k-NN).
    2. Probabilistically samples `k_total - k_local` additional neighbors
       from the remaining candidates using the 'random_log' method.

    Args:
        x (Union[Data, Protein]): Input data object.
        k_total (int): The total number of neighbors for each node.
        k_local (int): The number of local neighbors to select via k-NN first.
        idx (int): The conformation index to use.

    Returns:
        torch.Tensor: Tensor of shape (2, |E|) containing the combined edge indices.
    """
    pos = x.coords[:, idx][:, 1]
    num_nodes = pos.shape[0]
    k_random = k_total - k_local
    
    if k_random < 0:
        raise ValueError("k_total must be greater than or equal to k_local.")

    # --- Step 1: Secure Local Edges via k-NN ---
    
    # Compute pairwise distance matrix
    D = torch.cdist(pos, pos)
    
    # Create a mask for valid neighbors (intra-chain and not self)
    chain_ids = x.chains[:, idx]
    mask = chain_ids.unsqueeze(1) == chain_ids.unsqueeze(0)
    mask.fill_diagonal_(False)
    
    # Apply mask to distances
    D_masked = D.clone()
    D_masked[~mask] = torch.inf
    
    # Find the k_local nearest neighbors
    _, local_indices = torch.topk(
        D_masked, k=min(k_local, num_nodes - 1), dim=-1, largest=False
    )
    
    # Construct edge index for local edges
    src_nodes_local = torch.arange(num_nodes, device=pos.device).repeat_interleave(
        local_indices.shape[1]
    )
    edge_index_local = torch.stack([src_nodes_local, local_indices.flatten()], dim=0)

    # --- Step 2: Sample Long-Range Edges from Remaining Candidates ---

    # Create a mask to exclude the already-selected local neighbors
    remaining_mask = torch.ones_like(mask)
    remaining_mask.scatter_(1, local_indices, False)
    
    # Combine with the original mask
    final_mask = mask & remaining_mask
    
    # Perturb distances using the Gumbel-Max trick
    logp_edge = -3.0 * torch.log(D + 1e-6)
    z = torch.rand_like(logp_edge)
    gumbel_noise = -torch.log(-torch.log(z))
    D_key = logp_edge + gumbel_noise
    D_key[~final_mask] = -torch.inf # Apply final mask

    # Select top k_random neighbors from the remaining candidates
    _, random_indices = torch.topk(
        D_key, k=min(k_random, final_mask.sum(dim=-1).min()), dim=-1
    )

    # Construct edge index for random edges
    src_nodes_random = torch.arange(num_nodes, device=pos.device).repeat_interleave(
        random_indices.shape[1]
    )
    edge_index_random = torch.stack([src_nodes_random, random_indices.flatten()], dim=0)

    # --- Step 3: Combine Edges ---
    return torch.cat([edge_index_local, edge_index_random], dim=1)

@typechecker
def sequence_edges(
    b: Union[Data, Batch, Protein, ProteinBatch],
    chains: Optional[torch.Tensor] = None,
    direction: Literal["forward", "backward"] = "forward",
):
    """Computes edges between adjacent residues in a sequence.

    :param b: Input data object to compute edges for
    :type b: Union[Data, Batch, Protein, ProteinBatch]
    :param chains: Tensor of shape (``N``) indicating the chain ID of each node.
        This is required for correct boundary handling. Defaults to ``None``
    :type chains: Optional[torch.Tensor], optional
    :param direction: Direction of edges to compute. Must be ``forward`` or ``backward``. Defaults to ``forward``
    :type direction: Literal["forward", "backward"], optional
    :raises ValueError: Raised if ``direction`` is not ``forward`` or ``backward``
    :return: Tensor of shape (``2 x |E|``) indicating the edge indices
    """
    if isinstance(b, Batch):
        idx_a = torch.arange(0, b.ptr[-1] - 1, device=b.ptr.device)
        idx_b = torch.arange(1, b.ptr[-1], device=b.ptr.device)
    elif isinstance(b, Data):
        idx_a = torch.arange(0, b.coords.shape[0] - 1, device=b.coords.device)
        idx_b = torch.arange(1, b.coords.shape[0], device=b.coords.device)
    # Concatenate indices to create edge list
    if direction == "forward":
        e_index = torch.stack([idx_a, idx_b], dim=0)
    elif direction == "backward":
        e_index = torch.stack([idx_b, idx_a], dim=0)
    else:
        raise ValueError(f"Unknown direction: {direction}. Must be 'forward' or 'backward'")
    # Remove edges that cross batch boundaries
    if isinstance(b, Batch):
        mask = torch.ones_like(idx_a, device=b.coords.device).bool()
        mask[b.ptr[1:-2]] = 0
        e_index = e_index[:, mask]
    if chains is None and isinstance(b, Batch):
        chains = b.chains
    if chains is not None:
        # Remove edges between chains
        e_mask = chains[e_index]
        e_mask = (e_mask[0, :] - e_mask[1, :]) == 0
        e_index = e_index[:, e_mask]
    return e_index
