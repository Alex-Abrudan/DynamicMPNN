from loguru import logger
import random
import numpy as np

import torch
from torch_geometric.nn.encoding import PositionalEncoding

from torch_geometric.data import Batch, Data
from torch_geometric.utils import coalesce, to_undirected

from dynamicmpnn.features.edge_features import (
    compute_scalar_edge_features,
    compute_vector_edge_features,
)
from dynamicmpnn.features.edges import compute_edges
from dynamicmpnn.features.node_features import (
    compute_scalar_node_features,
    compute_vector_node_features,
)
from dynamicmpnn.types import (
    DISTANCE_EPS,
    FILL_VALUE,
    HOMOMER_NEGATIVE,
    INDEX_CHAIN_PAD,
    RESIDUE_TYPE_PAD,
    SEQ_MASK,
    List,
    ScalarEdgeFeature,
    ScalarNodeFeature,
    VectorEdgeFeature,
    VectorNodeFeature,
)


class ProteinGraphFeaturiser(object):
    """Protein Graph Featurizer

    Builds backbone protein graphs from a list of conformations.

    Returned graph is of type `torch_geometric.data.Data` with attributes:
    - seq        sequence converted to int tensor, shape [num_nodes]
    - node_s     node scalar features, shape [num_nodes, num_conf, num_bb_atoms x 5]
    - node_v     node vector features, shape [num_nodes, num_conf, 2 + (num_bb_atoms - 1), 3]
    - edge_s     edge scalar features, shape [num_edges, num_conf, num_bb_atoms x num_rbf + num_posenc + num_bb_atoms]  # noqa E501
    - edge_v     edge vector features, shape [num_edges, num_conf, num_bb_atoms, 3]
    - edge_index edge indices, shape [2, num_edges]
    - mask       node mask, `False` for nodes with missing data

    Args:
        split: train/validation/test split; coords are noised during training
        num_rbf: number of radial basis functions
        num_posenc: number of positional encodings per edge
        max_num_conformers: maximum number of conformers sampled per sequence
        noise_scale: standard deviation of gaussian noise added to coordinates
    """

    def __init__(
        self,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
        representation="ca",
        split="train",
        threshold=0.7,
        noise_scale=0.1,
        distance_eps=DISTANCE_EPS,
        device="cpu",
        pair_tm_tau: float = 0.0,
    ):
        super().__init__()

        self.scalar_node_features = scalar_node_features
        self.vector_node_features = vector_node_features
        self.edge_types = edge_types
        self.scalar_edge_features = scalar_edge_features
        self.vector_edge_features = vector_edge_features
        self.representation = representation
        self.split = split
        self.noise_scale = noise_scale
        self.distance_eps = distance_eps
        self.threshold = threshold
        self.device = device
        self.pair_tm_tau = pair_tm_tau

        if "sequence_positional_encoding" in self.scalar_node_features:
            self.positional_encoding = PositionalEncoding(16)

    def _validate_input(self):
        if len(self.scalar_node_features) == 0:
            raise ValueError("No scalar node features specified!")

    def add_node_features(self, pyg_object):
        """
        Adds node features to batched conformations of a protein using graphein's optimized batch featurization
        functions. Based on ProteinWorkshop and can create most of the features available in it.
        """
        if pyg_object.conf_mask[1] == False:
            # Only compute features for the first conformation, then duplicate
            node_s_0 = compute_scalar_node_features(pyg_object, self.scalar_node_features, idx=0)
            node_v_0 = compute_vector_node_features(pyg_object, self.vector_node_features, idx=0)
            
            # Duplicate the first conformation's features
            node_s = torch.stack([node_s_0, node_s_0.clone()], dim=1)
            node_v = torch.stack([node_v_0, node_v_0.clone()], dim=1)
            
            # Handle positional encoding similarly
            if hasattr(self, "positional_encoding"):
                pos_enc_0 = self.positional_encoding(pyg_object.residue_index[:, 0])
                pos_encodings = torch.stack([pos_enc_0, pos_enc_0.clone()], dim=1)
                node_s = torch.cat([node_s, pos_encodings], dim=-1)
        
        else:
            # Both conformations are valid - compute normally
            node_s = torch.stack([
                compute_scalar_node_features(pyg_object, self.scalar_node_features, idx=i)
                for i in range(pyg_object.coords.shape[1])
            ], dim=1)
            node_v = torch.stack([
                compute_vector_node_features(pyg_object, self.vector_node_features, idx=i)
                for i in range(pyg_object.coords.shape[1])
            ], dim=1)

            # Add positional encoding if available
            if hasattr(self, "positional_encoding"):
                pos_encodings = torch.stack([
                    self.positional_encoding(pyg_object.residue_index[:, i]) 
                    for i in range(pyg_object.residue_index.shape[1])
                ], dim=1)
                node_s = torch.cat([node_s, pos_encodings], dim=-1)
        
        pyg_object.node_s = node_s  # Shape: [2, num_nodes, num_scalar_features]
        pyg_object.node_v = node_v  # Shape: [2, num_nodes, num_vector_features, 3]
        pyg_object.node_s = torch.nan_to_num(pyg_object.node_s, nan=0.0, posinf=0.0, neginf=0.0)
        pyg_object.node_v = torch.nan_to_num(pyg_object.node_v, nan=0.0, posinf=0.0, neginf=0.0)
            
        return pyg_object

    def add_edge_features(self, pyg_object, fill_value=HOMOMER_NEGATIVE):
        """
        Simplified edge feature computation with virtual nodes.
        No more scatter_indices or original_lengths needed!
        """
        # TODO update the edge_type
        # TODO implement simplified case for duplicated conformations
        # Encoder edge index and features
        n_confs = pyg_object.conf_mask.sum().item()

        edge_results = [compute_edges(pyg_object, self.edge_types, i)
                        for i in range(n_confs)]
        edge_indexes = [result[0] for result in edge_results]
        edge_types = [result[1] for result in edge_results]

        # Step 2: Filter out virtual-real edges if virtual mask exists
        filtered_results = [
            self.filter_virtual_edges(edge_indexes[i], edge_types[i], pyg_object.virtual_mask[:, i])
            for i in range(n_confs)
        ]
        edge_indexes = [result[0] for result in filtered_results]
        edge_types = [result[1] for result in filtered_results]
        
        pyg_object.edge_index = edge_indexes
        pyg_object.edge_type = edge_types

        # Step 4: NOW compute edge features (after edge_index is available)
        edge_s = []
        edge_v = []
        
        if self.scalar_edge_features:
            edge_s = [compute_scalar_edge_features(pyg_object, self.scalar_edge_features, i) # edge_type is used here
                    for i in range(n_confs)]
        
        if self.vector_edge_features:
            edge_v = [compute_vector_edge_features(pyg_object, self.vector_edge_features, i)
                    for i in range(n_confs)]

        if n_confs == 1: 
            pyg_object.edge_s = edge_s[0]
            pyg_object.edge_v = edge_v[0]
            pyg_object.edge_index = edge_indexes[0]
            non_padding_offset = 0
        
        else: 
            pyg_object.edge_s = torch.cat(edge_s, dim=0)
            pyg_object.edge_v = torch.cat(edge_v, dim=0)
            
            non_padding_offset = pyg_object.padding_mask[:, 0].sum()
            edge_indexes[1] += non_padding_offset
            pyg_object.edge_index = torch.cat(edge_indexes, dim=1)

        # Decoder edge index and features
        homo_ids = [torch.unique(pyg_object.homo_idx[:, i])[torch.unique(pyg_object.homo_idx[:, i]) != HOMOMER_NEGATIVE]
            for i in range(n_confs)]

        unique_id_counter = 0
        chain_data = []
        max_node_idx = pyg_object.coords.shape[0] * pyg_object.coords.shape[1]
        node_to_chain = torch.full((max_node_idx + 1,), -1, dtype=torch.long, device=pyg_object.edge_index.device)

        # In this loop work with global indices
        for conf_idx in range(n_confs):
            for hid in homo_ids[conf_idx]:
                indices = torch.where(pyg_object.homo_idx[:, conf_idx] == hid)[0]

                if conf_idx == 1:
                    offset_indices = indices + non_padding_offset
                else:
                    offset_indices = indices

                node_to_chain[offset_indices] = unique_id_counter

                chain_data.append({
                    'offset_indices': offset_indices,
                    'unique_id': unique_id_counter,
                })
                unique_id_counter += 1

        # Step 1: Create global node-to-chain mapping (O(total_nodes))
        original_indices = torch.arange(pyg_object.edge_index.shape[1])

        src_chains = node_to_chain[pyg_object.edge_index[0]]  # [n_edges]
        dst_chains = node_to_chain[pyg_object.edge_index[1]]  # [n_edges]

        mask = (src_chains >= 0) & (dst_chains >= 0) & (src_chains == dst_chains)
        intra_chain_edges = pyg_object.edge_index[:, mask]
        intra_chain_original_indices = original_indices[mask]

        chain_local_edges = []
        chain_original_indices = []
        start_node_indices = [chain["offset_indices"][0] for chain in chain_data]

        for i, start_node in enumerate(start_node_indices):
            mask_for_this_chain = (src_chains[mask] == i)
            edges_for_this_chain = intra_chain_edges[:, src_chains[mask] == i]
            local_edges = edges_for_this_chain - start_node
            chain_local_edges.append(local_edges)

            original_indices_for_this_chain = intra_chain_original_indices[mask_for_this_chain]
            chain_original_indices.append(original_indices_for_this_chain)

        all_local_edges = torch.cat(chain_local_edges, dim=1)
        edge_reunion = coalesce(to_undirected(all_local_edges))

        num_chains = len(chain_data)
        num_reunion_edges = edge_reunion.shape[1]

        # Create a stable, canonical representation of the search space once.
        keys_s = edge_reunion[0].long() << 32 | edge_reunion[1].long()
        s_keys_sorted, s_original_indices = torch.sort(keys_s)

        reunion_features_s_stacked = torch.zeros(num_chains, num_reunion_edges, pyg_object.edge_s.shape[-1])
        reunion_features_v_stacked = torch.zeros(num_chains, num_reunion_edges, *pyg_object.edge_v.shape[1:])
        concatenated_mask_stacked = torch.zeros(num_chains, num_reunion_edges, dtype=torch.bool)

        for i in range(num_chains):
            local_edges = chain_local_edges[i]
            original_idxs = chain_original_indices[i]

            reverse_local_edges = local_edges.flip(0) # Swaps rows so (A,B) becomes (B,A)
            mask_fwd, reunion_indices_fwd = self.find_edge_indices(local_edges, s_keys_sorted, s_original_indices)
            mask_rev, reunion_indices_rev = self.find_edge_indices(reverse_local_edges, s_keys_sorted, s_original_indices)

            # Combine the indices and the original feature indices that need to be gathered
            valid_reunion_indices = torch.cat([reunion_indices_fwd[mask_fwd], reunion_indices_rev[mask_rev]])
            valid_original_idxs = torch.cat([original_idxs[mask_fwd], original_idxs[mask_rev]])

            concatenated_mask_stacked[i, valid_reunion_indices] = True

            features_s_to_scatter = pyg_object.edge_s[valid_original_idxs]
            features_v_to_scatter = pyg_object.edge_v[valid_original_idxs]

            reunion_features_s_stacked[i, valid_reunion_indices] = features_s_to_scatter
            reunion_features_v_stacked[i, valid_reunion_indices] = features_v_to_scatter

        pyg_object.decoder_edge_index = edge_reunion
        pyg_object.decoder_edge_idx_length = num_reunion_edges
        pyg_object.decoder_edge_s = reunion_features_s_stacked.view(-1, reunion_features_s_stacked.shape[-1]) 
        pyg_object.decoder_edge_v = reunion_features_v_stacked.view(-1, *reunion_features_v_stacked.shape[2:])
        pyg_object.decoder_edge_feat_length = pyg_object.decoder_edge_s.shape[0]
        pyg_object.decoder_edge_mask = concatenated_mask_stacked.view(-1)

        return pyg_object
    
    def find_edge_indices(self, edges_to_find, s_keys_sorted, s_original_indices): 

        keys_f = edges_to_find[0].long() << 32 | edges_to_find[1].long()
        
        insertion_points = torch.searchsorted(s_keys_sorted, keys_f)
        insertion_points.clamp_(max=s_keys_sorted.shape[0] - 1)
        found_keys = s_keys_sorted[insertion_points]
        mask = (found_keys == keys_f)
        found_indices = s_original_indices[insertion_points]
        
        return mask, found_indices

    def filter_virtual_edges(self, edge_index, edge_type, virtual_mask):
        """Remove edges between virtual and real nodes, return filtered data and mask"""
        src, dst = edge_index

        both_real = virtual_mask[src] & virtual_mask[dst]

        return edge_index[:, both_real], edge_type[both_real], both_real

    @staticmethod
    def virtualize_masked_nodes(protein, masks):  # Use large positive value instead of -1e-5
        """
        Instead of removing masked nodes, move them to +infinity and mark valid nodes in virtual_mask.
        """
        nodes_to_virtualize = ~masks
    
        # Move coordinates of INVALID nodes to +infinity
        protein.coords[:, 0][nodes_to_virtualize[:, 0]] = FILL_VALUE
        protein.coords[:, 1][nodes_to_virtualize[:, 1]] = FILL_VALUE
        
        # Store the original mask, which correctly identifies VALID nodes
        protein.virtual_mask = masks
        
        return protein

    def _call_wrapper(self, protein):
        """Simple wrapper function to avoid large tab in __call__."""
        with torch.no_grad():
            return self.__call__(protein)

    def stack_conformations(self, confs_list, pdb_chains: List[str]):
        """
        Stack and pad 2 conformations with appropriate padding values.
        """
        if len(confs_list) > 2:
            raise ValueError(f"Expected 1 or 2 conformations, got {len(confs_list)}")
        
        elif len(confs_list) == 1:
            conf = confs_list[0]
            max_len = len(conf.residue_type)
            
            # Create stacked tensors with appropriate padding
            stacked_coords = conf.coords.unsqueeze(1)  # [1, num_residues, 3, 3]
            stacked_residue_type = conf.residue_type.unsqueeze(1)  # [1, num_residues]
            stacked_residue_index = conf.residue_index.unsqueeze(1)
            stacked_chains = conf.chains.unsqueeze(1)
            stacked_homo_idx = conf.homo_idx.unsqueeze(1)
            stacked_mask_seq = conf.mask_seq.unsqueeze(1)
            padding_mask = torch.ones(max_len, 1, dtype=torch.bool)

            chain_ids = (pdb_chains[0].split('_')[1], pdb_chains[1].split('_')[1])
            chain_idx = (
                conf.homomer_mapping[chain_ids[0]], 
                conf.homomer_mapping[chain_ids[1]]
            )
            
            # Find where homo_idx matches the chains from pdb_chain
            mask1 = (conf.homo_idx == chain_idx[0])
            mask2 = (conf.homo_idx == chain_idx[1])
            stacked_pdb_chain_pair_mask = (mask1 | mask2).unsqueeze(1)

        else:
            n_confs = len(confs_list)
            conf1, conf2 = confs_list
            len1, len2 = len(conf1.residue_type), len(conf2.residue_type)
            max_len = max(len1, len2)
            
            # Create stacked tensors with appropriate padding
            stacked_coords = torch.full((max_len, n_confs, 3, 3), FILL_VALUE, dtype=torch.float32)
            stacked_residue_type = torch.full((max_len, n_confs), RESIDUE_TYPE_PAD, dtype=torch.long)
            stacked_residue_index = torch.full((max_len, n_confs), INDEX_CHAIN_PAD, dtype=torch.long)
            stacked_chains = torch.full((max_len, n_confs), INDEX_CHAIN_PAD, dtype=torch.long)
            stacked_homo_idx = torch.full((max_len, n_confs), HOMOMER_NEGATIVE, dtype=torch.long)
            stacked_mask_seq = torch.full((max_len, n_confs), False, dtype=torch.bool)
            padding_mask = torch.zeros(max_len, n_confs, dtype=torch.bool)
            
            for i, conf in enumerate([conf1, conf2]):
                length = len(conf.residue_type)
                stacked_coords[:length, i] = conf.coords
                stacked_residue_type[:length, i] = conf.residue_type
                stacked_residue_index[:length, i] = conf.residue_index
                stacked_chains[:length, i] = conf.chains
                stacked_homo_idx[:length, i] = conf.homo_idx
                stacked_mask_seq[:length, i] = conf.mask_seq
                padding_mask[:length, i] = True
            
            # Convert the pdb_chains and homomer_mappings objects into one torch Tensor object.
            chain_ids = (pdb_chains[0].split('_')[1], pdb_chains[1].split('_')[1])
            chain_idx = (
                confs_list[0].homomer_mapping[chain_ids[0]], 
                confs_list[1].homomer_mapping[chain_ids[1]]
            )
            
            # Find where homo_idx matches the chains from pdb_chain
            pdb_chain_pair_mask = (
                confs_list[0].homo_idx == chain_idx[0],
                confs_list[1].homo_idx == chain_idx[1],
            )
            stacked_pdb_chain_pair_mask = torch.zeros(max_len, n_confs, dtype=torch.bool)
            stacked_pdb_chain_pair_mask[:len1, 0] = pdb_chain_pair_mask[0]
            stacked_pdb_chain_pair_mask[:len2, 1] = pdb_chain_pair_mask[1]

        return Data(
            coords=stacked_coords,              # [2, max_len, 3, 3] - padded with 1e6
            residue_type=stacked_residue_type,  # [2, max_len] - padded with 22
            residue_index=stacked_residue_index, # [2, max_len] - padded with -1
            chains=stacked_chains,              # [2, max_len] - padded with -1
            homo_idx=stacked_homo_idx,      # [2, max_len] - padded with HOMOMER_NEGATIVE
            pdb_chain_pair_mask=stacked_pdb_chain_pair_mask,  # [2, max_len] - True for the positions of the 2 chains in pdb_chains
            mask_seq=stacked_mask_seq,        # [2, max_len] - True for the positions of the real residues
            padding_mask=padding_mask,          # [2, max_len] - True for real data
        )

    def __call__(self, protein, pdb_code: str = None):
        """
        Featurize a protein PyG Data object.

        Args:
            protein: Protein PyG Batch object where each batch is a protein with multiple conformations.
                - coords: Coordinates of the protein atoms (num_res, 3, 3). Only the 3 backbone atoms N, Ca, and C were kept from the 37 atom types - dim = 1.
                - residue_type: Amino acid sequence in integer code.
                - residue_id: Residue ID in the format "chain:three-letter-code:residue-number".
                - chains: List of chain indices of the residues.
        """
        try:
            # We check the first conformation as a representative sample.
            first_conf_key = protein.cluster_members[0].split('_')[0].upper()
            first_conf_data = protein.pyg_dict[first_conf_key]
            
            # This checks if there are any valid chain IDs assigned.
            if not torch.any(first_conf_data.homo_idx != HOMOMER_NEGATIVE):
                logger.warning(f"Skipping {pdb_code}: No valid homomer IDs found in homo_idx.")
                return None
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Skipping {pdb_code}: Malformed input protein object. Error: {e}")
            return None

        confs_list, pdb_chain = self.get_entries(protein) # TODO fix after TM score is implemented

        stacked_data = self.stack_conformations(confs_list, pdb_chains=pdb_chain)
        pyg_object = self.eliminate_gaps(stacked_data)

        mask = (~torch.isin(pyg_object.residue_type, torch.tensor([20, 21]))) & pyg_object.padding_mask
        if mask[:, 0].sum() < 10 or mask[:, 1].sum() < 10:
            print(f"Skipping protein due to insufficient valid coordinates (coords_mask has {mask.sum()} True elements).")
            return None  

        if self.split == "train":
            noise = torch.randn_like(pyg_object.coords) * self.noise_scale
            pyg_object.coords = pyg_object.coords + noise

        pyg_object.seq = self.get_sequence(pyg_object)
        if pyg_object.seq.shape[0] == 0:
            logger.warning(f"Skipping {pdb_code}: Consensus sequence is empty.")
            return None
        pyg_object = self.virtualize_masked_nodes(pyg_object, mask)     
        pyg_object = self.add_node_features(pyg_object)
        pyg_object = self.add_edge_features(pyg_object, mask)
        pyg_object = self.transform_coords(pyg_object, self.representation)
        pyg_object = self._flatten_for_pyg(pyg_object)

        pyg_object.num_decoder_nodes=len(pyg_object.seq)

        del pyg_object.residue_index
        del pyg_object.chains 
        del pyg_object.residue_type  
        del pyg_object.edge_type
        del pyg_object.padding_mask

        '''   Example of the final PyG object structure:
        coords=[n_nodes, 1, 3]                    # Protein coordinates: [num_residues, num_conformations, num_atoms, xyz]
                                                     # Contains backbone atom positions (CA only since representation="ca")
        seq=[n_nodes]                                # Target sequence: [num_residues]
                                                     # Consensus sequence across conformations for training
        homo_idx=[n_nodes, 2]                     # Homomer chain mapping: [num_residues, num_conformations]
                                                     # Maps residues to their homomer chain identity
        pdb_chain_pair_mask=[n_nodes, 2]            # Chain pair selection mask: [num_residues, num_conformations]
                                                     # Identifies residues belonging to the specific chain pair being studied
        padding_mask=[n_nodes, 2]                   # Padding indicator: [num_residues, num_conformations]
                                                     # True for real data, False for padding added during batching
        node_s=[n_nodes, 27]                     # Scalar node features: [num_residues, num_conformations, scalar_features]
                                                     # Contains residue-level scalar properties (AA type, secondary structure, etc.)
        node_v=[n_nodes, 2, 3]                   # Vector node features: [num_residues, num_conformations, vector_features, xyz]
                                                     # Contains directional features (backbone orientations, etc.)
        edge_index=[2, n_edges]                  # Encoder edge connectivity: [num_conformations, 2, max_edges]
                                                     # Source/destination node pairs for encoder graph (intra-conformation edges)
        edge_s=[n_edges, 34]                     # Scalar edge features: [num_conf, max_edges, scalar_features]
                                                     # Distance, RBF features, positional encodings, etc.
        edge_v=[n_edges, 1, 3]                   # Vector edge features: [num_conf, max_edges, vector_features, xyz]
                                                     # Directional edge properties (bond vectors, etc.)
        decoder_edge_index=[2, n_decoder_edges]     # Decoder edge connectivity: [2, num_decoder_edges]
                                                     # Cross-conformation and intra-chain edges for decoder
        decoder_edge_s=[n_decoder_edges, 34]        # Decoder scalar edge features: [num_decoder_edges, scalar_features]
                                                     # Same feature types as encoder but for decoder edges
        decoder_edge_v=[n_decoder_edges, 1, 3]      # Decoder vector edge features: [num_decoder_edges, vector_features, xyz]
                                                     # Directional features for decoder edges
        edge_mask=[2, n_edges]                       # Edge validity mask: [num_conf, max_edges]'''

        if pyg_object.decoder_edge_index.numel() > 0:
            max_edge_index = pyg_object.decoder_edge_index.max().item()
            assert max_edge_index < len(pyg_object.seq), \
                (f"CRITICAL DATA ERROR: decoder_edge_index contains an index ({max_edge_index}) "
                 f"that is out of bounds for the number of decoder nodes ({len(pyg_object.seq)}).")
        # Optional: you might want to consider a graph with no edges as invalid
        else:
            logger.warning(f"Skipping {pdb_code}: No edges were created in the graph.")
            return None

        return pyg_object

    def eliminate_gaps(self, pyg):
        n_confs = pyg.coords.shape[1]

        if n_confs == 1:
            homo_ids = [torch.unique(pyg.homo_idx[:, 0])[torch.unique(pyg.homo_idx[:, 0]) != HOMOMER_NEGATIVE]]

            chain_res_types = (
                [pyg.residue_type[:, 0][torch.where(pyg.homo_idx[:, 0] == hid)[0]] for hid in homo_ids[0]],
            )
            chain_indices = (
                [torch.where(pyg.homo_idx[:, 0] == hid)[0] for hid in homo_ids[0]],
            )
            
            # Now we can stack like in n_confs=2 case
            all_homomers = chain_res_types[0]
            stacked_homomers = torch.stack(all_homomers, dim=0)
            gap_mask = ((stacked_homomers == 20) | (stacked_homomers == 21)).all(dim=0)
            gap_positions = torch.where(gap_mask)[0]
            
            positions_to_remove = (
                torch.cat([chain[gap_positions] for chain in chain_indices[0]]),
            )
            
            keep_masks = [torch.ones_like(pyg.residue_type[:, 0], dtype=torch.bool)]
            keep_masks[0][positions_to_remove[0]] = False
            
            # Use the same filtering approach as n_confs=2
            filtered_tensors = {
                'coords': [pyg.coords[:, 0][keep_masks[0]]],
                'residue_type': [pyg.residue_type[:, 0][keep_masks[0]]],
                'residue_index': [pyg.residue_index[:, 0][keep_masks[0]]],
                'chains': [pyg.chains[:, 0][keep_masks[0]]],
                'homo_idx': [pyg.homo_idx[:, 0][keep_masks[0]]],
                'pdb_chain_pair_mask': [pyg.pdb_chain_pair_mask[:, 0][keep_masks[0]]],
                'mask_seq': [pyg.mask_seq[:, 0][keep_masks[0]]],
                'padding_mask': [pyg.padding_mask[:, 0][keep_masks[0]]],
            }

            coords, residue_type, residue_index, chains, homo_idx, pdb_chain_pair_mask, mask_seq, padding_mask = self.stack_and_pad_conformations(filtered_tensors, num_confs=1)

            coords = torch.stack([coords, coords.clone()], dim=1)
            residue_type = torch.stack([residue_type[0], residue_type[0].clone()], dim=1)
            residue_index = torch.stack([residue_index[0], residue_index[0].clone()], dim=1)
            chains = torch.stack([chains[0], chains[0].clone()], dim=1)
            homo_idx = torch.stack([homo_idx[0], homo_idx[0].clone()], dim=1)
            pdb_chain_pair_mask = torch.stack([pdb_chain_pair_mask[0], pdb_chain_pair_mask[0].clone()], dim=1)
            mask_seq = torch.stack([mask_seq[0], mask_seq[0].clone()], dim=1)
            padding_mask = torch.stack([padding_mask[0], padding_mask[0].clone()], dim=1)

            conf_mask = torch.tensor([True, False], dtype=torch.bool)

        elif n_confs == 2:
            # !!! For max_rmsd pooling, it is possible to have residues which are both missing
            homo_ids = [torch.unique(pyg.homo_idx[:, i])[torch.unique(pyg.homo_idx[:, i]) != HOMOMER_NEGATIVE]
                        for i in range(n_confs)]
            
            chain_res_types = (
                [pyg.residue_type[:, 0][torch.where(pyg.homo_idx[:, 0] == hid)[0]] for hid in homo_ids[0]],
                [pyg.residue_type[:, 1][torch.where(pyg.homo_idx[:, 1] == hid)[0]] for hid in homo_ids[1]]
            )
            chain_indices = (
                [torch.where(pyg.homo_idx[:, 0] == hid)[0] for hid in homo_ids[0]],
                [torch.where(pyg.homo_idx[:, 1] == hid)[0] for hid in homo_ids[1]]
            )
            
            all_homomers = chain_res_types[0] + chain_res_types[1]
            stacked_homomers = torch.stack(all_homomers, dim=0)
            gap_mask = ((stacked_homomers == 20) | (stacked_homomers == 21)).all(dim=0)
            gap_positions = torch.where(gap_mask)[0]
            
            positions_to_remove = (
                torch.cat([chain[gap_positions] for chain in chain_indices[0]]),
                torch.cat([chain[gap_positions] for chain in chain_indices[1]])
            )
            
            keep_masks = [torch.ones_like(pyg.residue_type[:, i], dtype=torch.bool) for i in range(n_confs)]
            keep_masks[0][positions_to_remove[0]] = False
            keep_masks[1][positions_to_remove[1]] = False

            filtered_tensors = {
                'coords': [pyg.coords[:, 0][keep_masks[0]], pyg.coords[:, 1][keep_masks[1]]],
                'residue_type': [pyg.residue_type[:, 0][keep_masks[0]], pyg.residue_type[:, 1][keep_masks[1]]],
                'residue_index': [pyg.residue_index[:, 0][keep_masks[0]], pyg.residue_index[:, 1][keep_masks[1]]],
                'chains': [pyg.chains[:, 0][keep_masks[0]], pyg.chains[:, 1][keep_masks[1]]],
                'homo_idx': [pyg.homo_idx[:, 0][keep_masks[0]], pyg.homo_idx[:, 1][keep_masks[1]]],
                'pdb_chain_pair_mask': [pyg.pdb_chain_pair_mask[:, 0][keep_masks[0]], pyg.pdb_chain_pair_mask[:, 1][keep_masks[1]]],
                'mask_seq': [pyg.mask_seq[:, 0][keep_masks[0]], pyg.mask_seq[:, 1][keep_masks[1]]],
                'padding_mask': [pyg.padding_mask[:, 0][keep_masks[0]], pyg.padding_mask[:, 1][keep_masks[1]]],
            }

            coords, residue_type, residue_index, chains, homo_idx, pdb_chain_pair_mask, mask_seq, padding_mask = self.stack_and_pad_conformations(filtered_tensors, num_confs=2)

            conf_mask = torch.tensor([True, True], dtype=torch.bool)

        return ProteinBatchData(
            coords=coords,
            residue_type=residue_type,
            residue_index=residue_index, 
            chains=chains, 
            homo_idx=homo_idx,
            pdb_chain_pair_mask=pdb_chain_pair_mask,
            mask_seq=mask_seq,
            padding_mask=padding_mask,
            conf_mask=conf_mask
        )

    def stack_and_pad_conformations(self, filtered_tensors, num_confs=2):
        if num_confs == 1:
            coords = filtered_tensors['coords'][0]
            residue_type = [filtered_tensors['residue_type'][0]]
            residue_index = [filtered_tensors['residue_index'][0]]
            chains = [filtered_tensors['chains'][0]]
            homo_idx = [filtered_tensors['homo_idx'][0]]
            pdb_chain_pair_mask = [filtered_tensors['pdb_chain_pair_mask'][0]]
            mask_seq = [filtered_tensors['mask_seq'][0]]
            padding_mask = [filtered_tensors['padding_mask'][0]]

            return coords, residue_type, residue_index, chains, homo_idx, pdb_chain_pair_mask, mask_seq, padding_mask

        elif num_confs == 2:
            max_len = max(len(filtered_tensors['residue_type'][i]) for i in range(num_confs))

            # Pre-allocate padded tensors
            coords = torch.full((max_len, num_confs, filtered_tensors['coords'][0].shape[1], filtered_tensors['coords'][0].shape[2]), FILL_VALUE, dtype=torch.float32)
            residue_type = torch.full((max_len, num_confs), RESIDUE_TYPE_PAD, dtype=torch.long)
            residue_index = torch.full((max_len, num_confs), INDEX_CHAIN_PAD, dtype=torch.long)
            chains = torch.full((max_len, num_confs), INDEX_CHAIN_PAD, dtype=torch.long)
            homo_idx = torch.full((max_len, num_confs), HOMOMER_NEGATIVE, dtype=torch.long)
            pdb_chain_pair_mask = torch.zeros(max_len, num_confs, dtype=torch.bool)  # False for padding
            stacked_mask_seq = torch.full((max_len, num_confs), False, dtype=torch.bool)  # False for padding
            padding_mask = torch.zeros(max_len, num_confs, dtype=torch.bool)  # False for padding

            for i in range(num_confs):
                length = len(filtered_tensors["coords"][i])
                coords[:length, i] = filtered_tensors["coords"][i]
                residue_type[:length, i] = filtered_tensors["residue_type"][i]
                residue_index[:length, i] = filtered_tensors["residue_index"][i]
                chains[:length, i] = filtered_tensors["chains"][i]
                homo_idx[:length, i] = filtered_tensors["homo_idx"][i]
                pdb_chain_pair_mask[:length, i] = filtered_tensors["pdb_chain_pair_mask"][i]
                stacked_mask_seq[:length, i] = filtered_tensors['mask_seq'][i]
                padding_mask[:length, i] = filtered_tensors['padding_mask'][i]

            return coords, residue_type, residue_index, chains, homo_idx, pdb_chain_pair_mask, stacked_mask_seq, padding_mask

    def get_sequence(self, pyg):

        if pyg.conf_mask[1] == False:
            n_confs = 1
            # Check if homo_idx exists and has elements
            if pyg.homo_idx is None or pyg.homo_idx.shape[0] == 0:
                logger.warning("get_sequence: homo_idx is empty or None for single conf.")
                return torch.tensor([], dtype=torch.long) # Return empty tensor
        else:
            n_confs = 2
            if pyg.homo_idx is None or pyg.homo_idx.shape[0] == 0:
                logger.warning("get_sequence: homo_idx is empty or None for dual conf.")
                return torch.tensor([], dtype=torch.long) # Return empty tensor
    
        if pyg.conf_mask[1] == False:
            n_confs = 1
            homo_ids = [torch.unique(pyg.homo_idx[:, 0])[torch.unique(pyg.homo_idx[:, 0]) != HOMOMER_NEGATIVE]]

            chain_res_types = (
                [pyg.residue_type[:, 0][torch.where(pyg.homo_idx[:, 0] == hid)[0]] for hid in homo_ids[0]],
            )
            
            # Now we can stack like in n_confs=2 case
            all_homomers = chain_res_types[0]
            stacked_homomers = torch.stack(all_homomers, dim=0)
            value_mask = torch.isin(stacked_homomers, torch.tensor([20, 21]))
            seq_length = stacked_homomers.shape[1]
            sequence = torch.zeros(seq_length, dtype=torch.long, device=stacked_homomers.device)

            for pos in range(seq_length):
                valid_residues = stacked_homomers[~value_mask[:, pos], pos]
                if len(valid_residues) > 0:
                    sequence[pos] = valid_residues[torch.randint(0, len(valid_residues), (1,))]
                else:
                    sequence[pos] = 20
                    print(f"No valid residues found at position {pos} in sequence {stacked_homomers}")

        elif pyg.conf_mask[1] == True:
            n_confs = 2
            homo_ids = [torch.unique(pyg.homo_idx[:, i])[torch.unique(pyg.homo_idx[:, i]) != HOMOMER_NEGATIVE]
                        for i in range(n_confs)]
            
            chain_res_types = (
                [pyg.residue_type[:, 0][torch.where(pyg.homo_idx[:, 0] == hid)[0]] for hid in homo_ids[0]],
                [pyg.residue_type[:, 1][torch.where(pyg.homo_idx[:, 1] == hid)[0]] for hid in homo_ids[1]]
            )

            all_homomers = chain_res_types[0] + chain_res_types[1]
            if not all_homomers:
                logger.warning("get_sequence: `all_homomers` list is empty before stacking. Skipping.")
                return torch.tensor([], dtype=torch.long)
            
            stacked_homomers = torch.stack(all_homomers, dim=0)
            value_mask = torch.isin(stacked_homomers, torch.tensor([20, 21]))
            seq_length = stacked_homomers.shape[1]
            sequence = torch.zeros(seq_length, dtype=torch.long, device=stacked_homomers.device)

            for pos in range(seq_length):
                valid_residues = stacked_homomers[~value_mask[:, pos], pos]
                if len(valid_residues) > 0:
                    sequence[pos] = valid_residues[torch.randint(0, len(valid_residues), (1,))]
                else:
                    sequence[pos] = 20
                    print(f"No valid residues found at position {pos} in sequence {stacked_homomers}")

        return sequence

    def transform_coords(self, pyg_object, representation="ca_bb"):
        """
        Transforms the coordinates of a protein PyG Data object to a different representation.
        Currently supports:
        - "ca_bb": C-alpha and backbone atoms
        - "ca": C-alpha atoms only

        Args:
            protein (Data): Protein PyG Data object containing multiple conformations. Specifically, the coords
                attribute for each is assumed to have shape (num_residues, num_atoms, 3); num_atoms is typically 37.
            representation (str): Representation to transform the coordinates to
        Returns:
            protein (Data): Transformed Protein PyG Data object containing multiple conformations
        """
        if representation == "ca_bb":
            idx = 3
        elif representation == "ca":
            idx = 1
        else:
            raise ValueError(f"Invalid representation: {representation}")
        
        if idx == 1:
            pyg_object.coords = pyg_object.coords[:, :, idx].unsqueeze(2)
        elif idx == 3:
            pyg_object.coords = pyg_object.coords[:, :, :idx]

        return pyg_object
    
    def compute_tm_probabilities(self, tm_scores, pairs, temperature=0.3):
        """Convert TM scores to sampling probabilities.

        Lower TM scores (more dissimilar structures) get higher probabilities.

        Args:
            tm_scores: 2D torch tensor of TM scores
            pairs: List of (i, j) index pairs
            temperature: Controls sharpness (lower = sharper, higher = more uniform)

        Returns:
            probabilities: numpy array of probabilities for each pair
        """
        pair_tm_scores = np.array([tm_scores[i, j].item() for i, j in pairs], dtype=np.float32)

        # Handle NaN values
        nan_mask = np.isnan(pair_tm_scores)
        if nan_mask.all():
            return np.ones(len(pairs), dtype=np.float32) / len(pairs)

        pair_tm_scores = np.nan_to_num(pair_tm_scores, nan=0.0)

        # Transform: lower TM -> higher probability
        dissimilarity = 1.0 - pair_tm_scores
        logits = dissimilarity / temperature
        logits = logits - logits.max()  # Numerical stability
        probs = np.exp(logits)
        probs = probs / probs.sum()

        return probs

    def get_entries(self, protein):
        """Returns 2 entries, optionally sampled by TM-score dissimilarity.

        When pair_tm_tau > 0 and protein has tm_scores, pairs with lower TM scores
        (more structurally dissimilar) are sampled with higher probability.

        Args:
            protein: Protein object with cluster_members, pyg_dict, and optionally tm_scores

        Returns:
            confs_list (list): List of selected conformer PyG Data objects
            pdb_chain (list): List of selected chain IDs
        """
        n = len(protein.cluster_members)
        conf_ids = protein.cluster_members
        pdb_ids = [member.split('_')[0].upper() for member in conf_ids]

        if n == 2:
            # Only 2 conformations - no choice to make
            pdb_chain = [conf_ids[i] for i in range(2)]
            if pdb_ids[0] == pdb_ids[1]:
                confs_list = [protein.pyg_dict[pdb_ids[0]]]
            else:
                confs_list = [protein.pyg_dict[pdb_ids[0]], protein.pyg_dict[pdb_ids[1]]]

        elif n == 1:
            raise ValueError(f"Only one conformation found in cluster with members {protein.cluster_members}")

        else:
            # n >= 3: select a pair
            use_tm_sampling = (
                self.pair_tm_tau > 0.0
                and hasattr(protein, 'tm_scores')
                and protein.tm_scores is not None
            )

            if use_tm_sampling:
                # Generate all possible pairs
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                probs = self.compute_tm_probabilities(protein.tm_scores, pairs, temperature=self.pair_tm_tau)
                pair_idx = np.random.choice(len(pairs), p=probs)
                selected_indices = list(pairs[pair_idx])
            else:
                # Uniform random sampling
                selected_indices = random.sample(range(n), 2)

            pdb_chain = [conf_ids[selected_indices[i]] for i in range(2)]
            pdbs = [pdb_chain[i].split('_')[0].upper() for i in range(2)]
            confs_list = [protein.pyg_dict[pdb] for pdb in pdbs]

        return confs_list, pdb_chain

    def _flatten_for_pyg(self, pyg_object):
        """Flatten all tensors to PyG-compatible format and remove padding"""
        n_nodes = pyg_object.coords.shape[0]
        n_confs = pyg_object.coords.shape[1]
        valid_mask = pyg_object.padding_mask  # Shape: [n_nodes, n_confs]

        if pyg_object.conf_mask[1] == False:

            valid_mask_flat = valid_mask[:, 0]

            coords_flat = pyg_object.coords[:, 0]
            node_s_flat = pyg_object.node_s[:, 0]
            node_v_flat = pyg_object.node_v[:, 0]
            virtual_mask_flat = pyg_object.virtual_mask[:, 0]
            homo_idx_flat = pyg_object.homo_idx[:, 0]
            pdb_chain_pair_mask_flat = pyg_object.pdb_chain_pair_mask[:, 0]

            homo_idx_flat = pyg_object.homo_idx[:, 0]

        else:
            # Create flattened valid mask
            valid_mask_flat = valid_mask.transpose(0, 1).reshape(-1)  # Shape: [n_nodes * n_confs]
            
            coords_flat = pyg_object.coords.transpose(0, 1).reshape(n_nodes * n_confs, 1, 3)
            node_s_flat = pyg_object.node_s.transpose(0, 1).reshape(n_nodes * n_confs, -1)
            node_v_flat = pyg_object.node_v.transpose(0, 1).reshape(n_nodes * n_confs, -1, 3)
            virtual_mask_flat = pyg_object.virtual_mask.transpose(0, 1).reshape(-1)
            homo_idx_flat = pyg_object.homo_idx.transpose(0, 1).reshape(-1)
            pdb_chain_pair_mask_flat = pyg_object.pdb_chain_pair_mask.transpose(0, 1).reshape(-1)

            homo_idx_with_offset = pyg_object.homo_idx.clone()  # [n_nodes, n_confs]
            max_homo_id = pyg_object.homo_idx[:, 0][pyg_object.homo_idx[:, 0] != HOMOMER_NEGATIVE].max()
            second_conf_mask = homo_idx_with_offset[:, 1] != HOMOMER_NEGATIVE
            homo_idx_with_offset[:, 1][second_conf_mask] += (max_homo_id + 1)
            homo_idx_flat = homo_idx_with_offset.transpose(0, 1).reshape(-1)
        
        pyg_object.coords = coords_flat[valid_mask_flat]
        pyg_object.node_s = node_s_flat[valid_mask_flat]
        pyg_object.node_v = node_v_flat[valid_mask_flat]
        pyg_object.virtual_mask = virtual_mask_flat[valid_mask_flat]
        pyg_object.homo_idx = homo_idx_flat[valid_mask_flat]
        pyg_object.pdb_chain_pair_mask = pdb_chain_pair_mask_flat[valid_mask_flat]
        
        return pyg_object


class ProteinBatchData(Data):
    """
    A custom PyG Data class to handle inconsistent node counts between
    the full structure (for the encoder) and the decoder's union graph.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'decoder_edge_index':
            return self.seq.shape[0]

        # For all other attributes, use the standard PyG behavior.
        return super().__inc__(key, value, *args, **kwargs)


class ProteinGraphFeaturiserSingleChain(object):
    """Protein Graph Featurizer for aligned single-chain data with k conformations.

    Builds backbone protein graphs from k conformations (stacked, not flattened).
    All k conformations are of the same chain, pre-aligned to have the same length.

    Returned graph is of type `torch_geometric.data.Data` with attributes:
    - seq        sequence converted to int tensor, shape [num_nodes]
    - node_s     node scalar features, shape [num_nodes, k, ...]
    - node_v     node vector features, shape [num_nodes, k, ...]
    - edge_s     edge scalar features, shape [num_edges, k, ...]
    - edge_v     edge vector features, shape [num_edges, k, ...]
    - edge_index edge indices (union topology), shape [2, num_edges]
    - virtual_mask node mask, shape [num_nodes, k]
    - coords     coordinates, shape [num_nodes, k, num_atoms, 3]

    Args:
        k: number of conformations to stack
        split: train/validation/test split; coords are noised during training
        noise_scale: standard deviation of gaussian noise added to coordinates
    """

    def __init__(
        self,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
        k: int = 2,
        representation="ca",
        split="train",
        threshold=0.7,
        noise_scale=0.1,
        distance_eps=DISTANCE_EPS,
        device="cpu",
    ):
        super().__init__()

        self.scalar_node_features = scalar_node_features
        self.vector_node_features = vector_node_features
        self.edge_types = edge_types
        self.scalar_edge_features = scalar_edge_features
        self.vector_edge_features = vector_edge_features
        self.k = k
        self.representation = representation
        self.split = split
        self.noise_scale = noise_scale
        self.distance_eps = distance_eps
        self.threshold = threshold
        self.device = device

        if "sequence_positional_encoding" in self.scalar_node_features:
            self.positional_encoding = PositionalEncoding(16)

    def add_node_features(self, pyg_object):
        """
        Adds node features to k stacked conformations of a protein.
        Returns features with shape [num_nodes, k, feature_dim]
        """
        k = pyg_object.coords.shape[1]

        # Compute node features for each conformation
        node_s = torch.stack([
            compute_scalar_node_features(pyg_object, self.scalar_node_features, idx=i)
            for i in range(k)
        ], dim=1)

        node_v = torch.stack([
            compute_vector_node_features(pyg_object, self.vector_node_features, idx=i)
            for i in range(k)
        ], dim=1)

        # Add positional encoding if available
        if hasattr(self, "positional_encoding"):
            pos_encodings = torch.stack([
                self.positional_encoding(pyg_object.residue_index[:, i])
                for i in range(k)
            ], dim=1)
            node_s = torch.cat([node_s, pos_encodings], dim=-1)

        pyg_object.node_s = node_s  # Shape: [num_nodes, k, num_scalar_features]
        pyg_object.node_v = node_v  # Shape: [num_nodes, k, num_vector_features, 3]
        pyg_object.node_s = torch.nan_to_num(pyg_object.node_s, nan=0.0, posinf=0.0, neginf=0.0)
        pyg_object.node_v = torch.nan_to_num(pyg_object.node_v, nan=0.0, posinf=0.0, neginf=0.0)

        return pyg_object

    def add_edge_features(self, pyg_object):
        """
        Computes the UNION of edge indices across all conformations first.
        Then computes features for this fixed topology for every conformation.
        """
        k = pyg_object.coords.shape[1]

        # --- Step 1: Compute Edges for all k (Indices Only) ---
        all_edge_indices = []
        all_edge_types = []

        for i in range(k):
            e_idx, e_type = compute_edges(pyg_object, self.edge_types, idx=i)
            all_edge_indices.append(e_idx)
            all_edge_types.append(e_type)

        # --- Step 2: Create the UNION Topology ---
        cat_indices = torch.cat(all_edge_indices, dim=1)
        cat_types = torch.cat(all_edge_types, dim=0)

        # Coalesce to remove duplicates
        union_edge_index, union_edge_type = coalesce(cat_indices, cat_types, reduce='max')

        # --- Step 3: Filter Virtual Edges ---
        is_node_real_any = pyg_object.virtual_mask.any(dim=1)  # [N]
        src, dst = union_edge_index
        mask_real = is_node_real_any[src] & is_node_real_any[dst]

        union_edge_index = union_edge_index[:, mask_real]
        union_edge_type = union_edge_type[mask_real]

        # Temporarily store edge_index as a list for feature computation
        # (compute_scalar_edge_features expects x.edge_index[idx])
        pyg_object.edge_index = [union_edge_index] * k
        pyg_object.edge_type = [union_edge_type] * k

        # --- Step 4: Compute Features for the Union Topology ---
        edge_s_list = []
        edge_v_list = []

        if self.scalar_edge_features:
            edge_s_list = [
                compute_scalar_edge_features(pyg_object, self.scalar_edge_features, i)
                for i in range(k)
            ]

        if self.vector_edge_features:
            edge_v_list = [
                compute_vector_edge_features(pyg_object, self.vector_edge_features, i)
                for i in range(k)
            ]

        # --- Step 5: Stack Features ---
        if edge_s_list:
            pyg_object.edge_s = torch.stack(edge_s_list, dim=1)  # [E, k, S_dim]
        else:
            pyg_object.edge_s = None

        if edge_v_list:
            pyg_object.edge_v = torch.stack(edge_v_list, dim=1)  # [E, k, V_dim, 3]
        else:
            pyg_object.edge_v = None

        # Convert edge_index back to tensor (model expects tensor, not list)
        # For single-chain, all k conformations share the same union topology
        pyg_object.edge_index = union_edge_index
        pyg_object.edge_type = union_edge_type

        pyg_object.num_conformations = k

        # --- Decoder edges for autoregressive sampling ---
        # For single-chain mode, decoder edges = encoder edges (same union topology)
        # but we need to replicate features for each conformation
        num_edges = union_edge_index.shape[1]
        pyg_object.decoder_edge_index = union_edge_index
        pyg_object.decoder_edge_idx_length = torch.tensor(num_edges, dtype=torch.long)

        # edge_s is [E, k, S_dim] -> permute to [k, E, S_dim] -> flatten to [k*E, S_dim]
        if pyg_object.edge_s is not None:
            pyg_object.decoder_edge_s = pyg_object.edge_s.permute(1, 0, 2).reshape(-1, pyg_object.edge_s.shape[-1])
        else:
            pyg_object.decoder_edge_s = None

        # edge_v is [E, k, V_dim, 3] -> permute to [k, E, V_dim, 3] -> flatten to [k*E, V_dim, 3]
        if pyg_object.edge_v is not None:
            pyg_object.decoder_edge_v = pyg_object.edge_v.permute(1, 0, 2, 3).reshape(-1, *pyg_object.edge_v.shape[2:])
        else:
            pyg_object.decoder_edge_v = None

        pyg_object.decoder_edge_feat_length = torch.tensor(k * num_edges, dtype=torch.long)
        # All edges valid since we use union topology
        pyg_object.decoder_edge_mask = torch.ones(k * num_edges, dtype=torch.bool)

        return pyg_object

    @staticmethod
    def virtualize_masked_nodes(protein, masks):
        """
        Move masked nodes to +infinity and mark valid nodes in virtual_mask.
        """
        nodes_to_virtualize = ~masks
        k = protein.coords.shape[1]

        for i in range(k):
            protein.coords[:, i][nodes_to_virtualize[:, i]] = FILL_VALUE

        protein.virtual_mask = masks
        return protein

    def stack_conformations(self, confs_list):
        """
        Stack k conformations (all are pre-aligned, same length).
        """
        assert len(confs_list) == self.k, f"Expected {self.k} conformations, got {len(confs_list)}"

        stacked_coords = torch.stack([conf.coords for conf in confs_list], dim=1)
        stacked_residue_type = torch.stack([conf.residue_type for conf in confs_list], dim=1)
        stacked_residue_index = torch.stack([conf.residue_index for conf in confs_list], dim=1)

        num_residues = stacked_coords.shape[0]
        k = self.k

        # Single-chain design should never expose the ground-truth residue identity
        # through the encoder node features, so keep the sequence feature fully masked.
        stacked_mask_seq = torch.zeros_like(stacked_residue_type, dtype=torch.bool)

        # Create chains: all zeros for single-chain (same chain for all residues)
        # This is needed by compute_edges
        stacked_chains = torch.zeros(num_residues, k, dtype=torch.long)

        # Create homo_idx: all zeros for single-chain (same homomer group)
        # This is needed by compute_edges
        stacked_homo_idx = torch.zeros(num_residues, k, dtype=torch.long)

        # Create pdb_chain_pair_mask: all True for single-chain (all residues belong to the chain)
        stacked_pdb_chain_pair_mask = torch.ones(num_residues, k, dtype=torch.bool)

        # Create padding_mask: all True (no padding, all conformations are pre-aligned)
        padding_mask = torch.ones(num_residues, k, dtype=torch.bool)

        return Data(
            coords=stacked_coords,
            residue_type=stacked_residue_type,
            residue_index=stacked_residue_index,
            mask_seq=stacked_mask_seq,
            chains=stacked_chains,
            homo_idx=stacked_homo_idx,
            pdb_chain_pair_mask=stacked_pdb_chain_pair_mask,
            padding_mask=padding_mask,
        )

    def __call__(self, protein, pdb_code: str = None):
        """
        Featurize a protein PyG Data object for single-chain mode.
        """
        try:
            if len(protein.cluster_members) == 0:
                logger.warning(f"Skipping {pdb_code}: No cluster members found.")
                return None
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Skipping {pdb_code}: Malformed input. Error: {e}")
            return None

        # Select k conformations
        confs_list = self.get_entries(protein)

        if confs_list is None:
            logger.warning(f"Skipping {pdb_code}: Failed to select conformations.")
            return None

        # Stack conformations
        stacked_data = self.stack_conformations(confs_list)

        # Eliminate gaps
        pyg_object = self.eliminate_gaps(stacked_data)

        # Create mask for valid residues
        mask = (~torch.isin(pyg_object.residue_type, torch.tensor([20, 21])))

        for i in range(self.k):
            if mask[:, i].sum() < 10:
                logger.warning(f"Skipping {pdb_code}: Insufficient valid residues in conf {i}.")
                return None

        # Add noise during training
        if self.split == "train":
            noise = torch.randn_like(pyg_object.coords) * self.noise_scale
            pyg_object.coords = pyg_object.coords + noise

        # Get consensus sequence
        pyg_object.seq = self.get_sequence(pyg_object)
        if pyg_object.seq.shape[0] == 0:
            logger.warning(f"Skipping {pdb_code}: Empty consensus sequence.")
            return None

        # Virtualize masked nodes
        pyg_object = self.virtualize_masked_nodes(pyg_object, mask)

        # Add features
        pyg_object = self.add_node_features(pyg_object)
        pyg_object = self.add_edge_features(pyg_object)

        # Transform coordinates
        pyg_object = self.transform_coords(pyg_object, self.representation)

        pyg_object.num_edges = torch.tensor(pyg_object.edge_index.shape[1])
        pyg_object.num_nodes = torch.tensor(len(pyg_object.seq))
        pyg_object.num_decoder_nodes = len(pyg_object.seq)

        # Cleanup
        del pyg_object.residue_index
        del pyg_object.residue_type
        del pyg_object.edge_type

        if pyg_object.edge_index.numel() == 0:
            logger.warning(f"Skipping {pdb_code}: No edges created.")
            return None

        return pyg_object

    def eliminate_gaps(self, pyg):
        """Remove positions where ALL k conformations have gaps."""
        gap_mask = ((pyg.residue_type == 20) | (pyg.residue_type == 21)).all(dim=1)
        keep_mask = ~gap_mask

        if keep_mask.sum() == 0:
            logger.warning("eliminate_gaps: All positions are gaps!")
            return pyg

        pyg.coords = pyg.coords[keep_mask]
        pyg.residue_type = pyg.residue_type[keep_mask]
        pyg.residue_index = pyg.residue_index[keep_mask]
        if hasattr(pyg, 'mask_seq'):
            pyg.mask_seq = pyg.mask_seq[keep_mask]
        if hasattr(pyg, 'chains'):
            pyg.chains = pyg.chains[keep_mask]
        if hasattr(pyg, 'homo_idx'):
            pyg.homo_idx = pyg.homo_idx[keep_mask]
        if hasattr(pyg, 'pdb_chain_pair_mask'):
            pyg.pdb_chain_pair_mask = pyg.pdb_chain_pair_mask[keep_mask]
        if hasattr(pyg, 'padding_mask'):
            pyg.padding_mask = pyg.padding_mask[keep_mask]

        return pyg

    def get_sequence(self, pyg):
        """Get consensus sequence by randomly selecting from valid residues."""
        seq_length = pyg.residue_type.shape[0]
        k = pyg.residue_type.shape[1]

        value_mask = torch.isin(pyg.residue_type, torch.tensor([20, 21]))
        sequence = torch.zeros(seq_length, dtype=torch.long, device=pyg.residue_type.device)

        for pos in range(seq_length):
            valid_residues = pyg.residue_type[pos][~value_mask[pos]]
            if len(valid_residues) > 0:
                sequence[pos] = valid_residues[torch.randint(0, len(valid_residues), (1,))]
            else:
                sequence[pos] = 20
                logger.warning(f"No valid residues at position {pos}")

        return sequence

    def transform_coords(self, pyg_object, representation="ca"):
        """Transform coordinates to desired representation."""
        if representation == "ca_bb":
            idx = 3
        elif representation == "ca":
            idx = 1
        else:
            raise ValueError(f"Invalid representation: {representation}")

        if idx == 1:
            pyg_object.coords = pyg_object.coords[:, :, idx].unsqueeze(2)
        elif idx == 3:
            pyg_object.coords = pyg_object.coords[:, :, :idx]

        return pyg_object

    def compute_sequential_probabilities(self, tm_scores, candidates, selected_indices):
        """Compute probabilities for selecting next structure based on TM dissimilarity."""
        if len(selected_indices) == 0:
            return torch.ones(len(candidates)) / len(candidates)

        avg_similarities = []
        for candidate in candidates:
            tm_to_selected = []
            for sel in selected_indices:
                tm_val = tm_scores[candidate, sel]
                if torch.isnan(tm_val):
                    continue
                elif tm_val == 0.0:
                    tm_to_selected.append(torch.tensor(0.5))
                else:
                    tm_to_selected.append(tm_val)

            if len(tm_to_selected) > 0:
                avg_tm = torch.mean(torch.stack(tm_to_selected))
            else:
                avg_tm = torch.tensor(0.5)

            avg_similarities.append(avg_tm)

        avg_similarities = torch.tensor(avg_similarities)
        dissimilarities = 1.0 - avg_similarities

        temperature = 2.0
        scaled = dissimilarities / temperature
        probabilities = torch.softmax(scaled, dim=0)

        min_prob = 0.01
        probabilities = torch.maximum(probabilities, torch.tensor(min_prob))
        probabilities = probabilities / probabilities.sum()

        return probabilities

    def get_entries(self, protein):
        """
        Select k conformations from cluster_members.
        - If n == k: take all
        - If n < k: take all and randomly duplicate
        - If n > k: use TM-score based sampling if available, else random
        """
        n = len(protein.cluster_members)

        if n < 2:
            logger.warning(f"Less than 2 conformations available (n={n}).")
            return None

        if n == self.k:
            chain_ids = protein.cluster_members
            confs_list = [protein.pyg_dict[chain_id] for chain_id in chain_ids]

        elif n < self.k:
            all_confs = [protein.pyg_dict[chain_id] for chain_id in protein.cluster_members]
            additional_indices = random.choices(range(n), k=self.k - n)
            additional_confs = [all_confs[i] for i in additional_indices]
            confs_list = all_confs + additional_confs
            random.shuffle(confs_list)

        else:  # n > self.k
            tm_sampling_success = False

            # Use tm_score_representatives if available, otherwise fall back to cluster_members
            reps = getattr(protein, "tm_score_representatives", None) or protein.cluster_members

            if (hasattr(protein, "tm_scores") and
                protein.tm_scores is not None and
                protein.tm_scores.shape[0] > 1 and
                len(reps) >= self.k):

                try:
                    # reps already set above (tm_score_representatives or cluster_members)
                    tm_scores = protein.tm_scores
                    n_reps = len(reps)

                    selected_rep_indices = []
                    available_indices = list(range(n_reps))

                    for _ in range(self.k):
                        probs = self.compute_sequential_probabilities(
                            tm_scores, available_indices, selected_rep_indices
                        )
                        chosen_idx_pos = np.random.choice(len(available_indices), p=probs.cpu().numpy())
                        chosen_idx = available_indices[chosen_idx_pos]
                        selected_rep_indices.append(chosen_idx)
                        available_indices.remove(chosen_idx)

                    chain_ids = [reps[i] for i in selected_rep_indices]

                    if all(c in protein.pyg_dict for c in chain_ids):
                        confs_list = [protein.pyg_dict[c] for c in chain_ids]
                        tm_sampling_success = True

                except Exception as e:
                    logger.debug(f"TM-score sampling failed: {e}")

            if not tm_sampling_success:
                selected_indices = random.sample(range(n), self.k)
                chain_ids = [protein.cluster_members[i] for i in selected_indices]
                confs_list = [protein.pyg_dict[chain_id] for chain_id in chain_ids]

        assert len(confs_list) == self.k, f"Expected {self.k} conformations, got {len(confs_list)}"
        return confs_list
