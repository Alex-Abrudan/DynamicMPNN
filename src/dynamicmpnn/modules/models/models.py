from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dynamicmpnn.modules.layers.layers import GVP, GVPConvLayer, MultiGVPConvLayer, StackedMultiGVPConvLayer, LayerNorm, tuple_index
from dynamicmpnn.types import HOMOMER_NEGATIVE


class DynamicMPNN(torch.nn.Module):
    """
    Autoregressive GVP-GNN for multi-conformation protein inverse folding.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns logits over 20 amino acids
    at each position in a `torch.Tensor` of shape [n_nodes, 20].

    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        edge_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (20 amino acids)
    """

    def __init__(
        self,
        node_in_dim=(64, 4),
        node_h_dim=(128, 16),
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_layers=3,  # Backwards compat: used if num_encoder/decoder_layers not set
        num_encoder_layers=None,
        num_decoder_layers=None,
        drop_rate=0.1,
        out_dim=4,
        temperature = 0.1,
        n_samples=1,
        n_edge_gvps=0,
        pooling_strategy='all_chains_equal',
        n_pooled_encoder_layers=0,
        refresh_interval=0,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        # Support separate encoder/decoder layer counts, fall back to num_layers
        self.num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_layers
        self.num_layers = num_layers  # Keep for backwards compat
        self.out_dim = out_dim
        activations = (F.silu, None)
        self.temperature = temperature
        self.n_samples = n_samples
        self.pooling_strategy = pooling_strategy
        self.n_pooled_encoder_layers = n_pooled_encoder_layers
        self.refresh_interval = refresh_interval

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True),
        )

        # Encoder layers: keep the legacy multi-chain path separate from the
        # stacked single-chain encoder so tensor layout assumptions do not leak.
        encoder_layer_cls = StackedMultiGVPConvLayer if self.pooling_strategy == 'single_chain_k' else MultiGVPConvLayer
        self.encoder_layers = nn.ModuleList(
            encoder_layer_cls(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                norm_first=True,
                n_edge_gvps=n_edge_gvps,
            )
            for _ in range(self.num_encoder_layers)
        )

        # Post-pooling encoder layers for sample_v3: global MP on pooled graph (no sequence),
        # applied once after pooling before the cheap local AR decode.
        edge_h_dim_pre_seq = self.edge_h_dim
        self.pooled_encoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim,
                edge_h_dim_pre_seq,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                norm_first=True,
                n_edge_gvps=n_edge_gvps,
            )
            for _ in range(n_pooled_encoder_layers)
        )

        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                autoregressive=True,
                norm_first=True,
                n_edge_gvps=n_edge_gvps,
            )
            for _ in range(self.num_decoder_layers)
        )

        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))

    def forward(self, batch):
        # Dispatch to single-chain forward if using single_chain_k pooling
        if self.pooling_strategy == 'single_chain_k':
            return self.forward_single_chain(batch)

        device = batch.edge_index.device
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        h_E_decoder = (batch.decoder_edge_s, batch.decoder_edge_v)
        edge_index = batch.edge_index
        edge_index_decoder = batch.decoder_edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        h_E_decoder = self.W_e(h_E_decoder)

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, edge_index, h_E)

        all_objects_h_V = list(zip(
            split_batch(h_V[0], batch.batch),
            split_batch(h_V[1], batch.batch)
        ))

        decoder_edge_feat_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.decoder_edge_feat_length, dim=0)
        ])

        pooled_h_V_s, pooled_h_V_v = [], []
        pooled_h_E_s, pooled_h_E_v = [], []

        for batch_idx, h_V_batch in enumerate(all_objects_h_V):
            sample_decoder_edge_s = batch.decoder_edge_s[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]
            sample_decoder_edge_v = batch.decoder_edge_v[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]
            sample_edge_mask = batch.decoder_edge_mask[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]

            h_V_pooled, h_E_pooled, _ = self.pool_for_training(
                h_V_=h_V_batch,
                h_E_=(self.W_e((sample_decoder_edge_s, sample_decoder_edge_v))),
                homomer_index=batch.homo_idx[batch.batch == batch_idx],
                virtual_mask=batch.virtual_mask[batch.batch == batch_idx],
                decoder_edge_mask=sample_edge_mask,
                n_edges=batch.decoder_edge_idx_length[batch_idx].item(),
            )

            pooled_h_V_s.append(h_V_pooled[0])
            pooled_h_V_v.append(h_V_pooled[1])
            pooled_h_E_s.append(h_E_pooled[0])
            pooled_h_E_v.append(h_E_pooled[1])

        h_V_decoder = (
            torch.cat(pooled_h_V_s, dim=0),
            torch.cat(pooled_h_V_v, dim=0),
        )
        h_E_decoder = (
            torch.cat(pooled_h_E_s, dim=0),
            torch.cat(pooled_h_E_v, dim=0),
        )
        final_valid_mask = torch.ones(seq.shape[0], dtype=torch.bool, device=device)

        # Mask the LABELS for the loss function
        seq[~final_valid_mask] = 20

        # Post-pooling global MP on the pooled graph (no sequence info).
        # Trains the pooled_encoder_layers so sample_v3 benefits from learned enrichment.
        for layer in self.pooled_encoder_layers:
            h_V_decoder, h_E_decoder = layer(h_V_decoder, edge_index_decoder, h_E_decoder)

        # encoder_embeddings must be set AFTER pooled encoder layers so the decoder's
        # autoregressive_x (used for unsampled/backward edges) reflects the enriched
        # representations — matching what sample_v3 uses as h_V_cache[0].
        encoder_embeddings = h_V_decoder

        h_S = self.W_s(seq)
        h_S = h_S[edge_index_decoder[0]]
        h_S[edge_index_decoder[0] >= edge_index_decoder[1]] = 0
        h_E_decoder = (torch.cat([h_E_decoder[0], h_S], dim=-1), h_E_decoder[1])

        for layer in self.decoder_layers:
            h_V_decoder, h_E_decoder = layer(h_V_decoder, edge_index_decoder, h_E_decoder, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V_decoder)

        return logits, final_valid_mask

    def forward_single_chain(self, batch):
        """
        Forward pass for single-chain mode with k aligned conformations.

        In this mode:
        - node_s, node_v have shape [N, k, D] where N is the same across all k conformations
        - edge_s, edge_v have shape [E, k, D] with a union edge topology
        - We embed per-conformation, encode with MultiGVPConvLayer, then pool across k
        - Decoder uses the same edge_index as encoder (no separate decoder edges)
        """
        device = batch.edge_index.device

        # Input: [N, k, D_s], [N, k, D_v, 3]
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        virtual_mask = batch.virtual_mask  # [N, k]
        seq = batch.seq  # [N]

        k = h_V[0].shape[1]

        # --- 1. Embed Input Features (Per Conformation) ---
        h_V_s_list, h_V_v_list = [], []
        h_E_s_list, h_E_v_list = [], []

        for i in range(k):
            h_V_i = (h_V[0][:, i], h_V[1][:, i])
            h_V_i_emb = self.W_v(h_V_i)
            h_V_s_list.append(h_V_i_emb[0])
            h_V_v_list.append(h_V_i_emb[1])

            h_E_i = (h_E[0][:, i], h_E[1][:, i])
            h_E_i_emb = self.W_e(h_E_i)
            h_E_s_list.append(h_E_i_emb[0])
            h_E_v_list.append(h_E_i_emb[1])

        h_V = (torch.stack(h_V_s_list, dim=1), torch.stack(h_V_v_list, dim=1))
        h_E = (torch.stack(h_E_s_list, dim=1), torch.stack(h_E_v_list, dim=1))

        # --- 2. Encode Multi-Conformation Graph ---
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, edge_index, h_E)

        # --- 3. Pool Nodes across k conformations (Masked Mean) ---
        h_V_pooled = pool_conformations(h_V, virtual_mask)

        # --- 4. Pool Edges across k conformations (Masked Mean) ---
        src, dst = edge_index
        edge_mask = virtual_mask[src] & virtual_mask[dst]  # [E, k]
        h_E_pooled = pool_edges_conformations(h_E, edge_mask)

        # Decoder uses same edge topology
        edge_index_decoder = edge_index
        h_E_decoder = h_E_pooled

        # All nodes are valid in single-chain mode (no missing chain positions)
        final_valid_mask = torch.ones(seq.shape[0], dtype=torch.bool, device=device)

        # --- 5. Post-pooling encoder layers (if any) ---
        h_V_decoder = h_V_pooled
        for layer in self.pooled_encoder_layers:
            h_V_decoder, h_E_decoder = layer(h_V_decoder, edge_index_decoder, h_E_decoder)

        encoder_embeddings = h_V_decoder

        # --- 6. Prepare Sequence Features ---
        # Single-chain training must not see the target sequence at any position.
        # Unlike the multi-chain setting, there is no binder/context sequence to keep unmasked,
        # so the decoder always receives a fully masked sequence embedding here.
        h_S = torch.zeros(seq.shape[0], self.out_dim, device=device, dtype=h_E_decoder[0].dtype)
        h_S = h_S[edge_index_decoder[0]]
        h_S[edge_index_decoder[0] >= edge_index_decoder[1]] = 0
        h_E_decoder = (torch.cat([h_E_decoder[0], h_S], dim=-1), h_E_decoder[1])

        # --- 7. Decode ---
        for layer in self.decoder_layers:
            h_V_decoder, h_E_decoder = layer(
                h_V_decoder,
                edge_index_decoder,
                h_E_decoder,
                autoregressive_x=encoder_embeddings
            )

        logits = self.W_out(h_V_decoder)

        return logits, final_valid_mask

    @torch.no_grad()
    def sample(
        self,
        batch,
        logit_bias: Optional[torch.Tensor] = None,
        return_logits: Optional[bool] = False,
    ):
        # Dispatch to single-chain sample if using single_chain_k pooling
        if self.pooling_strategy == 'single_chain_k':
            return self.sample_single_chain(batch, return_logits=return_logits)

        # Two cases keep the per-graph reference path:
        #
        #  * logit_bias is indexed per residue position and has no obvious
        #    meaning across graphs of differing length.
        #  * n_edge_gvps > 0 makes the decoder layers update edge features at
        #    every layer. The fast decoders call layer.conv.message_func
        #    directly rather than going through GVPConvLayer.forward, so they
        #    never run edge_message_func and would silently use stale edge
        #    features. See _decoder_updates_edge_features.
        if logit_bias is not None or self._decoder_updates_edge_features():
            return self.sample_v3(batch, logit_bias=logit_bias,
                                  return_logits=return_logits,
                                  refresh_interval=self.refresh_interval)

        # For a SINGLE graph -- which is what the evaluation pipeline passes --
        # sample_batched reproduces sample_v3's sampled sequences byte-for-byte,
        # verified on all 96 benchmark targets, and is ~1.9x faster.
        #
        # For a batch of SEVERAL graphs it is numerically equivalent but NOT
        # byte-identical: draws interleave across graphs, so the RNG is consumed
        # in a different order than decoding them one at a time. The
        # distribution is unchanged; the particular seeded sequences are not.
        # Call sample_v3 directly if byte-identical multi-graph reproduction of
        # an earlier run matters.
        #
        # Re-assemble its per-graph lists into sample_v3's return contract.
        result = self.sample_batched(batch, return_logits=return_logits,
                                     refresh_interval=self.refresh_interval)
        if not return_logits:
            return torch.cat(result, dim=1)

        sequences, per_graph_logits, pooled_indices = result
        batch_indices = torch.cat([
            torch.full((len(idx),), i, device=idx.device, dtype=torch.long)
            for i, idx in enumerate(pooled_indices)
        ], dim=0)
        return (torch.cat(sequences, dim=1),
                torch.cat(per_graph_logits, dim=1),
                torch.cat(pooled_indices, dim=0),
                batch_indices)

    @torch.no_grad()
    def sample_v3(
        self,
        batch,
        logit_bias: Optional[torch.Tensor] = None,
        return_logits: Optional[bool] = False,
        refresh_interval: int = 0,
        forced_seq: Optional[torch.Tensor] = None,
    ):
        """
        Hybrid sampling: run global MP on the pooled graph once (via self.pooled_encoder_layers),
        then use cheap local AR decoding (only neighbours of the decoded node are updated at each step).
        This avoids the O(n^2) cost of full global MP at every AR step while
        enriching the node representations beyond the plain mean-pooled encoder output.

        Args:
            refresh_interval: If > 0, refresh cached node features for all decoded nodes every
                              refresh_interval steps. This reduces staleness at the cost of O(n^2/interval).
                              Set to 0 to disable (default).
        """
        device = batch.edge_index.device

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        original_indices = torch.arange(batch.seq.shape[0], device=device)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, edge_index, h_E)

        decoder_edge_idx_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.decoder_edge_idx_length, dim=0)
        ])
        decoder_edge_feat_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.decoder_edge_feat_length, dim=0)
        ])
        decoder_node_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.num_decoder_nodes, dim=0)
        ])

        all_objects_h_V = list(zip(
            split_batch(h_V[0], batch.batch),
            split_batch(h_V[1], batch.batch)
        ))

        all_sampled_seqs = []
        all_logits = []
        all_pooled_indices = []

        for batch_idx, h_V_batch in enumerate(all_objects_h_V):

            sample_decoder_edge_index_GLOBAL = batch.decoder_edge_index[:, decoder_edge_idx_ptr[batch_idx]:decoder_edge_idx_ptr[batch_idx + 1]]
            offset_for_this_graph = decoder_node_ptr[batch_idx]
            edge_index_decoder = sample_decoder_edge_index_GLOBAL - offset_for_this_graph

            sample_decoder_edge_s = batch.decoder_edge_s[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]
            sample_decoder_edge_v = batch.decoder_edge_v[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]
            sample_edge_mask = batch.decoder_edge_mask[decoder_edge_feat_ptr[batch_idx]:decoder_edge_feat_ptr[batch_idx + 1]]
            original_indices_graph = original_indices[decoder_node_ptr[batch_idx]:decoder_node_ptr[batch_idx + 1]]

            h_V_pooled, h_E_pooled, edge_index_decoder, original_indices_pooled = self.pool_for_sampling(
                h_V_=h_V_batch,
                h_E_=(self.W_e((sample_decoder_edge_s, sample_decoder_edge_v))),
                edge_index_local=edge_index_decoder,
                homomer_index=batch.homo_idx[batch.batch == batch_idx],
                virtual_mask=batch.virtual_mask[batch.batch == batch_idx],
                decoder_edge_mask=sample_edge_mask,
                n_edges=batch.decoder_edge_idx_length[batch_idx].item(),
                original_indices=original_indices_graph,
            )

            all_pooled_indices.append(original_indices_pooled)

            num_nodes = h_V_pooled[0].shape[0]

            if edge_index_decoder.numel() > 0:
                max_idx = edge_index_decoder.max().item()
                assert max_idx < num_nodes, (
                    f"CRITICAL DATA MISMATCH in batch_idx={batch_idx}! "
                    f"Pooling reduced nodes to {num_nodes}, but edge_index "
                    f"still has a max index of {max_idx}."
                )

            # Post-pooling global MP on the pooled graph (once, no sequence info).
            # This enriches node/edge representations before the cheap local AR decode.
            for layer in self.pooled_encoder_layers:
                h_V_pooled, h_E_pooled = layer(h_V_pooled, edge_index_decoder, h_E_pooled)

            # Expand node/edge features for n_samples parallel decoding
            h_V_decoder = (h_V_pooled[0].repeat(self.n_samples, 1),
                           h_V_pooled[1].repeat(self.n_samples, 1, 1))
            h_E_decoder = (h_E_pooled[0].repeat(self.n_samples, 1),
                           h_E_pooled[1].repeat(self.n_samples, 1, 1))

            edge_index_decoder = edge_index_decoder.expand(self.n_samples, -1, -1)
            offset = num_nodes * torch.arange(self.n_samples, device=device).view(-1, 1, 1)
            edge_index_decoder = torch.cat(tuple(edge_index_decoder + offset), dim=-1)

            seq = torch.zeros(self.n_samples * num_nodes, device=device, dtype=torch.int)
            model_dtype = h_V_decoder[0].dtype

            h_S = torch.zeros(self.n_samples * num_nodes, self.out_dim, device=device, dtype=model_dtype)
            logits = torch.zeros(self.n_samples * num_nodes, self.out_dim, device=device, dtype=model_dtype)

            # Static encoder embeddings for autoregressive_x - never modified.
            # BUG FIX: Previously h_V_cache[0] was used for both layer 0 input AND autoregressive_x,
            # but layer 0 modifies its input in-place, polluting autoregressive_x with layer-0 outputs.
            encoder_embeddings = (h_V_decoder[0].clone(), h_V_decoder[1].clone())

            # h_V_cache[j] holds layer (j-1)'s output for decoded nodes (= input to layer j).
            # For j=0, we use encoder_embeddings directly (with save/restore to prevent pollution).
            h_V_cache = [(h_V_decoder[0].clone(), h_V_decoder[1].clone()) for _ in self.decoder_layers]

            for i in range(num_nodes):
                # Build edge features: inject already-sampled sequence features
                h_S_ = h_S[edge_index_decoder[0]]
                h_S_[edge_index_decoder[0] >= edge_index_decoder[1]] = 0
                h_E_ = (torch.cat([h_E_decoder[0], h_S_], dim=-1), h_E_decoder[1])

                # Restrict to edges whose destination is node i (across all samples)
                edge_mask = edge_index_decoder[1] % num_nodes == i
                edge_index_ = edge_index_decoder[:, edge_mask]
                h_E_ = tuple_index(h_E_, edge_mask)

                node_mask = torch.zeros(self.n_samples * num_nodes, device=device, dtype=torch.bool)
                node_mask[i::num_nodes] = True

                # Local message passing through decoder layers for node i only
                for j, layer in enumerate(self.decoder_layers):
                    # Select input: encoder_embeddings for layer 0, h_V_cache[j] for layer j > 0
                    if j == 0:
                        layer_input = encoder_embeddings
                    else:
                        layer_input = h_V_cache[j]

                    # Save before layer modifies in-place (to restore correct layer-(j-1) output)
                    saved_s = layer_input[0][i::num_nodes].clone()
                    saved_v = layer_input[1][i::num_nodes].clone()

                    updated_nodes, updated_edges = layer(
                        layer_input, edge_index_, h_E_,
                        autoregressive_x=encoder_embeddings, node_mask=node_mask
                    )
                    out = tuple_index(updated_nodes, node_mask)
                    h_E_ = updated_edges

                    # Restore layer_input (undo in-place modification to keep correct semantics)
                    layer_input[0][i::num_nodes] = saved_s
                    layer_input[1][i::num_nodes] = saved_v

                    # Store layer j's output as input to layer j+1
                    if j < len(self.decoder_layers) - 1:
                        h_V_cache[j + 1][0][i::num_nodes] = out[0]
                        h_V_cache[j + 1][1][i::num_nodes] = out[1]

                lgts = self.W_out(out)
                if logit_bias is not None:
                    lgts += logit_bias[i]

                if lgts.dtype in [torch.float16, torch.bfloat16]:
                    lgts = torch.nan_to_num(lgts, nan=-1e4, posinf=1e4, neginf=-1e4)
                lgts = torch.clamp(lgts, min=-1e4, max=1e4)

                # Test hook: teacher-force the decode so this path can be compared
                # against sample_batched without stochastic draws. None => unchanged.
                if forced_seq is not None:
                    seq[i::num_nodes] = forced_seq[i].to(torch.int)
                else:
                    seq[i::num_nodes] = Categorical(logits=lgts / self.temperature).sample()
                h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
                logits[i::num_nodes] = lgts

                # Periodic refresh: re-run decoder layers on all decoded nodes (0..i) to reduce staleness
                if refresh_interval > 0 and i > 0 and (i + 1) % refresh_interval == 0:
                    # Build edges among decoded nodes (src and dst both in 0..i)
                    src_pos = edge_index_decoder[0] % num_nodes
                    dst_pos = edge_index_decoder[1] % num_nodes
                    refresh_edge_mask = (src_pos <= i) & (dst_pos <= i)

                    if refresh_edge_mask.any():
                        refresh_edge_index = edge_index_decoder[:, refresh_edge_mask]

                        # Edge features with current sequence
                        h_S_refresh = h_S[edge_index_decoder[0]]
                        h_S_refresh[edge_index_decoder[0] >= edge_index_decoder[1]] = 0
                        h_E_refresh_full = (torch.cat([h_E_decoder[0], h_S_refresh], dim=-1), h_E_decoder[1])
                        h_E_refresh = tuple_index(h_E_refresh_full, refresh_edge_mask)

                        # Node mask for all decoded nodes 0..i
                        refresh_node_mask = (torch.arange(self.n_samples * num_nodes, device=device) % num_nodes) <= i

                        # Run through decoder layers to refresh h_V_cache
                        for j, layer in enumerate(self.decoder_layers):
                            if j == 0:
                                refresh_input = encoder_embeddings
                            else:
                                refresh_input = h_V_cache[j]

                            # Save before in-place modification
                            refresh_saved_s = refresh_input[0][refresh_node_mask].clone()
                            refresh_saved_v = refresh_input[1][refresh_node_mask].clone()

                            refresh_updated, h_E_refresh = layer(
                                refresh_input, refresh_edge_index, h_E_refresh,
                                autoregressive_x=encoder_embeddings, node_mask=refresh_node_mask
                            )
                            refresh_out = tuple_index(refresh_updated, refresh_node_mask)

                            # Restore input
                            refresh_input[0][refresh_node_mask] = refresh_saved_s
                            refresh_input[1][refresh_node_mask] = refresh_saved_v

                            # Update h_V_cache[j+1] for refreshed nodes
                            if j < len(self.decoder_layers) - 1:
                                h_V_cache[j + 1][0][refresh_node_mask] = refresh_out[0]
                                h_V_cache[j + 1][1][refresh_node_mask] = refresh_out[1]

            all_sampled_seqs.append(seq.view(self.n_samples, num_nodes))
            if return_logits:
                all_logits.append(logits.view(self.n_samples, num_nodes, self.out_dim))

        final_seqs = torch.cat(all_sampled_seqs, dim=1)
        final_indices = torch.cat(all_pooled_indices, dim=0)
        batch_indices = torch.cat([
            torch.full((len(idx),), i, device=device, dtype=torch.long)
            for i, idx in enumerate(all_pooled_indices)
        ], dim=0)

        if return_logits:
            final_logits = torch.cat(all_logits, dim=1)
            return final_seqs, final_logits, final_indices, batch_indices
        else:
            return final_seqs

    @torch.no_grad()
    def sample_single_chain(self, batch, return_logits=False):
        """
        Sampling for single-chain mode with k aligned conformations.

        Similar to the training forward pass but with autoregressive decoding.
        """
        device = batch.edge_index.device

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        virtual_mask = batch.virtual_mask

        k = h_V[0].shape[1]

        # --- 1. Embed per conformation ---
        h_V_s_list, h_V_v_list = [], []
        h_E_s_list, h_E_v_list = [], []

        for i in range(k):
            h_V_emb = self.W_v((h_V[0][:, i], h_V[1][:, i]))
            h_V_s_list.append(h_V_emb[0])
            h_V_v_list.append(h_V_emb[1])

            h_E_emb = self.W_e((h_E[0][:, i], h_E[1][:, i]))
            h_E_s_list.append(h_E_emb[0])
            h_E_v_list.append(h_E_emb[1])

        h_V = (torch.stack(h_V_s_list, dim=1), torch.stack(h_V_v_list, dim=1))
        h_E = (torch.stack(h_E_s_list, dim=1), torch.stack(h_E_v_list, dim=1))

        # --- 2. Encode ---
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, edge_index, h_E)

        # --- 3. Pool nodes across conformations ---
        h_V_pooled = pool_conformations(h_V, virtual_mask)

        # --- 4. Pool edges across conformations ---
        src, dst = edge_index
        edge_mask = virtual_mask[src] & virtual_mask[dst]
        h_E_pooled = pool_edges_conformations(h_E, edge_mask)

        # --- 5. Build per-graph pointers ---
        node_ptr = batch.ptr
        num_graphs = node_ptr.shape[0] - 1

        src_batch = batch.batch[edge_index[0]]
        num_edges_per_graph = torch.bincount(src_batch, minlength=num_graphs)
        edge_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(num_edges_per_graph, dim=0)
        ])

        # --- 6. Post-pooling encoder layers ---
        h_V_decoder = h_V_pooled
        h_E_decoder = h_E_pooled
        for layer in self.pooled_encoder_layers:
            h_V_decoder, h_E_decoder = layer(h_V_decoder, edge_index, h_E_decoder)

        # --- 7. Per-graph autoregressive sampling ---
        all_sampled_seqs = []
        all_logits = []
        all_indices = []  # Track node indices for each graph

        for graph_idx in range(num_graphs):
            node_start = node_ptr[graph_idx].item()
            node_end = node_ptr[graph_idx + 1].item()
            num_nodes = node_end - node_start

            if num_nodes == 0:
                continue

            h_V_graph = (
                h_V_decoder[0][node_start:node_end].clone(),
                h_V_decoder[1][node_start:node_end].clone()
            )
            encoder_embeddings = (h_V_graph[0].clone(), h_V_graph[1].clone())

            edge_start = edge_ptr[graph_idx].item()
            edge_end = edge_ptr[graph_idx + 1].item()

            graph_edge_index = edge_index[:, edge_start:edge_end] - node_start
            h_E_graph = (
                h_E_decoder[0][edge_start:edge_end].clone(),
                h_E_decoder[1][edge_start:edge_end].clone()
            )

            # Initialize sequence with zeros (will be filled autoregressively)
            sampled_seq = torch.zeros(self.n_samples, num_nodes, dtype=torch.long, device=device)
            graph_logits = []

            for i in range(num_nodes):
                # Prepare sequence embeddings for edges
                h_S = self.W_s(sampled_seq)  # [n_samples, num_nodes, out_dim]
                h_S_edges = h_S[:, graph_edge_index[0]]  # [n_samples, E, out_dim]
                # Mask future positions
                h_S_edges[:, graph_edge_index[0] >= graph_edge_index[1]] = 0

                # Expand edge features for n_samples
                h_E_expanded = (
                    h_E_graph[0].unsqueeze(0).expand(self.n_samples, -1, -1),
                    h_E_graph[1].unsqueeze(0).expand(self.n_samples, -1, -1, -1)
                )

                # Concat sequence info to edge features
                h_E_decode = (
                    torch.cat([h_E_expanded[0], h_S_edges], dim=-1),
                    h_E_expanded[1]
                )

                # Expand node features for n_samples
                h_V_decode = (
                    h_V_graph[0].unsqueeze(0).expand(self.n_samples, -1, -1).clone(),
                    h_V_graph[1].unsqueeze(0).expand(self.n_samples, -1, -1, -1).clone()
                )
                enc_emb = (
                    encoder_embeddings[0].unsqueeze(0).expand(self.n_samples, -1, -1),
                    encoder_embeddings[1].unsqueeze(0).expand(self.n_samples, -1, -1, -1)
                )

                # Run decoder layers
                for layer in self.decoder_layers:
                    # Reshape for layer: merge batch and n_samples
                    h_V_flat = (
                        h_V_decode[0].reshape(-1, h_V_decode[0].shape[-1]),
                        h_V_decode[1].reshape(-1, *h_V_decode[1].shape[-2:])
                    )
                    h_E_flat = (
                        h_E_decode[0].reshape(-1, h_E_decode[0].shape[-1]),
                        h_E_decode[1].reshape(-1, *h_E_decode[1].shape[-2:])
                    )
                    enc_flat = (
                        enc_emb[0].reshape(-1, enc_emb[0].shape[-1]),
                        enc_emb[1].reshape(-1, *enc_emb[1].shape[-2:])
                    )

                    # Offset edge indices for batched processing
                    edge_offset = torch.arange(self.n_samples, device=device) * num_nodes
                    batched_edge_index = graph_edge_index.unsqueeze(0) + edge_offset.view(-1, 1, 1)
                    batched_edge_index = batched_edge_index.reshape(2, -1)

                    h_V_flat, h_E_flat = layer(
                        h_V_flat, batched_edge_index, h_E_flat,
                        autoregressive_x=enc_flat
                    )

                    # Reshape back
                    h_V_decode = (
                        h_V_flat[0].reshape(self.n_samples, num_nodes, -1),
                        h_V_flat[1].reshape(self.n_samples, num_nodes, *h_V_decode[1].shape[-2:])
                    )
                    h_E_decode = (
                        h_E_flat[0].reshape(self.n_samples, -1, h_E_decode[0].shape[-1]),
                        h_E_flat[1].reshape(self.n_samples, -1, *h_E_decode[1].shape[-2:])
                    )

                # Get logits for position i
                logits_i = self.W_out((h_V_decode[0][:, i], h_V_decode[1][:, i]))
                logits_i = logits_i / self.temperature

                # Sample
                probs = F.softmax(logits_i, dim=-1)
                sampled_seq[:, i] = torch.multinomial(probs, 1).squeeze(-1)

                if return_logits:
                    graph_logits.append(logits_i)

            all_sampled_seqs.append(sampled_seq)
            if return_logits:
                all_logits.append(torch.stack(graph_logits, dim=1))
            # Track indices: original node indices for this graph
            all_indices.append(torch.arange(node_start, node_end, device=device))

        # Concatenate results
        final_seqs = torch.cat(all_sampled_seqs, dim=1)
        final_indices = torch.cat(all_indices, dim=0)
        batch_indices = batch.batch  # Already computed by PyG

        if return_logits:
            final_logits = torch.cat(all_logits, dim=1)
            return final_seqs, final_logits, final_indices, batch_indices
        else:
            return final_seqs

    def pool_for_sampling(
            self,
            h_V_,
            h_E_,
            edge_index_local,
            homomer_index,
            virtual_mask,
            decoder_edge_mask,
            n_edges,
            original_indices,
            ):

        n_chains = len(h_E_[0]) // n_edges
        chunk_size = n_edges
        h_E_chunks = [
            (h_E_[0][i*chunk_size:(i+1)*chunk_size], h_E_[1][i*chunk_size:(i+1)*chunk_size])
            for i in range(n_chains)
        ]
        edge_mask_chunks = [
            decoder_edge_mask[i*chunk_size:(i+1)*chunk_size]
            for i in range(n_chains)
        ]

        homo_ids = torch.unique(homomer_index)[torch.unique(homomer_index) != HOMOMER_NEGATIVE]
        chain_indices = [torch.where(homomer_index == hid)[0] for hid in homo_ids]
        selected_h_E_chunks = h_E_chunks
        selected_mask_chunks = edge_mask_chunks

        all_extractions = [
            (h_V_[0][chain_idx_list],
            h_V_[1][chain_idx_list],
            virtual_mask[chain_idx_list],
            virtual_mask[chain_idx_list])
            for chain_idx_list in chain_indices
        ]

        h_E_stacked = (
            torch.stack([chunk[0] for chunk in selected_h_E_chunks], dim=0),
            torch.stack([chunk[1] for chunk in selected_h_E_chunks], dim=0)
        )

        edge_mask_stacked = torch.stack(selected_mask_chunks, dim=0)

        edge_mask_expanded_s = edge_mask_stacked.unsqueeze(-1)
        edge_mask_expanded_v = edge_mask_stacked.unsqueeze(-1).unsqueeze(-1)  # [n_selected_chains, n_edges, 1, 1]
        masked_h_E_s = h_E_stacked[0] * edge_mask_expanded_s  # [n_confs, n_edges, features]
        masked_h_E_v = h_E_stacked[1] * edge_mask_expanded_v  # [n_confs, n_edges, vec_features, 3]
        
        valid_confs_per_edge_s = edge_mask_expanded_s.sum(dim=0, keepdim=True)  # [1, n_edges, 1]
        valid_confs_per_edge_v = edge_mask_expanded_v.sum(dim=0, keepdim=True)  # [1, n_edges, 1, 1]

        assert (valid_confs_per_edge_s.squeeze(0) > 0).all(), "Division by zero detected"
        h_E_pooled = (
            masked_h_E_s.sum(dim=0) / (valid_confs_per_edge_s.squeeze(0)),  # [n_edges, features]
            masked_h_E_v.sum(dim=0) / (valid_confs_per_edge_v.squeeze(0))  # [n_edges, vec_features, 3]
        )
        
        h_V_s_chains, h_V_v_chains, mask_s_chains, mask_v_chains = zip(*all_extractions)
    
        h_V_s_stacked = torch.stack(h_V_s_chains, dim=0)
        h_V_v_stacked = torch.stack(h_V_v_chains, dim=0)

        mask_s_stacked = torch.stack(mask_s_chains, dim=0).unsqueeze(-1)
        mask_v_stacked = torch.stack(mask_v_chains, dim=0).unsqueeze(-1).unsqueeze(-1)
        s_valid_count = mask_s_stacked.sum(dim=0)
        v_valid_count = mask_v_stacked.sum(dim=0)

        h_V_s_sum = (h_V_s_stacked * mask_s_stacked).sum(dim=0)
        h_V_v_sum = (h_V_v_stacked * mask_v_stacked).sum(dim=0)

        assert (s_valid_count > 0).all(), "Division by zero detected"
        h_V_pooled = (h_V_s_sum / s_valid_count, h_V_v_sum / v_valid_count)

        return h_V_pooled, h_E_pooled, edge_index_local, original_indices

    def pool_for_training(
            self,
            h_V_,
            h_E_,
            homomer_index,
            virtual_mask,
            decoder_edge_mask,
            n_edges,
            ):

        n_chains = len(h_E_[0]) // n_edges
        chunk_size = n_edges
        h_E_chunks = [
            (h_E_[0][i*chunk_size:(i+1)*chunk_size], h_E_[1][i*chunk_size:(i+1)*chunk_size])
            for i in range(n_chains)
        ]
        edge_mask_chunks = [
            decoder_edge_mask[i*chunk_size:(i+1)*chunk_size]
            for i in range(n_chains)
        ]

        homo_ids = torch.unique(homomer_index)[torch.unique(homomer_index) != HOMOMER_NEGATIVE]
        chain_indices = [torch.where(homomer_index == hid)[0] for hid in homo_ids]
        selected_h_E_chunks = h_E_chunks
        selected_mask_chunks = edge_mask_chunks

        all_extractions = [
            (h_V_[0][chain_idx_list],
            h_V_[1][chain_idx_list],
            virtual_mask[chain_idx_list],
            virtual_mask[chain_idx_list])
            for chain_idx_list in chain_indices
        ]
        
        h_V_s_chains, h_V_v_chains, mask_s_chains, mask_v_chains = zip(*all_extractions)
    
        h_V_s_stacked = torch.stack(h_V_s_chains, dim=0)
        h_V_v_stacked = torch.stack(h_V_v_chains, dim=0)

        mask_s_stacked = torch.stack(mask_s_chains, dim=0).unsqueeze(-1)
        mask_v_stacked = torch.stack(mask_v_chains, dim=0).unsqueeze(-1).unsqueeze(-1)
        s_valid_count = mask_s_stacked.sum(dim=0)
        v_valid_count = mask_v_stacked.sum(dim=0)

        assert (s_valid_count > 0).all(), "Division by zero detected"

        h_V_s_sum = (h_V_s_stacked * mask_s_stacked).sum(dim=0)
        h_V_v_sum = (h_V_v_stacked * mask_v_stacked).sum(dim=0)

        h_V_pooled = (h_V_s_sum / s_valid_count, h_V_v_sum / v_valid_count)

        h_E_stacked = (
            torch.stack([chunk[0] for chunk in selected_h_E_chunks], dim=0),  # [n_selected_chains, n_edges, features]
            torch.stack([chunk[1] for chunk in selected_h_E_chunks], dim=0)   # [n_selected_chains, n_edges, vec_features, 3]
        )
        
        edge_mask_stacked = torch.stack(selected_mask_chunks, dim=0)

        edge_mask_expanded_s = edge_mask_stacked.unsqueeze(-1)
        edge_mask_expanded_v = edge_mask_stacked.unsqueeze(-1).unsqueeze(-1)  # [n_selected_chains, n_edges, 1, 1]
        masked_h_E_s = h_E_stacked[0] * edge_mask_expanded_s  # [n_confs, n_edges, features]
        masked_h_E_v = h_E_stacked[1] * edge_mask_expanded_v  # [n_confs, n_edges, vec_features, 3]
        
        valid_confs_per_edge_s = edge_mask_expanded_s.sum(dim=0, keepdim=True)
        valid_confs_per_edge_v = edge_mask_expanded_v.sum(dim=0, keepdim=True)

        assert (valid_confs_per_edge_s.squeeze(0) > 0).all(), "Division by zero detected"

        h_E_pooled = (
            masked_h_E_s.sum(dim=0) / valid_confs_per_edge_s.squeeze(0),
            masked_h_E_v.sum(dim=0) / valid_confs_per_edge_v.squeeze(0)
        )

        return h_V_pooled, h_E_pooled, None

    # ------------------------------------------------------------------
    # Local (CSR) autoregressive decode.
    #
    # Numerically equivalent to sample_v3, but each AR step touches only the
    # in-edges of the residue being decoded instead of the whole expanded edge
    # set. sample_v3 rebuilds sequence-conditioned features for all S*N*d edges
    # at every one of the N steps and then discards all but ~d of them, which
    # makes the decode O(N) more expensive than it needs to be.
    #
    # Equivalence relies on four properties of the shipped configuration;
    # see tests/ for the checks that pin them down:
    #   1. cache[j][i] is layer (j-1)'s output at node i, not enc[i].
    #   2. backward edges (src >= dst) take BOTH endpoints from enc.
    #   3. norm[0] is skipped on the autoregressive path despite norm_first.
    #   4. messages are summed then divided by the destination's in-degree.
    # ------------------------------------------------------------------

    def _decoder_updates_edge_features(self) -> bool:
        """Do the decoder layers rewrite edge features as they go?

        `GVPConvLayer.forward` updates `edge_attr` through `edge_message_func`
        before convolving, but only when the layer was built with
        `n_edge_gvps > 0`. The fast decoders below deliberately skip
        `GVPConvLayer.forward` and call `layer.conv.message_func` directly, so
        they build edge features once per step and reuse them across layers.

        That is exact when `n_edge_gvps == 0` -- which is how every shipped
        checkpoint is configured -- and silently wrong otherwise, since the
        per-layer edge updates would simply never happen. Callers use this to
        fall back to `sample_v3`, which goes through the full layer.
        """
        return any(getattr(layer, "edge_message_func", None) is not None
                   for layer in self.decoder_layers)

    @staticmethod
    def _build_decoder_csr(edge_index, num_nodes):
        """Destination-sorted, padded neighbour table for the static decoder graph.

        Returns (order, nbr_src, nbr_eid, valid, degree) where order sorts the
        edge arrays by destination, and the [N, K] tables give each node's
        in-edges padded to K = max in-degree.
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]
        order = torch.argsort(dst, stable=True)
        src_sorted = src[order]

        degree = torch.bincount(dst, minlength=num_nodes)
        max_degree = int(degree.max().item()) if degree.numel() else 0
        ptr = torch.cat([degree.new_zeros(1), degree.cumsum(0)])

        arange_k = torch.arange(max_degree, device=device)
        valid = arange_k.unsqueeze(0) < degree.unsqueeze(1)           # [N, K]
        nbr_eid = ptr[:-1].unsqueeze(1) + arange_k.unsqueeze(0)       # [N, K]
        nbr_eid = nbr_eid.clamp(max=max(src_sorted.numel() - 1, 0))
        nbr_src = torch.where(valid, src_sorted[nbr_eid], torch.zeros_like(nbr_eid))
        return order, nbr_src, nbr_eid, valid, degree

    def _local_decode(self, node_ids, cache, enc, h_S, e_s_sorted, e_v_sorted,
                      nbr_src, nbr_eid, valid):
        """Run the decoder stack for `node_ids` and return the last layer's output.

        Shapes: cache[j] is ([S, N, ds], [S, N, dv, 3]); enc is ([N, ds], [N, dv, 3])
        and is shared across samples because it is never mutated. Returns
        ([S, M, ds], [S, M, dv, 3]) for the M nodes in `node_ids`.

        The periodic refresh uses `_refresh_decode` instead: it updates a
        growing set of nodes where most padded neighbour slots would be dead,
        so a compact edge list beats this padded table there.
        """
        n_samples = cache[0][0].shape[0]
        num_sel = node_ids.shape[0]

        sel_src = nbr_src[node_ids]                                   # [M, K]
        sel_eid = nbr_eid[node_ids]                                   # [M, K]
        sel_valid = valid[node_ids]
        forward = sel_valid & (sel_src < node_ids.unsqueeze(1))       # src < dst
        degree = sel_valid.sum(dim=1).clamp(min=1)                    # [M]
        num_nbr = sel_src.shape[1]

        def expand_samples(tensor):
            return tensor.unsqueeze(0).expand(n_samples, *tensor.shape)

        # Edge features are constant across decoder layers (n_edge_gvps == 0),
        # so build them once per call rather than once per layer.
        edge_s = expand_samples(e_s_sorted[sel_eid])                  # [S, M, K, de]
        edge_v = expand_samples(e_v_sorted[sel_eid])                  # [S, M, K, dve, 3]
        seq_feat = h_S[:, sel_src] * forward.unsqueeze(0).unsqueeze(-1)
        edge_s = torch.cat([edge_s, seq_feat], dim=-1)

        fwd_s = forward.unsqueeze(0).unsqueeze(-1)
        fwd_v = fwd_s.unsqueeze(-1)
        enc_src_s = expand_samples(enc[0][sel_src])
        enc_src_v = expand_samples(enc[1][sel_src])
        enc_dst_s = expand_samples(enc[0][node_ids]).unsqueeze(2).expand(-1, -1, num_nbr, -1)
        enc_dst_v = expand_samples(enc[1][node_ids]).unsqueeze(2).expand(-1, -1, num_nbr, -1, -1)

        flat_edge = (edge_s.reshape(-1, edge_s.shape[-1]),
                     edge_v.reshape(-1, *edge_v.shape[-2:]))
        mask_flat_s = sel_valid.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, 1)
        mask_flat_v = mask_flat_s.unsqueeze(-1)

        out = None
        for layer_idx, layer in enumerate(self.decoder_layers):
            node_s, node_v = cache[layer_idx]

            src_s = torch.where(fwd_s, node_s[:, sel_src], enc_src_s)
            src_v = torch.where(fwd_v, node_v[:, sel_src], enc_src_v)
            dst_cache_s = node_s[:, node_ids].unsqueeze(2).expand(-1, -1, num_nbr, -1)
            dst_cache_v = node_v[:, node_ids].unsqueeze(2).expand(-1, -1, num_nbr, -1, -1)
            dst_s = torch.where(fwd_s, dst_cache_s, enc_dst_s)
            dst_v = torch.where(fwd_v, dst_cache_v, enc_dst_v)

            # GVP assumes 2-D scalars / 3-D vectors, so flatten S*M*K into rows.
            message = layer.conv.message_func((
                torch.cat([src_s.reshape(-1, src_s.shape[-1]),
                           flat_edge[0],
                           dst_s.reshape(-1, dst_s.shape[-1])], dim=-1),
                torch.cat([src_v.reshape(-1, *src_v.shape[-2:]),
                           flat_edge[1],
                           dst_v.reshape(-1, *dst_v.shape[-2:])], dim=-2),
            ))

            msg_s = (message[0] * mask_flat_s).view(n_samples, num_sel, num_nbr, -1)
            msg_v = (message[1] * mask_flat_v).view(n_samples, num_sel, num_nbr,
                                                    *message[1].shape[-2:])
            delta_s = msg_s.sum(dim=2) / degree.view(1, -1, 1)
            delta_v = msg_v.sum(dim=2) / degree.view(1, -1, 1, 1)

            base_s = node_s[:, node_ids]
            base_v = node_v[:, node_ids]
            hid = (base_s + delta_s, base_v + delta_v)

            flat_hid = (hid[0].reshape(-1, hid[0].shape[-1]),
                        hid[1].reshape(-1, *hid[1].shape[-2:]))
            ff_out = layer.ff_func(layer.norm[1](flat_hid))
            out = (hid[0] + ff_out[0].view_as(hid[0]),
                   hid[1] + ff_out[1].view_as(hid[1]))

            if layer_idx < len(self.decoder_layers) - 1:
                cache[layer_idx + 1][0][:, node_ids] = out[0]
                cache[layer_idx + 1][1][:, node_ids] = out[1]

        return out

    def _refresh_decode(self, nodes, selected, cache, enc, h_S, e_s_sorted, e_v_sorted,
                        src_sorted, dst_sorted):
        """Periodic staleness refresh over all already-decoded nodes.

        `nodes` are the node ids to update and `selected` indexes the edges that
        participate; callers compute both, which lets the single-graph and the
        cross-target batched decoders share this routine.

        Uses a compact edge list rather than the padded [M, K] neighbour table:
        during a refresh M grows with the step index while most padded slots are
        invalid (in-degree varies ~39 mean vs ~65 max, and `src <= limit` masks
        out most of the rest), so padding wastes a large fraction of the work.

        Matches sample_v3's refresh semantics: only edges with BOTH endpoints
        already decoded participate, which is a different edge set from the
        per-residue step.
        """
        n_samples, num_nodes = cache[0][0].shape[0], cache[0][0].shape[1]

        if selected.numel() == 0:
            return  # sample_v3 skips the refresh entirely when no such edges exist

        src = src_sorted[selected]
        dst = dst_sorted[selected]
        forward = src < dst

        degree = torch.zeros(num_nodes, device=src.device, dtype=cache[0][0].dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=degree.dtype))
        degree = degree.clamp(min=1)

        def expand_samples(tensor):
            return tensor.unsqueeze(0).expand(n_samples, *tensor.shape)

        edge_s = torch.cat([
            expand_samples(e_s_sorted[selected]),
            h_S[:, src] * forward.view(1, -1, 1),
        ], dim=-1)
        edge_v = expand_samples(e_v_sorted[selected])
        flat_edge_s = edge_s.reshape(-1, edge_s.shape[-1])
        flat_edge_v = edge_v.reshape(-1, *edge_v.shape[-2:])

        fwd_s = forward.view(1, -1, 1)
        fwd_v = fwd_s.unsqueeze(-1)
        enc_src_s, enc_src_v = expand_samples(enc[0][src]), expand_samples(enc[1][src])
        enc_dst_s, enc_dst_v = expand_samples(enc[0][dst]), expand_samples(enc[1][dst])

        for layer_idx, layer in enumerate(self.decoder_layers):
            node_s, node_v = cache[layer_idx]

            src_s = torch.where(fwd_s, node_s[:, src], enc_src_s)
            src_v = torch.where(fwd_v, node_v[:, src], enc_src_v)
            dst_s = torch.where(fwd_s, node_s[:, dst], enc_dst_s)
            dst_v = torch.where(fwd_v, node_v[:, dst], enc_dst_v)

            message = layer.conv.message_func((
                torch.cat([src_s.reshape(-1, src_s.shape[-1]), flat_edge_s,
                           dst_s.reshape(-1, dst_s.shape[-1])], dim=-1),
                torch.cat([src_v.reshape(-1, *src_v.shape[-2:]), flat_edge_v,
                           dst_v.reshape(-1, *dst_v.shape[-2:])], dim=-2),
            ))

            msg_s = message[0].view(n_samples, -1, message[0].shape[-1])
            msg_v = message[1].view(n_samples, -1, *message[1].shape[-2:])

            delta_s = torch.zeros(n_samples, num_nodes, msg_s.shape[-1],
                                  device=msg_s.device, dtype=msg_s.dtype)
            delta_v = torch.zeros(n_samples, num_nodes, *msg_v.shape[-2:],
                                  device=msg_v.device, dtype=msg_v.dtype)
            delta_s.index_add_(1, dst, msg_s)
            delta_v.index_add_(1, dst, msg_v)
            delta_s = delta_s / degree.view(1, -1, 1)
            delta_v = delta_v / degree.view(1, -1, 1, 1)

            hid = (node_s[:, nodes] + delta_s[:, nodes],
                   node_v[:, nodes] + delta_v[:, nodes])
            flat_hid = (hid[0].reshape(-1, hid[0].shape[-1]),
                        hid[1].reshape(-1, *hid[1].shape[-2:]))
            ff_out = layer.ff_func(layer.norm[1](flat_hid))
            out = (hid[0] + ff_out[0].view_as(hid[0]),
                   hid[1] + ff_out[1].view_as(hid[1]))

            if layer_idx < len(self.decoder_layers) - 1:
                cache[layer_idx + 1][0][:, nodes] = out[0]
                cache[layer_idx + 1][1][:, nodes] = out[1]

    @torch.no_grad()
    def sample_batched(
        self,
        batch,
        return_logits: Optional[bool] = False,
        refresh_interval: int = 0,
        forced_seq: Optional[torch.Tensor] = None,
    ):
        """Decode every graph in the batch in lockstep.

        sample_v3 decodes graphs one after another. Because the decode is bound
        by per-step dispatch overhead rather than arithmetic, advancing all
        graphs together costs barely more per step than advancing one, so this
        is close to a free throughput multiplier for screening many targets.

        Graphs are concatenated into a single node space with per-graph offsets
        (no padding), so step i decodes node `offset_b + i` for every graph still
        running. Graphs shorter than the longest simply drop out of `active`.

        Returns per-graph sequences as a list, since lengths differ.

        Reproducibility:
          * ONE graph  -- sampled sequences are byte-identical to sample_v3,
            measured on all 96 benchmark targets.
          * SEVERAL graphs -- numerically equivalent but not byte-identical:
            draws interleave across graphs, so the RNG is consumed in a
            different order than decoding them separately. The distribution is
            unchanged; the particular seeded sequences are not.

        `forced_seq` teacher-forces the decode, which is how equivalence is
        checked without stochastic draws in the way.

        Requires `n_edge_gvps == 0`; see `_decoder_updates_edge_features`.
        `sample()` falls back to `sample_v3` in that case, but this raises for
        callers that reach it directly rather than returning wrong numbers.
        """
        if self._decoder_updates_edge_features():
            raise NotImplementedError(
                "sample_batched does not implement the per-layer edge-feature "
                "update that n_edge_gvps > 0 enables: it builds edge features "
                "once per step and reuses them across decoder layers, which is "
                "exact only when the layers leave edge features untouched. Use "
                "sample_v3 for this model, or implement the edge-update path."
            )

        device = batch.edge_index.device

        h_V = self.W_v((batch.node_s, batch.node_v))
        h_E = self.W_e((batch.edge_s, batch.edge_v))
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, batch.edge_index, h_E)

        original_indices = torch.arange(batch.seq.shape[0], device=device)
        decoder_edge_idx_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.decoder_edge_idx_length, dim=0)])
        decoder_edge_feat_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.decoder_edge_feat_length, dim=0)])
        decoder_node_ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(batch.num_decoder_nodes, dim=0)])

        all_objects_h_V = list(zip(split_batch(h_V[0], batch.batch),
                                   split_batch(h_V[1], batch.batch)))

        # --- per-graph encode + pool (cheap: once per target, not per step) ---
        enc_s_parts, enc_v_parts = [], []
        edge_parts, e_s_parts, e_v_parts = [], [], []
        sizes, pooled_indices = [], []
        node_offset = 0

        for batch_idx, h_V_batch in enumerate(all_objects_h_V):
            edge_index_decoder = (
                batch.decoder_edge_index[:, decoder_edge_idx_ptr[batch_idx]:
                                            decoder_edge_idx_ptr[batch_idx + 1]]
                - decoder_node_ptr[batch_idx])
            feat_slice = slice(decoder_edge_feat_ptr[batch_idx],
                               decoder_edge_feat_ptr[batch_idx + 1])

            h_V_pooled, h_E_pooled, edge_index_decoder, original_indices_pooled = \
                self.pool_for_sampling(
                    h_V_=h_V_batch,
                    h_E_=(self.W_e((batch.decoder_edge_s[feat_slice],
                                    batch.decoder_edge_v[feat_slice]))),
                    edge_index_local=edge_index_decoder,
                    homomer_index=batch.homo_idx[batch.batch == batch_idx],
                    virtual_mask=batch.virtual_mask[batch.batch == batch_idx],
                    decoder_edge_mask=batch.decoder_edge_mask[feat_slice],
                    n_edges=batch.decoder_edge_idx_length[batch_idx].item(),
                    original_indices=original_indices[
                        decoder_node_ptr[batch_idx]:decoder_node_ptr[batch_idx + 1]],
                )

            for layer in self.pooled_encoder_layers:
                h_V_pooled, h_E_pooled = layer(h_V_pooled, edge_index_decoder, h_E_pooled)

            num_nodes = h_V_pooled[0].shape[0]
            enc_s_parts.append(h_V_pooled[0])
            enc_v_parts.append(h_V_pooled[1])
            edge_parts.append(edge_index_decoder + node_offset)
            e_s_parts.append(h_E_pooled[0])
            e_v_parts.append(h_E_pooled[1])
            sizes.append(num_nodes)
            pooled_indices.append(original_indices_pooled)
            node_offset += num_nodes

        # --- fuse into one global graph ---
        enc = (torch.cat(enc_s_parts, dim=0), torch.cat(enc_v_parts, dim=0))
        edge_index_global = torch.cat(edge_parts, dim=1)
        total_nodes = node_offset

        order, nbr_src, nbr_eid, valid, _deg = self._build_decoder_csr(
            edge_index_global, total_nodes)
        e_s_sorted = torch.cat(e_s_parts, dim=0)[order]
        e_v_sorted = torch.cat(e_v_parts, dim=0)[order]
        src_sorted = edge_index_global[0][order]
        dst_sorted = edge_index_global[1][order]

        sizes_t = torch.tensor(sizes, device=device)
        offsets_t = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                               torch.cumsum(sizes_t, dim=0)[:-1]])
        local_index = torch.cat([torch.arange(n, device=device) for n in sizes])
        graph_of = torch.cat([torch.full((n,), b, device=device, dtype=torch.long)
                              for b, n in enumerate(sizes)])

        n_samples = self.n_samples
        model_dtype = enc[0].dtype
        cache = [(enc[0].unsqueeze(0).repeat(n_samples, 1, 1),
                  enc[1].unsqueeze(0).repeat(n_samples, 1, 1, 1))
                 for _ in self.decoder_layers]
        seq = torch.zeros(n_samples, total_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples, total_nodes, self.out_dim,
                          device=device, dtype=model_dtype)
        logits = torch.zeros(n_samples, total_nodes, self.out_dim,
                             device=device, dtype=model_dtype)

        for i in range(max(sizes)):
            active = (sizes_t > i).nonzero(as_tuple=True)[0]
            node_ids = offsets_t[active] + i

            out = self._local_decode(node_ids, cache, enc, h_S,
                                     e_s_sorted, e_v_sorted, nbr_src, nbr_eid, valid)

            lgts = self.W_out((out[0].reshape(-1, out[0].shape[-1]),
                               out[1].reshape(-1, *out[1].shape[-2:])))
            lgts = lgts.view(n_samples, node_ids.shape[0], self.out_dim)
            if lgts.dtype in [torch.float16, torch.bfloat16]:
                lgts = torch.nan_to_num(lgts, nan=-1e4, posinf=1e4, neginf=-1e4)
            lgts = torch.clamp(lgts, min=-1e4, max=1e4)

            if forced_seq is not None:
                drawn = forced_seq[node_ids].to(torch.int).unsqueeze(0).expand(
                    n_samples, -1)
            else:
                drawn = Categorical(logits=lgts / self.temperature).sample()
            seq[:, node_ids] = drawn.to(torch.int)
            h_S[:, node_ids] = self.W_s(drawn.long())
            logits[:, node_ids] = lgts

            if refresh_interval > 0 and i > 0 and (i + 1) % refresh_interval == 0:
                # A graph that has finished must stop refreshing, so that each
                # graph's refresh history matches its standalone decode exactly.
                node_ok = (local_index <= i) & (sizes_t[graph_of] > i)
                selected = (node_ok[src_sorted] & node_ok[dst_sorted]).nonzero(
                    as_tuple=True)[0]
                self._refresh_decode(node_ok.nonzero(as_tuple=True)[0], selected,
                                     cache, enc, h_S, e_s_sorted, e_v_sorted,
                                     src_sorted, dst_sorted)

        sequences = [seq[:, offsets_t[b]:offsets_t[b] + sizes[b]] for b in range(len(sizes))]
        if return_logits:
            per_graph_logits = [logits[:, offsets_t[b]:offsets_t[b] + sizes[b]]
                                for b in range(len(sizes))]
            return sequences, per_graph_logits, pooled_indices
        return sequences

def split_batch(x, batch):
    """Split tensor x according to batch indices."""
    batch_size = int(batch.max()) + 1
    return [x[batch == i] for i in range(batch_size)]


def pool_conformations(node_features, virtual_mask):
    """
    Average node features across k conformations, accounting for virtual nodes.
    Used for single-chain mode with k aligned conformations.

    Args:
        node_features: (s=[n_nodes, k, d_s], v=[n_nodes, k, d_v, 3])
        virtual_mask: [n_nodes, k] - True for valid nodes, False for virtual

    Returns:
        Pooled features: (s=[n_nodes, d_s], v=[n_nodes, d_v, 3])
    """
    s, v = node_features

    mask_s = virtual_mask.unsqueeze(-1)
    mask_v = virtual_mask.unsqueeze(-1).unsqueeze(-1)

    s_masked = s * mask_s
    v_masked = v * mask_v

    valid_count_s = mask_s.sum(dim=1, keepdim=False).clamp(min=1)
    valid_count_v = mask_v.sum(dim=1, keepdim=False).clamp(min=1)

    s_pooled = s_masked.sum(dim=1) / valid_count_s
    v_pooled = v_masked.sum(dim=1) / valid_count_v

    return s_pooled, v_pooled


def pool_edges_conformations(edge_features, edge_mask):
    """
    Average edge features across k conformations, accounting for virtual edges.
    Used for single-chain mode with k aligned conformations.

    Args:
        edge_features: (s=[n_edges, k, d_s], v=[n_edges, k, d_v, 3])
        edge_mask: [n_edges, k] - True for valid edges, False for virtual

    Returns:
        Pooled features: (s=[n_edges, d_s], v=[n_edges, d_v, 3])
    """
    s, v = edge_features

    mask_s = edge_mask.unsqueeze(-1).float()
    mask_v = mask_s.unsqueeze(-1)

    edge_denom_s = mask_s.sum(dim=1, keepdim=False).clamp(min=1)
    edge_denom_v = edge_denom_s.unsqueeze(-1)

    s_pooled = (s * mask_s).sum(dim=1) / edge_denom_s
    v_pooled = (v * mask_v).sum(dim=1) / edge_denom_v

    return s_pooled, v_pooled

