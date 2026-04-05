import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index, scatter, softmax as pyg_softmax
from torch_scatter import scatter_max

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_edge')
class HeteroGNNEdgeHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.edge_decoding = cfg.model.edge_decoding
        self.use_pair_chain_head = self.edge_decoding in {
            'pair_chain',
            'pair_chain_contextresid',
            'pair_chain_contextseqresid',
            'pair_chain_contextseqmetapathresid',
        }
        self.use_chain_context_residual = self.edge_decoding in {
            'pair_chain_contextresid',
            'pair_chain_contextseqresid',
            'pair_chain_contextseqmetapathresid',
        }
        self.use_sequence_context_residual = self.edge_decoding in {
            'pair_chain_contextseqresid',
            'pair_chain_contextseqmetapathresid',
        }
        self.use_metapath_token_residual = (
            self.edge_decoding == 'pair_chain_contextseqmetapathresid'
        )
        self.head_layers = max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt)
        # self.train_edge_inds = mask_to_index(data[cfg.dataset.task_entity].train_edge_mask).to(cfg.device)
        # self.val_edge_inds = mask_to_index(data[cfg.dataset.task_entity].val_edge_mask).to(cfg.device)
        # self.test_edge_inds = mask_to_index(data[cfg.dataset.task_entity].test_edge_mask).to(cfg.device)
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        if self.use_pair_chain_head:
            self.edge_proj = MLP(dim_in * 3, dim_in,
                                 num_layers=self.head_layers,
                                 bias=True)
            self.pair_proj = MLP(dim_in * 3, dim_in,
                                 num_layers=self.head_layers,
                                 bias=True)
            self.chain_update = MLP(dim_in * 3, dim_in,
                                    num_layers=self.head_layers,
                                    bias=True)
            self.chain_gate = nn.Linear(dim_in * 3, dim_in)
            self.pair_residual_alpha = nn.Parameter(
                torch.full((1,), math.log(0.15 / 0.85))
            )
            self.chain_residual_alpha = nn.Parameter(
                torch.full((1,), math.log(0.10 / 0.90))
            )
            self.layer_post_mp = MLP(dim_in * 3, dim_out,
                                     num_layers=self.head_layers,
                                     bias=True)
            if self.use_chain_context_residual:
                self.context_proj = MLP(dim_in * 3, dim_in,
                                        num_layers=self.head_layers,
                                        bias=True)
                self.context_head = MLP(dim_in, dim_out,
                                        num_layers=self.head_layers,
                                        bias=True)
                self.context_residual_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
            if self.use_sequence_context_residual:
                self.sequence_len = 4
                self.outgoing_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                self.incoming_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                self.sequence_proj = MLP(dim_in * 3, dim_in,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_head = MLP(dim_in, dim_out,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_residual_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
                self.sequence_time_scale = nn.Parameter(torch.tensor(86400.0))
                if self.use_metapath_token_residual:
                    self.reciprocal_proj = MLP(dim_in * 4, dim_in,
                                               num_layers=self.head_layers,
                                               bias=True)
                    self.bridge_proj = MLP(dim_in * 3, dim_in,
                                           num_layers=self.head_layers,
                                           bias=True)
                    self.metapath_query = nn.Linear(dim_in, dim_in)
                    self.metapath_key = nn.Linear(dim_in, dim_in)
                    self.metapath_value = nn.Linear(dim_in, dim_in)
                    self.metapath_head = MLP(dim_in, dim_out,
                                             num_layers=self.head_layers,
                                             bias=True)
                    self.metapath_residual_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.08 / 0.92))
                    )
        else:
            self.layer_post_mp = MLP(dim_in * 3, dim_out,
                                     num_layers=self.head_layers,
                                     bias=True)
        # requires parameter
        # self.decode_module = lambda v1, v2: \
        #     self.layer_post_mp(torch.cat((v1, v2), dim=-1))

    def _edge_mask(self, batch):
        task = cfg.dataset.task_entity
        return torch.isin(batch[task].e_id,
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])

    def _edge_inputs(self, batch):
        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index
        return torch.cat((batch[task[0]].x[edge_index[0]],
                          batch[task[2]].x[edge_index[1]],
                          batch[task].edge_attr), dim=-1), edge_index

    def _build_recent_sequence_bank(self, pair_repr, pair_nodes, pair_timestamps,
                                    num_nodes, latest_timestamps):
        seq_bank = pair_repr.new_zeros((num_nodes, self.sequence_len, pair_repr.size(-1)))
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            slot_scores, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            chosen_times = pair_timestamps[chosen_indices]
            chosen_repr = pair_repr[chosen_indices]
            recency = torch.exp(
                -(
                    latest_timestamps[valid_nodes] - chosen_times
                ).clamp(min=0) / time_scale
            ).unsqueeze(-1)
            seq_bank[valid_nodes, slot] = chosen_repr * recency
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return seq_bank

    def _build_recent_sequence_bank_with_indices(self, pair_repr, pair_nodes, pair_timestamps,
                                                 num_nodes, latest_timestamps):
        seq_bank = pair_repr.new_zeros((num_nodes, self.sequence_len, pair_repr.size(-1)))
        index_bank = pair_nodes.new_full((num_nodes, self.sequence_len), -1)
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            slot_scores, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            chosen_times = pair_timestamps[chosen_indices]
            chosen_repr = pair_repr[chosen_indices]
            recency = torch.exp(
                -(
                    latest_timestamps[valid_nodes] - chosen_times
                ).clamp(min=0) / time_scale
            ).unsqueeze(-1)
            seq_bank[valid_nodes, slot] = chosen_repr * recency
            index_bank[valid_nodes, slot] = chosen_indices
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return seq_bank, index_bank

    def _pair_chain_head(self, batch):
        task = cfg.dataset.task_entity
        mask = self._edge_mask(batch)
        edge_inputs, edge_index = self._edge_inputs(batch)
        src_nodes, dst_nodes = edge_index
        edge_repr = self.edge_proj(edge_inputs)

        num_dst_nodes = batch[task[2]].x.size(0)
        pair_key = src_nodes.to(torch.long) * num_dst_nodes + dst_nodes.to(torch.long)
        pair_keys, pair_inv = torch.unique(pair_key, sorted=True, return_inverse=True)
        num_pairs = pair_keys.numel()

        pair_mean = scatter(edge_repr, pair_inv, dim=0, dim_size=num_pairs, reduce='mean')
        pair_max, _ = scatter_max(edge_repr, pair_inv, dim=0, dim_size=num_pairs)
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, torch.zeros_like(pair_max))
        pair_repr = self.pair_proj(torch.cat((pair_mean, pair_max, pair_max - pair_mean), dim=-1))
        pair_context_repr = None
        pair_sequence_repr = None
        pair_bridge_repr = None
        pair_reciprocal_repr = None

        if task[0] == task[2]:
            num_nodes = batch[task[0]].x.size(0)
            pair_src = torch.div(pair_keys, num_dst_nodes, rounding_mode='floor')
            pair_dst = torch.remainder(pair_keys, num_dst_nodes)
            predecessor_bank = scatter(pair_repr, pair_dst, dim=0, dim_size=num_nodes, reduce='mean')
            successor_bank = scatter(pair_repr, pair_src, dim=0, dim_size=num_nodes, reduce='mean')
            prev_context = predecessor_bank[pair_src]
            next_context = successor_bank[pair_dst]
            chain_input = torch.cat((prev_context, pair_repr, next_context), dim=-1)
            chain_gate = torch.sigmoid(self.chain_gate(chain_input))
            pair_repr = pair_repr + (
                torch.sigmoid(self.chain_residual_alpha) *
                chain_gate *
                self.chain_update(chain_input)
            )
            if self.use_metapath_token_residual:
                reverse_pair_key = pair_dst.to(torch.long) * num_dst_nodes + pair_src.to(torch.long)
                reverse_pos = torch.searchsorted(pair_keys, reverse_pair_key)
                safe_reverse_pos = reverse_pos.clamp(max=max(num_pairs - 1, 0))
                reciprocal_valid = (reverse_pos < num_pairs) & (
                    pair_keys[safe_reverse_pos] == reverse_pair_key
                )
                reciprocal_pair = torch.zeros_like(pair_repr)
                reciprocal_pair[reciprocal_valid] = pair_repr[safe_reverse_pos[reciprocal_valid]]
                pair_reciprocal_repr = self.reciprocal_proj(torch.cat(
                    (
                        pair_repr,
                        reciprocal_pair,
                        pair_repr * reciprocal_pair,
                        pair_repr - reciprocal_pair,
                    ),
                    dim=-1,
                ))
            if self.use_chain_context_residual:
                pair_scores = pair_repr.norm(dim=-1)
                predecessor_focus_weights = pyg_softmax(
                    pair_scores, pair_dst, num_nodes=num_nodes
                )
                successor_focus_weights = pyg_softmax(
                    pair_scores, pair_src, num_nodes=num_nodes
                )
                predecessor_focus_bank = scatter(
                    pair_repr * predecessor_focus_weights.unsqueeze(-1),
                    pair_dst,
                    dim=0,
                    dim_size=num_nodes,
                    reduce='sum'
                )
                successor_focus_bank = scatter(
                    pair_repr * successor_focus_weights.unsqueeze(-1),
                    pair_src,
                    dim=0,
                    dim_size=num_nodes,
                    reduce='sum'
                )
                pair_context_repr = self.context_proj(torch.cat(
                    (
                        predecessor_focus_bank[pair_src],
                        successor_focus_bank[pair_dst],
                        predecessor_focus_bank[pair_src] * successor_focus_bank[pair_dst],
                    ),
                    dim=-1,
                ))
            if self.use_sequence_context_residual and hasattr(batch[task], 'timestamps'):
                edge_timestamps = batch[task].timestamps.to(edge_repr.device).float().view(-1)
                pair_timestamps, _ = scatter_max(
                    edge_timestamps, pair_inv, dim=0, dim_size=num_pairs
                )
                pair_timestamps = torch.where(
                    torch.isfinite(pair_timestamps),
                    pair_timestamps,
                    torch.zeros_like(pair_timestamps),
                )
                outgoing_latest, _ = scatter_max(
                    pair_timestamps, pair_src, dim=0, dim_size=num_nodes
                )
                outgoing_latest = torch.where(
                    torch.isfinite(outgoing_latest),
                    outgoing_latest,
                    torch.zeros_like(outgoing_latest),
                )
                incoming_latest, _ = scatter_max(
                    pair_timestamps, pair_dst, dim=0, dim_size=num_nodes
                )
                incoming_latest = torch.where(
                    torch.isfinite(incoming_latest),
                    incoming_latest,
                    torch.zeros_like(incoming_latest),
                )
                if self.use_metapath_token_residual:
                    outgoing_sequence_bank, outgoing_sequence_indices = (
                        self._build_recent_sequence_bank_with_indices(
                            pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest
                        )
                    )
                    incoming_sequence_bank, incoming_sequence_indices = (
                        self._build_recent_sequence_bank_with_indices(
                            pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest
                        )
                    )
                else:
                    outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest
                    )
                    incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest
                    )
                    outgoing_sequence_indices = None
                    incoming_sequence_indices = None
                outgoing_state = self.outgoing_sequence_encoder(
                    outgoing_sequence_bank[pair_src]
                )[1].squeeze(0)
                incoming_state = self.incoming_sequence_encoder(
                    incoming_sequence_bank[pair_dst]
                )[1].squeeze(0)
                sequence_interaction = outgoing_state * incoming_state
                pair_sequence_repr = self.sequence_proj(torch.cat(
                    (
                        outgoing_state,
                        incoming_state,
                        sequence_interaction,
                    ),
                    dim=-1,
                ))
                if self.use_metapath_token_residual:
                    src_sequence_bank = outgoing_sequence_bank[pair_src]
                    dst_sequence_bank = incoming_sequence_bank[pair_dst]
                    src_sequence_indices = outgoing_sequence_indices[pair_src]
                    dst_sequence_indices = incoming_sequence_indices[pair_dst]
                    src_mid = torch.full_like(src_sequence_indices, -1)
                    dst_mid = torch.full_like(dst_sequence_indices, -1)
                    src_valid = src_sequence_indices >= 0
                    dst_valid = dst_sequence_indices >= 0
                    src_mid[src_valid] = pair_dst[src_sequence_indices[src_valid]]
                    dst_mid[dst_valid] = pair_src[dst_sequence_indices[dst_valid]]
                    match_mask = (
                        src_valid.unsqueeze(2) &
                        dst_valid.unsqueeze(1) &
                        (src_mid.unsqueeze(2) == dst_mid.unsqueeze(1))
                    )
                    match_weight = match_mask.float()
                    match_count = match_weight.sum(dim=(1, 2)).clamp(min=1.0)
                    bridge_out = (
                        match_weight.unsqueeze(-1) * src_sequence_bank.unsqueeze(2)
                    ).sum(dim=(1, 2)) / match_count.unsqueeze(-1)
                    bridge_in = (
                        match_weight.unsqueeze(-1) * dst_sequence_bank.unsqueeze(1)
                    ).sum(dim=(1, 2)) / match_count.unsqueeze(-1)
                    bridge_valid = (match_mask.sum(dim=(1, 2)) > 0).unsqueeze(-1)
                    bridge_out = bridge_out * bridge_valid.float()
                    bridge_in = bridge_in * bridge_valid.float()
                    pair_bridge_repr = self.bridge_proj(torch.cat(
                        (
                            bridge_out,
                            bridge_in,
                            bridge_out * bridge_in,
                        ),
                        dim=-1,
                    ))

        pair_edge_repr = pair_repr[pair_inv]
        edge_repr = edge_repr + torch.sigmoid(self.pair_residual_alpha) * pair_edge_repr
        pred = self.layer_post_mp(torch.cat(
            (edge_repr[mask], pair_edge_repr[mask], edge_repr[mask] * pair_edge_repr[mask]),
            dim=-1
        ))
        if self.use_chain_context_residual:
            if pair_context_repr is None:
                pair_context_repr = torch.zeros_like(pair_repr)
            context_logits = self.context_head(pair_context_repr[pair_inv][mask])
            pred = pred + torch.sigmoid(self.context_residual_alpha) * context_logits
        if self.use_sequence_context_residual:
            if pair_sequence_repr is None:
                pair_sequence_repr = torch.zeros_like(pair_repr)
            sequence_logits = self.sequence_head(pair_sequence_repr[pair_inv][mask])
            pred = pred + torch.sigmoid(self.sequence_residual_alpha) * sequence_logits
        if self.use_metapath_token_residual:
            if pair_bridge_repr is None:
                pair_bridge_repr = torch.zeros_like(pair_repr)
            if pair_reciprocal_repr is None:
                pair_reciprocal_repr = torch.zeros_like(pair_repr)
            if pair_context_repr is None:
                pair_context_repr = torch.zeros_like(pair_repr)
            if pair_sequence_repr is None:
                pair_sequence_repr = torch.zeros_like(pair_repr)
            metapath_tokens = torch.stack(
                (
                    pair_context_repr,
                    pair_sequence_repr,
                    pair_reciprocal_repr,
                    pair_bridge_repr,
                ),
                dim=1,
            )
            token_mask = metapath_tokens.abs().sum(dim=-1) > 0
            token_query = self.metapath_query(pair_repr).unsqueeze(1)
            token_keys = self.metapath_key(metapath_tokens)
            token_values = self.metapath_value(metapath_tokens)
            token_scores = (token_query * token_keys).sum(dim=-1) / math.sqrt(pair_repr.size(-1))
            token_scores = token_scores.masked_fill(~token_mask, -1e9)
            token_weights = torch.softmax(token_scores, dim=-1).unsqueeze(-1)
            token_weights = token_weights * token_mask.unsqueeze(-1).float()
            token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            metapath_repr = (token_weights * token_values).sum(dim=1)
            metapath_logits = self.metapath_head(
                metapath_repr[pair_inv][mask]
            )
            pred = pred + (
                torch.sigmoid(self.metapath_residual_alpha) *
                metapath_logits
            )
        return pred, batch[task].y[mask]

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        # There could be multi-edge between node pair, using edge id is the safest way
        # mask = torch.isin(getattr(self, f'{batch.split}_edge_inds')[batch[task].e_id], 
        #                   getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        mask = self._edge_mask(batch)

        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        return torch.cat((batch[task[0]].x[edge_index[0, mask]], 
                          batch[task[2]].x[edge_index[1, mask]], 
                          batch[task].edge_attr[mask]), dim=-1), \
               batch[task].y[mask]
    

    def forward(self, batch):
        # TODO: add homogeneous graph support
        # batch.x_dict[cfg.dataset.task_entity] = self.layer_post_mp(batch.x_dict[cfg.dataset.task_entity])
        # pred, label = self._apply_index(batch)
    
        # if cfg.model.edge_decoding != 'concat':
        #     batch = self.layer_post_mp(batch)
        if self.use_pair_chain_head:
            return self._pair_chain_head(batch)
        pred, label = self._apply_index(batch)
        # nodes_first = pred[0]
        # nodes_second = pred[1]
        # pred = self.decode_module(nodes_first, nodes_second)
        pred = self.layer_post_mp(pred)

        return pred, label
    
        # if not self.training:  # Compute extra stats when in evaluation mode.
        #     stats = self.compute_mrr(batch)
        #     return pred, label, stats
        # else:
        #     return pred, label
