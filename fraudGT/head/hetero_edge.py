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
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_chain_context_residual = self.edge_decoding in {
            'pair_chain_contextresid',
            'pair_chain_contextseqresid',
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_sequence_context_residual = self.edge_decoding in {
            'pair_chain_contextseqresid',
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_pair_internal_sequence = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebank',
                'pair_chain_contextseqpairseqbridgebankmotiflite',
                'pair_chain_contextseqpairseqbridgebankwindow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_sequence_bridge_bank = self.edge_decoding in {
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_sequence_bridge_motif_lite = (
            self.edge_decoding == 'pair_chain_contextseqpairseqbridgebankmotiflite'
        )
        self.use_target_sequence_select = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_difference_fusion = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_terminal_role_flow = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_boundary_lag_flow = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            }
        )
        self.use_support_conditioned_mixture = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix'
        )
        self.use_sequence_bridge_bank_window = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.head_layers = max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt)
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        if self.use_pair_chain_head:
            self.edge_proj = MLP(dim_in * 3, dim_in,
                                 num_layers=self.head_layers,
                                 bias=True)
            if self.use_pair_internal_sequence:
                self.pair_sequence_len = 4
                self.pair_edge_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                self.pair_proj = MLP(dim_in * 4, dim_in,
                                     num_layers=self.head_layers,
                                     bias=True)
            else:
                self.pair_proj = MLP(dim_in * 3, dim_in,
                                     num_layers=self.head_layers,
                                     bias=True)
            if self.use_sequence_bridge_bank_window:
                self.pair_window_gate = nn.Linear(dim_in * 3, dim_in)
                self.pair_window_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
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
                context_fusion_mult = 4 if self.use_difference_fusion else 3
                self.context_proj = MLP(dim_in * context_fusion_mult, dim_in,
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
                sequence_fusion_mult = 4 if self.use_difference_fusion else 3
                self.sequence_proj = MLP(dim_in * sequence_fusion_mult, dim_in,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_head = MLP(dim_in, dim_out,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_residual_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
                if self.use_target_sequence_select:
                    self.outgoing_sequence_select_score = MLP(
                        dim_in * 3, 1,
                        num_layers=self.head_layers,
                        bias=True,
                    )
                    self.incoming_sequence_select_score = MLP(
                        dim_in * 3, 1,
                        num_layers=self.head_layers,
                        bias=True,
                    )
                    self.sequence_select_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.30 / 0.70))
                    )
                self.sequence_time_scale = nn.Parameter(torch.tensor(86400.0))
                if self.use_sequence_bridge_bank_window:
                    self.fast_time_scale_log = nn.Parameter(torch.tensor(math.log(0.35)))
                    self.slow_time_scale_log = nn.Parameter(torch.tensor(math.log(3.0)))
                    self.sequence_window_gate = nn.Linear(dim_in * 3, dim_in)
                    self.sequence_window_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.10 / 0.90))
                    )
                if self.use_sequence_bridge_bank:
                    self.bridge_partner_proj = MLP(dim_in * 2 + 2, dim_in,
                                                   num_layers=self.head_layers,
                                                   bias=True)
                    bridge_fusion_mult = 4 if self.use_difference_fusion else 3
                    self.bridge_bank_proj = MLP(dim_in * bridge_fusion_mult, dim_in,
                                                num_layers=self.head_layers,
                                                bias=True)
                    self.bridge_bank_gate = nn.Linear(dim_in * 3, dim_in)
                    self.bridge_bank_update = MLP(dim_in * 3, dim_in,
                                                  num_layers=self.head_layers,
                                                  bias=True)
                    self.bridge_bank_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.10 / 0.90))
                    )
                    if self.use_terminal_role_flow:
                        self.terminal_role_proj = MLP(
                            dim_in * bridge_fusion_mult, dim_in,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_proj = MLP(
                            dim_in * bridge_fusion_mult + 6, dim_in,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_head = MLP(
                            dim_in, dim_out,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_residual_alpha = nn.Parameter(
                            torch.full((1,), math.log(0.08 / 0.92))
                        )
                        if self.use_boundary_lag_flow:
                            self.boundary_lag_slot_proj = MLP(
                                dim_in * bridge_fusion_mult + 2, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_proj = MLP(
                                dim_in * 2 + 2, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_residual_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.05 / 0.95))
                            )
                    if self.use_support_conditioned_mixture:
                        self.support_feature_dim = 14
                        self.edge_fallback_head = MLP(
                            dim_in, dim_out,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.structure_mix_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.structure_mix_bias = nn.Parameter(
                            torch.tensor(math.log(0.20 / 0.80))
                        )
                        self.sequence_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.sequence_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.20 / 0.80))
                        )
                        self.terminal_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.15 / 0.85))
                        )
                        self.boundary_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.boundary_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.10 / 0.90))
                        )
                    if self.use_sequence_bridge_motif_lite:
                        self.bridge_leg_pair_proj = MLP(dim_in * 3 + 1, dim_in,
                                                        num_layers=self.head_layers,
                                                        bias=True)
                        self.bridge_leg_bank_proj = MLP(dim_in * 3, dim_in,
                                                        num_layers=self.head_layers,
                                                        bias=True)
                        self.bridge_leg_bank_gate = nn.Linear(dim_in * 3, dim_in)
                        self.bridge_leg_bank_update = MLP(dim_in * 3, dim_in,
                                                          num_layers=self.head_layers,
                                                          bias=True)
                        self.bridge_leg_bank_alpha = nn.Parameter(
                            torch.full((1,), math.log(0.08 / 0.92))
                        )
        else:
            self.layer_post_mp = MLP(dim_in * 3, dim_out,
                                     num_layers=self.head_layers,
                                     bias=True)

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
                                    num_nodes, latest_timestamps, time_scale=None):
        seq_bank = pair_repr.new_zeros((num_nodes, self.sequence_len, pair_repr.size(-1)))
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        if time_scale is None:
            time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
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

    def _build_recent_pair_sequence_bank(self, edge_repr, pair_inv, edge_timestamps,
                                         num_pairs, latest_timestamps, time_scale=None):
        seq_bank = edge_repr.new_zeros((num_pairs, self.pair_sequence_len, edge_repr.size(-1)))
        remaining_scores = edge_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        if time_scale is None:
            time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(edge_timestamps, dtype=torch.bool)

        for slot in range(self.pair_sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_inv, dim=0, dim_size=num_pairs
            )
            valid_pairs = scatter(
                remaining_mask.float(), pair_inv, dim=0, dim_size=num_pairs, reduce='sum'
            ) > 0
            if not valid_pairs.any():
                break
            chosen_indices = slot_indices[valid_pairs]
            chosen_times = edge_timestamps[chosen_indices]
            chosen_repr = edge_repr[chosen_indices]
            recency = torch.exp(
                -(
                    latest_timestamps[valid_pairs] - chosen_times
                ).clamp(min=0) / time_scale
            ).unsqueeze(-1)
            seq_bank[valid_pairs, slot] = chosen_repr * recency
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return seq_bank

    def _build_recent_partner_bank(self, pair_nodes, partner_nodes, pair_timestamps,
                                   num_nodes):
        partner_bank = pair_nodes.new_full((num_nodes, self.sequence_len), -1)
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            partner_bank[valid_nodes, slot] = partner_nodes[chosen_indices]
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return partner_bank

    def _build_recent_timestamp_bank(self, pair_nodes, pair_timestamps, num_nodes):
        time_bank = pair_timestamps.new_zeros((num_nodes, self.sequence_len))
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            time_bank[valid_nodes, slot] = pair_timestamps[chosen_indices]
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return time_bank

    def _filter_sequence_bank(self, sequence_bank, query_repr, scorer):
        query_bank = query_repr.unsqueeze(1).expand(-1, sequence_bank.size(1), -1)
        gate_input = torch.cat(
            (
                query_bank,
                sequence_bank,
                query_bank * sequence_bank,
            ),
            dim=-1,
        )
        gate = torch.sigmoid(
            scorer(gate_input.reshape(-1, gate_input.size(-1)))
        ).view(sequence_bank.size(0), sequence_bank.size(1), 1)
        alpha = torch.sigmoid(self.sequence_select_alpha)
        scale = 1.0 - alpha + alpha * gate
        valid = (sequence_bank.abs().sum(dim=-1, keepdim=True) > 0).float()
        return sequence_bank * valid * scale

    def _cosine_feature(self, left_repr, right_repr):
        if left_repr is None or right_repr is None:
            return None
        left_norm = left_repr.norm(dim=-1, keepdim=True)
        right_norm = right_repr.norm(dim=-1, keepdim=True)
        valid = (left_norm > 0) & (right_norm > 0)
        cosine = F.cosine_similarity(left_repr, right_repr, dim=-1, eps=1e-6).unsqueeze(-1)
        cosine = 0.5 * (cosine + 1.0)
        return torch.where(valid, cosine, torch.zeros_like(cosine))

    def _partner_overlap_ratio(self, bank_a, bank_b):
        valid_a = bank_a >= 0
        valid_b = bank_b >= 0
        eq = (
            bank_a.unsqueeze(2) == bank_b.unsqueeze(1)
        ) & valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
        match_any = eq.any(dim=2)
        overlap = (match_any.float() * valid_a.float()).sum(dim=1, keepdim=True)
        denom = valid_a.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return overlap / denom

    def _boundary_support_features(self, src_time, dst_time):
        valid = (src_time > 0) & (dst_time > 0)
        valid_count = valid.float().sum(dim=1, keepdim=True)
        valid_ratio = valid_count / float(self.sequence_len)
        lag = dst_time - src_time
        after_ratio = (
            ((lag >= 0) & valid).float().sum(dim=1, keepdim=True) /
            valid_count.clamp(min=1.0)
        )
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        lag_align = torch.exp(-lag.abs() / time_scale) * valid.float()
        lag_align = lag_align.sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)
        return valid_ratio, after_ratio, lag_align

    def _recent_overlap_partner_repr(self, bank_a, bank_b, node_x):
        valid_a = bank_a >= 0
        valid_b = bank_b >= 0
        eq = (
            bank_a.unsqueeze(2) == bank_b.unsqueeze(1)
        ) & valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
        match_any = eq.any(dim=2)
        overlap_count = (match_any.float() * valid_a.float()).sum(dim=1)
        slot_index = torch.arange(
            bank_a.size(1), device=bank_a.device, dtype=torch.float32
        )
        slot_weight = 1.0 / (1.0 + slot_index)
        weighted_overlap = (
            match_any.float() * slot_weight.unsqueeze(0)
        ).sum(dim=1)
        gather_ids = bank_a.clamp(min=0)
        matched_emb = node_x[gather_ids] * match_any.unsqueeze(-1).float()
        mean_emb = matched_emb.sum(dim=1) / overlap_count.unsqueeze(-1).clamp(min=1.0)
        max_emb = matched_emb.masked_fill(~match_any.unsqueeze(-1), -1e9).max(dim=1).values
        max_emb = torch.where(
            overlap_count.unsqueeze(-1) > 0,
            max_emb,
            torch.zeros_like(max_emb),
        )
        return self.bridge_partner_proj(torch.cat(
            (
                mean_emb,
                max_emb,
                overlap_count.unsqueeze(-1),
                weighted_overlap.unsqueeze(-1),
            ),
            dim=-1,
        ))

    def _recent_overlap_leg_repr(self, bank_a, bank_b, seq_a, seq_b, time_a, time_b):
        batch_size = bank_a.size(0)
        dim = seq_a.size(-1)
        device = seq_a.device
        acc_sum = seq_a.new_zeros((batch_size, dim))
        acc_weight = seq_a.new_zeros((batch_size, 1))
        max_token = seq_a.new_full((batch_size, dim), -1e9)
        has_match = torch.zeros(batch_size, dtype=torch.bool, device=device)
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)

        for i in range(self.sequence_len):
            a_ids = bank_a[:, i]
            a_valid = a_ids >= 0
            a_seq = seq_a[:, i]
            a_time = time_a[:, i]
            for j in range(self.sequence_len):
                b_ids = bank_b[:, j]
                match = a_valid & (b_ids >= 0) & (a_ids == b_ids)
                if not match.any():
                    continue
                b_seq = seq_b[:, j]
                b_time = time_b[:, j]
                align = torch.exp(
                    -(a_time - b_time).abs() / time_scale
                ).unsqueeze(-1)
                token = self.bridge_leg_pair_proj(torch.cat(
                    (
                        a_seq,
                        b_seq,
                        a_seq * b_seq,
                        align,
                    ),
                    dim=-1,
                ))
                masked_align = align * match.unsqueeze(-1).float()
                acc_sum = acc_sum + token * masked_align
                acc_weight = acc_weight + masked_align
                max_token = torch.where(
                    match.unsqueeze(-1),
                    torch.maximum(max_token, token),
                    max_token,
                )
                has_match = has_match | match

        mean_token = acc_sum / acc_weight.clamp(min=1e-6)
        max_token = torch.where(
            has_match.unsqueeze(-1),
            max_token,
            torch.zeros_like(max_token),
        )
        return self.bridge_leg_bank_proj(torch.cat(
            (
                mean_token,
                max_token,
                mean_token * max_token,
            ),
            dim=-1,
        ))

    def _recent_boundary_lag_repr(self, src_seq, dst_seq, src_time, dst_time):
        valid = (src_time > 0) & (dst_time > 0)
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        lag = (dst_time - src_time).unsqueeze(-1)
        lag_align = torch.exp(-lag.abs() / time_scale)
        after = (lag >= 0).float()
        slot_input = torch.cat(
            (
                self._pairwise_fusion_inputs(src_seq, dst_seq),
                lag_align,
                after,
            ),
            dim=-1,
        )
        slot_repr = self.boundary_lag_slot_proj(
            slot_input.reshape(-1, slot_input.size(-1))
        ).view(src_seq.size(0), src_seq.size(1), -1)
        slot_repr = slot_repr * valid.unsqueeze(-1).float()
        count = valid.float().sum(dim=1, keepdim=True)
        mean_repr = slot_repr.sum(dim=1) / count.clamp(min=1.0)
        max_repr = slot_repr.masked_fill(~valid.unsqueeze(-1), -1e9).max(dim=1).values
        max_repr = torch.where(
            count > 0,
            max_repr,
            torch.zeros_like(max_repr),
        )
        after_ratio = (
            after.squeeze(-1) * valid.float()
        ).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        return self.boundary_lag_proj(torch.cat(
            (
                mean_repr,
                max_repr,
                count / float(self.sequence_len),
                after_ratio,
            ),
            dim=-1,
        ))

    def _pairwise_fusion_inputs(self, left_repr, right_repr):
        if self.use_difference_fusion:
            return torch.cat(
                (
                    left_repr,
                    right_repr,
                    left_repr * right_repr,
                    torch.abs(left_repr - right_repr),
                ),
                dim=-1,
            )
        return torch.cat(
            (
                left_repr,
                right_repr,
                left_repr * right_repr,
            ),
            dim=-1,
        )

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
        pair_count = scatter(
            torch.ones_like(pair_inv, dtype=edge_repr.dtype),
            pair_inv,
            dim=0,
            dim_size=num_pairs,
            reduce='sum',
        ).unsqueeze(-1)

        pair_mean = scatter(edge_repr, pair_inv, dim=0, dim_size=num_pairs, reduce='mean')
        pair_max, _ = scatter_max(edge_repr, pair_inv, dim=0, dim_size=num_pairs)
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, torch.zeros_like(pair_max))
        edge_local_repr = edge_repr
        pair_seq_state = None
        fast_pair_seq_state = None
        slow_pair_seq_state = None
        if self.use_pair_internal_sequence and hasattr(batch[task], 'timestamps'):
            edge_timestamps = batch[task].timestamps.to(edge_repr.device).float().view(-1)
            pair_latest, _ = scatter_max(
                edge_timestamps, pair_inv, dim=0, dim_size=num_pairs
            )
            pair_latest = torch.where(
                torch.isfinite(pair_latest),
                pair_latest,
                torch.zeros_like(pair_latest),
            )
            if self.use_sequence_bridge_bank_window:
                base_time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
                fast_time_scale = base_time_scale * self.fast_time_scale_log.exp().clamp(min=0.1, max=10.0)
                slow_time_scale = base_time_scale * self.slow_time_scale_log.exp().clamp(min=0.25, max=20.0)
                fast_pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest, fast_time_scale
                )
                slow_pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest, slow_time_scale
                )
                fast_pair_seq_state = self.pair_edge_sequence_encoder(
                    fast_pair_sequence_bank
                )[1].squeeze(0)
                slow_pair_seq_state = self.pair_edge_sequence_encoder(
                    slow_pair_sequence_bank
                )[1].squeeze(0)
                pair_window_input = torch.cat(
                    (
                        fast_pair_seq_state,
                        slow_pair_seq_state,
                        fast_pair_seq_state * slow_pair_seq_state,
                    ),
                    dim=-1,
                )
                pair_window_gate = torch.sigmoid(self.pair_window_gate(pair_window_input))
                pair_seq_state = fast_pair_seq_state + (
                    torch.sigmoid(self.pair_window_alpha) *
                    pair_window_gate *
                    (slow_pair_seq_state - fast_pair_seq_state)
                )
            else:
                pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest
                )
                pair_seq_state = self.pair_edge_sequence_encoder(
                    pair_sequence_bank
                )[1].squeeze(0)
        if self.use_pair_internal_sequence:
            if pair_seq_state is None:
                pair_seq_state = torch.zeros_like(pair_mean)
            pair_repr = self.pair_proj(torch.cat(
                (pair_mean, pair_max, pair_max - pair_mean, pair_seq_state), dim=-1
            ))
        else:
            pair_repr = self.pair_proj(torch.cat(
                (pair_mean, pair_max, pair_max - pair_mean), dim=-1
            ))
        pair_context_repr = None
        pair_sequence_repr = None
        fast_sequence_repr = None
        slow_sequence_repr = None
        pair_terminal_role_repr = None
        pair_boundary_lag_repr = None
        pair_structure_mix = None
        pair_sequence_support = None
        pair_terminal_support = None
        pair_boundary_support = None

        if task[0] == task[2]:
            num_nodes = batch[task[0]].x.size(0)
            pair_src = torch.div(pair_keys, num_dst_nodes, rounding_mode='floor')
            pair_dst = torch.remainder(pair_keys, num_dst_nodes)
            zero_support = pair_repr.new_zeros((num_pairs, 1))
            src_out_cov = zero_support
            dst_in_cov = zero_support
            src_in_cov = zero_support
            dst_out_cov = zero_support
            forward_overlap = zero_support
            cycle_overlap = zero_support
            boundary_valid_ratio = zero_support
            boundary_after_ratio = zero_support
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
                pair_context_repr = self.context_proj(
                    self._pairwise_fusion_inputs(
                        predecessor_focus_bank[pair_src],
                        successor_focus_bank[pair_dst],
                    )
                )
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
                if self.use_sequence_bridge_bank_window:
                    base_time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
                    fast_time_scale = base_time_scale * self.fast_time_scale_log.exp().clamp(min=0.1, max=10.0)
                    slow_time_scale = base_time_scale * self.slow_time_scale_log.exp().clamp(min=0.25, max=20.0)
                    outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest, fast_time_scale
                    )
                    incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest, fast_time_scale
                    )
                    slow_outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest, slow_time_scale
                    )
                    slow_incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest, slow_time_scale
                    )
                    pair_outgoing_sequence_bank = outgoing_sequence_bank[pair_src]
                    pair_incoming_sequence_bank = incoming_sequence_bank[pair_dst]
                    pair_slow_outgoing_sequence_bank = slow_outgoing_sequence_bank[pair_src]
                    pair_slow_incoming_sequence_bank = slow_incoming_sequence_bank[pair_dst]
                    if self.use_target_sequence_select:
                        pair_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                        pair_slow_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_slow_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_slow_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_slow_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                    outgoing_state = self.outgoing_sequence_encoder(
                        pair_outgoing_sequence_bank
                    )[1].squeeze(0)
                    incoming_state = self.incoming_sequence_encoder(
                        pair_incoming_sequence_bank
                    )[1].squeeze(0)
                    slow_outgoing_state = self.outgoing_sequence_encoder(
                        pair_slow_outgoing_sequence_bank
                    )[1].squeeze(0)
                    slow_incoming_state = self.incoming_sequence_encoder(
                        pair_slow_incoming_sequence_bank
                    )[1].squeeze(0)
                    fast_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            outgoing_state,
                            incoming_state,
                        )
                    )
                    slow_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            slow_outgoing_state,
                            slow_incoming_state,
                        )
                    )
                    pair_sequence_repr = fast_sequence_repr
                else:
                    outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest
                    )
                    incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest
                    )
                    pair_outgoing_sequence_bank = outgoing_sequence_bank[pair_src]
                    pair_incoming_sequence_bank = incoming_sequence_bank[pair_dst]
                    if self.use_target_sequence_select:
                        pair_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                    outgoing_state = self.outgoing_sequence_encoder(
                        pair_outgoing_sequence_bank
                    )[1].squeeze(0)
                    incoming_state = self.incoming_sequence_encoder(
                        pair_incoming_sequence_bank
                    )[1].squeeze(0)
                    pair_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            outgoing_state,
                            incoming_state,
                        )
                    )
                if self.use_sequence_bridge_bank:
                    outgoing_partner_bank = self._build_recent_partner_bank(
                        pair_src, pair_dst, pair_timestamps, num_nodes
                    )
                    incoming_partner_bank = self._build_recent_partner_bank(
                        pair_dst, pair_src, pair_timestamps, num_nodes
                    )
                    src_out_cov = (
                        (outgoing_partner_bank[pair_src] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    dst_in_cov = (
                        (incoming_partner_bank[pair_dst] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    src_in_cov = (
                        (incoming_partner_bank[pair_src] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    dst_out_cov = (
                        (outgoing_partner_bank[pair_dst] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    forward_overlap = self._partner_overlap_ratio(
                        outgoing_partner_bank[pair_src],
                        incoming_partner_bank[pair_dst],
                    )
                    cycle_overlap = self._partner_overlap_ratio(
                        incoming_partner_bank[pair_src],
                        outgoing_partner_bank[pair_dst],
                    )
                    node_x = batch[task[0]].x
                    forward_bridge = self._recent_overlap_partner_repr(
                        outgoing_partner_bank[pair_src],
                        incoming_partner_bank[pair_dst],
                        node_x,
                    )
                    cycle_bridge = self._recent_overlap_partner_repr(
                        incoming_partner_bank[pair_src],
                        outgoing_partner_bank[pair_dst],
                        node_x,
                    )
                    bridge_context = self.bridge_bank_proj(
                        self._pairwise_fusion_inputs(
                            forward_bridge,
                            cycle_bridge,
                        )
                    )
                    if self.use_sequence_bridge_bank_window:
                        window_input = torch.cat(
                            (
                                fast_sequence_repr,
                                slow_sequence_repr,
                                bridge_context,
                            ),
                            dim=-1,
                        )
                        window_gate = torch.sigmoid(self.sequence_window_gate(window_input))
                        pair_sequence_repr = fast_sequence_repr + (
                            torch.sigmoid(self.sequence_window_alpha) *
                            window_gate *
                            (slow_sequence_repr - fast_sequence_repr)
                        )
                    bridge_input = torch.cat(
                        (
                            bridge_context,
                            pair_sequence_repr,
                            bridge_context * pair_sequence_repr,
                        ),
                        dim=-1,
                    )
                    bridge_gate = torch.sigmoid(self.bridge_bank_gate(bridge_input))
                    pair_sequence_repr = pair_sequence_repr + (
                        torch.sigmoid(self.bridge_bank_alpha) *
                        bridge_gate *
                        self.bridge_bank_update(bridge_input)
                    )
                    if self.use_terminal_role_flow:
                        src_incoming_sequence_bank = incoming_sequence_bank[pair_src]
                        dst_outgoing_sequence_bank = outgoing_sequence_bank[pair_dst]
                        if self.use_target_sequence_select:
                            src_incoming_sequence_bank = self._filter_sequence_bank(
                                src_incoming_sequence_bank,
                                pair_repr,
                                self.incoming_sequence_select_score,
                            )
                            dst_outgoing_sequence_bank = self._filter_sequence_bank(
                                dst_outgoing_sequence_bank,
                                pair_repr,
                                self.outgoing_sequence_select_score,
                            )
                        src_incoming_state = self.incoming_sequence_encoder(
                            src_incoming_sequence_bank
                        )[1].squeeze(0)
                        dst_outgoing_state = self.outgoing_sequence_encoder(
                            dst_outgoing_sequence_bank
                        )[1].squeeze(0)
                        src_role_repr = self.terminal_role_proj(
                            self._pairwise_fusion_inputs(
                                outgoing_state,
                                src_incoming_state,
                            )
                        )
                        dst_role_repr = self.terminal_role_proj(
                            self._pairwise_fusion_inputs(
                                incoming_state,
                                dst_outgoing_state,
                            )
                        )
                        src_out_count = (
                            outgoing_partner_bank[pair_src] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        src_in_count = (
                            incoming_partner_bank[pair_src] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        dst_out_count = (
                            outgoing_partner_bank[pair_dst] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        dst_in_count = (
                            incoming_partner_bank[pair_dst] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        role_stats = torch.cat(
                            (
                                src_out_count,
                                src_in_count,
                                dst_out_count,
                                dst_in_count,
                                src_out_count - src_in_count,
                                dst_in_count - dst_out_count,
                            ),
                            dim=-1,
                        )
                        pair_terminal_role_repr = self.terminal_flow_proj(
                            torch.cat(
                                (
                                    self._pairwise_fusion_inputs(
                                        src_role_repr,
                                        dst_role_repr,
                                    ),
                                    role_stats,
                                ),
                                dim=-1,
                            )
                        )
                        if self.use_boundary_lag_flow:
                            outgoing_time_bank = self._build_recent_timestamp_bank(
                                pair_src, pair_timestamps, num_nodes
                            )
                            incoming_time_bank = self._build_recent_timestamp_bank(
                                pair_dst, pair_timestamps, num_nodes
                            )
                            pair_boundary_lag_repr = self._recent_boundary_lag_repr(
                                src_incoming_sequence_bank,
                                dst_outgoing_sequence_bank,
                                incoming_time_bank[pair_src],
                                outgoing_time_bank[pair_dst],
                            )
                            boundary_valid_ratio, boundary_after_ratio, _ = (
                                self._boundary_support_features(
                                    incoming_time_bank[pair_src],
                                    outgoing_time_bank[pair_dst],
                                )
                            )
                    if self.use_sequence_bridge_motif_lite:
                        outgoing_time_bank = self._build_recent_timestamp_bank(
                            pair_src, pair_timestamps, num_nodes
                        )
                        incoming_time_bank = self._build_recent_timestamp_bank(
                            pair_dst, pair_timestamps, num_nodes
                        )
                        forward_leg = self._recent_overlap_leg_repr(
                            outgoing_partner_bank[pair_src],
                            incoming_partner_bank[pair_dst],
                            outgoing_sequence_bank[pair_src],
                            incoming_sequence_bank[pair_dst],
                            outgoing_time_bank[pair_src],
                            incoming_time_bank[pair_dst],
                        )
                        cycle_leg = self._recent_overlap_leg_repr(
                            incoming_partner_bank[pair_src],
                            outgoing_partner_bank[pair_dst],
                            incoming_sequence_bank[pair_src],
                            outgoing_sequence_bank[pair_dst],
                            incoming_time_bank[pair_src],
                            outgoing_time_bank[pair_dst],
                        )
                        leg_input = torch.cat(
                            (
                                forward_leg,
                                cycle_leg,
                                forward_leg * cycle_leg,
                            ),
                            dim=-1,
                        )
                        leg_gate = torch.sigmoid(self.bridge_leg_bank_gate(leg_input))
                        pair_sequence_repr = pair_sequence_repr + (
                            torch.sigmoid(self.bridge_leg_bank_alpha) *
                            leg_gate *
                            self.bridge_leg_bank_update(leg_input)
                        )
            if self.use_support_conditioned_mixture:
                pair_fill = torch.clamp(
                    pair_count / float(self.pair_sequence_len),
                    min=0.0,
                    max=1.0,
                )
                pair_log_count = torch.log1p(pair_count) / math.log(33.0)
                fast_slow_align = self._cosine_feature(
                    fast_sequence_repr, slow_sequence_repr
                )
                seq_pair_align = self._cosine_feature(pair_sequence_repr, pair_repr)
                context_seq_align = self._cosine_feature(
                    pair_context_repr, pair_sequence_repr
                )
                role_seq_align = self._cosine_feature(
                    pair_terminal_role_repr, pair_sequence_repr
                )
                if fast_slow_align is None:
                    fast_slow_align = zero_support
                if seq_pair_align is None:
                    seq_pair_align = zero_support
                if context_seq_align is None:
                    context_seq_align = zero_support
                if role_seq_align is None:
                    role_seq_align = zero_support
                pair_support_features = torch.cat(
                    (
                        pair_fill,
                        pair_log_count,
                        src_out_cov,
                        dst_in_cov,
                        src_in_cov,
                        dst_out_cov,
                        forward_overlap,
                        cycle_overlap,
                        boundary_valid_ratio,
                        boundary_after_ratio,
                        fast_slow_align,
                        seq_pair_align,
                        context_seq_align,
                        role_seq_align,
                    ),
                    dim=-1,
                )
                pair_structure_mix = torch.sigmoid(
                    self.structure_mix_bias +
                    self.structure_mix_gate(torch.cat((pair_repr, pair_support_features), dim=-1))
                )
                pair_sequence_support = torch.sigmoid(
                    self.sequence_support_bias +
                    self.sequence_support_gate(
                        torch.cat(
                            (
                                pair_sequence_repr
                                if pair_sequence_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )
                pair_terminal_support = torch.sigmoid(
                    self.terminal_support_bias +
                    self.terminal_support_gate(
                        torch.cat(
                            (
                                pair_terminal_role_repr
                                if pair_terminal_role_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )
                pair_boundary_support = torch.sigmoid(
                    self.boundary_support_bias +
                    self.boundary_support_gate(
                        torch.cat(
                            (
                                pair_boundary_lag_repr
                                if pair_boundary_lag_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )

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
            if self.use_support_conditioned_mixture and pair_structure_mix is not None:
                context_logits = pair_structure_mix[pair_inv][mask] * context_logits
            pred = pred + torch.sigmoid(self.context_residual_alpha) * context_logits
        if self.use_sequence_context_residual:
            if pair_sequence_repr is None:
                pair_sequence_repr = torch.zeros_like(pair_repr)
            sequence_logits = self.sequence_head(pair_sequence_repr[pair_inv][mask])
            if self.use_support_conditioned_mixture and pair_sequence_support is not None:
                sequence_logits = pair_sequence_support[pair_inv][mask] * sequence_logits
            pred = pred + torch.sigmoid(self.sequence_residual_alpha) * sequence_logits
        if self.use_terminal_role_flow:
            if pair_terminal_role_repr is None:
                pair_terminal_role_repr = torch.zeros_like(pair_repr)
            terminal_role_logits = self.terminal_flow_head(
                pair_terminal_role_repr[pair_inv][mask]
            )
            if self.use_support_conditioned_mixture and pair_terminal_support is not None:
                terminal_role_logits = (
                    pair_terminal_support[pair_inv][mask] * terminal_role_logits
                )
            pred = pred + (
                torch.sigmoid(self.terminal_flow_residual_alpha) *
                terminal_role_logits
            )
        if self.use_boundary_lag_flow:
            if pair_boundary_lag_repr is None:
                pair_boundary_lag_repr = torch.zeros_like(pair_repr)
            boundary_lag_logits = self.boundary_lag_head(
                pair_boundary_lag_repr[pair_inv][mask]
            )
            if self.use_support_conditioned_mixture and pair_boundary_support is not None:
                boundary_lag_logits = (
                    pair_boundary_support[pair_inv][mask] * boundary_lag_logits
                )
            pred = pred + (
                torch.sigmoid(self.boundary_lag_residual_alpha) *
                boundary_lag_logits
            )
        if self.use_support_conditioned_mixture and pair_structure_mix is not None:
            fallback_pred = self.edge_fallback_head(edge_local_repr[mask])
            pred = fallback_pred + pair_structure_mix[pair_inv][mask] * (
                pred - fallback_pred
            )
        return pred, batch[task].y[mask]

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        mask = self._edge_mask(batch)

        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index

        return torch.cat((batch[task[0]].x[edge_index[0, mask]],
                          batch[task[2]].x[edge_index[1, mask]],
                          batch[task].edge_attr[mask]), dim=-1), \
               batch[task].y[mask]

    def forward(self, batch):
        if self.use_pair_chain_head:
            return self._pair_chain_head(batch)
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)

        return pred, label
