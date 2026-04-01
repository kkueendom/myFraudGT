import math, time
import torch
import torch_sparse
import numpy as np
from torch_scatter import scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import fraudGT.graphgym.register as register
from fraudGT.graphgym.config import cfg
from torch_geometric.data import HeteroData
from torch_geometric.nn.inits import glorot, zeros, ones, reset
from torch_geometric.nn import (Linear, MLP, HeteroConv, GraphConv, SAGEConv, GINConv, GINEConv, \
                                GATConv)
from torch_geometric.utils import softmax as pyg_softmax
from fraudGT.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats


class GTLayer(nn.Module):
    r"""Graph Transformer layer

    """
    def __init__(self, dim_in, dim_h, dim_out, metadata, local_gnn_type, global_model_type, index, num_heads=1,
                 layer_norm=False, batch_norm=False, return_attention=False, **kwargs):
        super(GTLayer, self).__init__()

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.index = index
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = register.act_dict[cfg.gt.act]
        self.metadata = metadata
        self.return_attention = return_attention
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.kHop = cfg.gt.hops
        self.bias = Parameter(torch.Tensor(self.kHop))
        self.attn_bi = Parameter(torch.empty(self.num_heads, self.kHop))
        self.temporal_bias_enabled = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.temporal_bias != 'none'
        )
        self.temporal_gate_enabled = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.temporal_gate
        )
        self.edge_writeback_enabled = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback != 'none'
        )
        self.directional_meanmax_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmax'
        )
        self.directional_meanspike_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanspike'
        )
        self.directional_meantail_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meantail'
        )
        self.directional_meanmax_scaled_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmax_scaled'
        )
        self.directional_meansoftmax_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meansoftmax'
        )
        self.directional_meantopk_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meantopk'
        )
        self.directional_meanwinner_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanwinner'
        )
        self.directional_meanmaxwinnermix_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnermix'
        )
        self.directional_meanmaxwinnerplus_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnerplus'
        )
        self.directional_meanmaxwinnerproj_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnerproj'
        )
        self.directional_meanmaxwinnerprojline_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnerprojline'
        )
        self.directional_meanmaxtopkprojpluswinner_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxtopkprojpluswinner'
        )
        self.directional_meanmaxwinnerdecomp_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnerdecomp'
        )
        self.directional_meanmaxwinnercohclip_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnercohclip'
        )
        self.directional_meanmaxwinnersoftclip_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnersoftclip'
        )
        self.directional_meanmaxwinnerresid_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnerresid'
        )
        self.directional_meanmaxwinnergap_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxwinnergap'
        )
        self.directional_meanmaxmix_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxmix'
        )
        self.directional_meanmaxadd_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxadd'
        )
        self.directional_meanmaxcount_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxcount'
        )
        self.directional_meanmaxgap_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxgap'
        )
        self.directional_meanmaxspikeresid_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxspikeresid'
        )
        self.directional_meanmaxplusspike_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxplusspike'
        )
        self.directional_meanmaxsoftclip_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxsoftclip'
        )
        self.directional_meanmaxsoftmix_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmaxsoftmix'
        )
        self.directional_meanmax_dualgate_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanmax_dualgate'
        )
        self.directional_meanspike_dualgate_writeback = (
            global_model_type == 'SparseNodeTransformer' and
            cfg.gt.edge_writeback == 'dir_meanspike_dualgate'
        )
        self.directional_dualgate_writeback = (
            self.directional_meanmax_dualgate_writeback or
            self.directional_meanspike_dualgate_writeback
        )

        # Residual connection
        self.skip_local = torch.nn.ParameterDict()
        self.skip_global = torch.nn.ParameterDict()
        for node_type in metadata[0]:
            self.skip_local[node_type] = Parameter(torch.Tensor(1))
            self.skip_global[node_type] = Parameter(torch.Tensor(1))


        # Global Attention
        if global_model_type == 'None':
            self.attn = None
        elif global_model_type == 'TorchTransformer':
            self.attn = torch.nn.MultiheadAttention(
                        dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
            # self.attn = torch.nn.ModuleDict()
            # for edge_type in metadata[1]:
            #     edge_type = '__'.join(edge_type)
            #     self.attn[edge_type] = torch.nn.MultiheadAttention(
            #             dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
        elif global_model_type == 'SparseNodeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.e_lin = torch.nn.ModuleDict()
            self.g_lin = torch.nn.ModuleDict()
            self.oe_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                # Different node type have a different projection matrix
                self.k_lin[node_type] = Linear(dim_in, dim_h)
                self.q_lin[node_type] = Linear(dim_in, dim_h)
                self.v_lin[node_type] = Linear(dim_in, dim_h)
                self.o_lin[node_type] = Linear(dim_h, dim_out)
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                self.e_lin[edge_type] = Linear(dim_in, dim_h)
                self.g_lin[edge_type] = Linear(dim_h, dim_out)
                self.oe_lin[edge_type] = Linear(dim_h, dim_out)
            H, D = self.num_heads, self.dim_h // self.num_heads
            if cfg.gt.edge_weight:
                self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                nn.init.xavier_uniform_(self.edge_weights)
                nn.init.xavier_uniform_(self.msg_weights)
            if self.temporal_bias_enabled:
                self.temporal_alpha = nn.Parameter(
                    torch.full((self.num_heads,), cfg.gt.temporal_bias_init)
                )
            if self.temporal_gate_enabled:
                self.temporal_gate_alpha = nn.Parameter(
                    torch.full((self.num_heads,), cfg.gt.temporal_gate_init)
                )
            if self.edge_writeback_enabled:
                self.writeback_update = torch.nn.ModuleDict()
                self.writeback_gate = torch.nn.ModuleDict()
                self.writeback_mean_update = torch.nn.ModuleDict()
                self.writeback_anomaly_update = torch.nn.ModuleDict()
                self.writeback_anomaly_gate = torch.nn.ModuleDict()
                self.writeback_spike_update = torch.nn.ModuleDict()
                self.writeback_winner_update = torch.nn.ModuleDict()
                writeback_context_dim = dim_out * (
                    4 if (
                        self.directional_meanmax_writeback or
                        self.directional_meanspike_writeback or
                        self.directional_meantail_writeback or
                        self.directional_meanmax_scaled_writeback or
                        self.directional_meansoftmax_writeback or
                        self.directional_meantopk_writeback or
                        self.directional_meanwinner_writeback or
                        self.directional_meanmaxwinnermix_writeback or
                        self.directional_meanmaxwinnerplus_writeback or
                        self.directional_meanmaxwinnerproj_writeback or
                        self.directional_meanmaxwinnerprojline_writeback or
                        self.directional_meanmaxtopkprojpluswinner_writeback or
                        self.directional_meanmaxwinnerdecomp_writeback or
                        self.directional_meanmaxwinnercohclip_writeback or
                        self.directional_meanmaxwinnersoftclip_writeback or
                        self.directional_meanmaxwinnerresid_writeback or
                        self.directional_meanmaxwinnergap_writeback or
                        self.directional_meanmaxmix_writeback or
                        self.directional_meanmaxadd_writeback or
                        self.directional_meanmaxcount_writeback or
                        self.directional_meanmaxgap_writeback or
                        self.directional_meanmaxspikeresid_writeback or
                        self.directional_meanmaxplusspike_writeback or
                        self.directional_meanmaxsoftclip_writeback or
                        self.directional_meanmaxsoftmix_writeback or
                        self.directional_dualgate_writeback
                    ) else 2
                )
                if self.directional_meanmax_scaled_writeback:
                    self.writeback_anomaly_scale = nn.Parameter(
                        torch.full((2,), math.log(0.25 / 0.75))
                    )
                if self.directional_meanmaxwinnermix_writeback:
                    self.writeback_winner_mix = nn.Parameter(
                        torch.full((2,), math.log(0.75 / 0.25))
                    )
                if self.directional_meanmaxwinnerplus_writeback:
                    self.writeback_winner_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                if self.directional_meanmaxwinnerproj_writeback:
                    self.writeback_winner_proj_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                if self.directional_meanmaxwinnerprojline_writeback:
                    self.writeback_winner_proj_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_line_prev_add = nn.Parameter(
                        torch.full((1,), math.log(0.1 / 0.9))
                    )
                    self.writeback_line_next_add = nn.Parameter(
                        torch.full((1,), math.log(0.1 / 0.9))
                    )
                if self.directional_meanmaxtopkprojpluswinner_writeback:
                    self.writeback_focus_proj_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_resid_add = nn.Parameter(
                        torch.full((2,), math.log(0.1 / 0.9))
                    )
                if self.directional_meanmaxwinnerdecomp_writeback:
                    self.writeback_winner_spike_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_resid_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                if self.directional_meanmaxwinnercohclip_writeback:
                    self.writeback_winner_spike_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_resid_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_coh_tau = nn.Parameter(
                        torch.full((2,), 4.0)
                    )
                if self.directional_meanmaxwinnersoftclip_writeback:
                    self.writeback_winner_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_tau = nn.Parameter(
                        torch.full((2,), 2.0)
                    )
                if self.directional_meanmaxwinnerresid_writeback:
                    self.writeback_winner_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                    self.writeback_winner_residual = nn.Parameter(
                        torch.full((1,), math.log(0.1 / 0.9))
                    )
                if self.directional_meanmaxwinnergap_writeback:
                    self.writeback_winner_gap_scale = nn.Parameter(
                        torch.full((2,), 3.0)
                    )
                    self.writeback_winner_gap_bias = nn.Parameter(
                        torch.full((2,), -1.5)
                    )
                if self.directional_meanmaxmix_writeback:
                    self.writeback_anomaly_mix = nn.Parameter(
                        torch.full((2,), math.log(0.75 / 0.25))
                    )
                if self.directional_meanmaxadd_writeback:
                    self.writeback_anomaly_add = nn.Parameter(
                        torch.full((1,), math.log(0.35 / 0.65))
                    )
                if self.directional_meanmaxcount_writeback:
                    self.writeback_count_scale = nn.Parameter(
                        torch.full((2,), 1.5)
                    )
                    self.writeback_count_bias = nn.Parameter(
                        torch.full((2,), -2.0)
                    )
                if self.directional_meanmaxgap_writeback:
                    self.writeback_gap_scale = nn.Parameter(
                        torch.full((2,), 3.0)
                    )
                    self.writeback_gap_bias = nn.Parameter(
                        torch.full((2,), -1.5)
                    )
                if self.directional_meanmaxspikeresid_writeback:
                    self.writeback_spike_residual = nn.Parameter(
                        torch.full((1,), math.log(0.1 / 0.9))
                    )
                if self.directional_meanmaxplusspike_writeback:
                    self.writeback_spike_add = nn.Parameter(
                        torch.full((2,), math.log(0.15 / 0.85))
                    )
                if self.directional_meanmaxsoftclip_writeback:
                    self.writeback_softclip_tau = nn.Parameter(
                        torch.full((2,), 1.0)
                    )
                if self.directional_meanmaxsoftmix_writeback:
                    self.writeback_softmix_tau = nn.Parameter(
                        torch.full((2,), 2.0)
                    )
                    self.writeback_softmix_alpha = nn.Parameter(
                        torch.full((2,), math.log(0.75 / 0.25))
                    )
                for node_type in metadata[0]:
                    if (
                        self.directional_meanmaxadd_writeback or
                        self.directional_dualgate_writeback
                    ):
                        self.writeback_mean_update[node_type] = Linear(
                            dim_out * 2, dim_out
                        )
                        self.writeback_anomaly_update[node_type] = Linear(
                            dim_out * 2, dim_out
                        )
                    if self.directional_dualgate_writeback:
                        self.writeback_anomaly_gate[node_type] = Linear(
                            dim_out * 3, dim_out
                        )
                        nn.init.constant_(
                            self.writeback_anomaly_gate[node_type].bias, -2.0
                        )
                    else:
                        self.writeback_update[node_type] = Linear(
                            writeback_context_dim, dim_out
                        )
                    if self.directional_meanmaxspikeresid_writeback:
                        self.writeback_spike_update[node_type] = Linear(
                            dim_out * 2, dim_out
                        )
                        nn.init.zeros_(
                            self.writeback_spike_update[node_type].weight
                        )
                        nn.init.zeros_(
                            self.writeback_spike_update[node_type].bias
                        )
                    if self.directional_meanmaxwinnerresid_writeback:
                        self.writeback_winner_update[node_type] = Linear(
                            dim_out * 2, dim_out
                        )
                        nn.init.zeros_(
                            self.writeback_winner_update[node_type].weight
                        )
                        nn.init.zeros_(
                            self.writeback_winner_update[node_type].bias
                        )
                    self.writeback_gate[node_type] = Linear(dim_out * 2, dim_out)
        elif global_model_type == 'SparseEdgeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.e_lin = torch.nn.ModuleDict()
            self.g_lin = torch.nn.ModuleDict()
            self.oe_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                # Different edge type have a different projection matrix
                self.k_lin[edge_type] = Linear(dim_in, dim_h)
                self.q_lin[edge_type] = Linear(dim_in, dim_h)
                self.v_lin[edge_type] = Linear(dim_in, dim_h)
                self.e_lin[edge_type] = Linear(dim_in, dim_h)
                self.g_lin[edge_type] = Linear(dim_h, dim_out)
                self.oe_lin[edge_type] = Linear(dim_h, dim_out)
                self.o_lin[edge_type] = Linear(dim_h, dim_out)
            H, D = self.num_heads, self.dim_h // self.num_heads
            if cfg.gt.edge_weight:
                self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                nn.init.xavier_uniform_(self.edge_weights)
                nn.init.xavier_uniform_(self.msg_weights)

        self.norm1_local = torch.nn.ModuleDict()
        self.norm1_global = torch.nn.ModuleDict()
        self.norm2_ffn = torch.nn.ModuleDict()
        self.project = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.project[node_type] = Linear(dim_h * 2, dim_h)
            if self.layer_norm:
                self.norm1_local[node_type] = nn.LayerNorm(dim_h)
                self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_local[node_type] = nn.BatchNorm1d(dim_h)
                self.norm1_global[node_type] = nn.BatchNorm1d(dim_h)
        self.norm1_edge_local = torch.nn.ModuleDict()
        self.norm1_edge_global = torch.nn.ModuleDict()
        self.norm2_edge_ffn = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            if self.layer_norm:
                self.norm1_edge_local[edge_type] = nn.LayerNorm(dim_h)
                self.norm1_edge_global[edge_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_edge_local[edge_type] = nn.BatchNorm1d(dim_h)
                self.norm1_edge_global[edge_type] = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(cfg.gnn.dropout)
        self.dropout_global = nn.Dropout(cfg.gt.dropout)
        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)
        self.writeback_dropout = nn.Dropout(cfg.gt.edge_writeback_dropout)

        # if cfg.gt.residual == 'Concat':
        #     dim_h *= 2
        for node_type in metadata[0]:
            # Different node type have a different projection matrix
            if self.layer_norm:
                self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm2_ffn[node_type] = nn.BatchNorm1d(dim_h)
        
        # Feed Forward block.
        if cfg.gt.ffn == 'Single':
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        elif cfg.gt.ffn == 'Type':
            self.ff_linear1_type = torch.nn.ModuleDict()
            self.ff_linear2_type = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                self.ff_linear1_type[node_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_type[node_type] = nn.Linear(dim_h * 2, dim_h)
            self.ff_linear1_edge_type = torch.nn.ModuleDict()
            self.ff_linear2_edge_type = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = "__".join(edge_type)
                self.ff_linear1_edge_type[edge_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_edge_type[edge_type] = nn.Linear(dim_h * 2, dim_h)
        
        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)
        self.reset_parameters()


    def reset_parameters(self):
        pass
        zeros(self.attn_bi)
        # ones(self.skip)

    def _collect_edge_timestamps(self, batch, edge_type_tensor, device):
        edge_timestamps = torch.zeros(edge_type_tensor.shape[0], device=device)
        for idx, edge_type in enumerate(batch.edge_types):
            if hasattr(batch[edge_type], 'timestamps'):
                mask = edge_type_tensor == idx
                edge_timestamps[mask] = batch[edge_type].timestamps.to(
                    device=device, dtype=torch.float32
                )
        return edge_timestamps

    def _compute_temporal_delta(self, dst_nodes, edge_timestamps, num_nodes):
        latest_timestamps, _ = scatter_max(
            edge_timestamps, dst_nodes, dim=0, dim_size=num_nodes
        )
        delta = (latest_timestamps[dst_nodes] - edge_timestamps).clamp(min=0)
        delta = torch.log1p(delta / max(float(cfg.gt.temporal_bias_scale), 1.0))
        if cfg.gt.temporal_bias_clamp > 0:
            delta = delta.clamp(max=cfg.gt.temporal_bias_clamp)
        return delta

    def _group_topk_mean(self, edge_values, group_nodes, group_scores, num_nodes):
        topk = max(int(cfg.gt.edge_writeback_topk), 1)
        selected_sum = torch.zeros(
            (num_nodes, edge_values.shape[-1]), device=edge_values.device
        )
        selected_count = torch.zeros(num_nodes, device=edge_values.device)
        working_scores = group_scores.clone()

        for _ in range(topk):
            max_scores, max_indices = scatter_max(
                working_scores, group_nodes, dim=0, dim_size=num_nodes
            )
            valid_groups = torch.isfinite(max_scores)
            if not valid_groups.any():
                break
            valid_nodes = valid_groups.nonzero(as_tuple=False).view(-1)
            selected_edges = max_indices[valid_nodes]
            valid_edges = (
                (selected_edges >= 0) &
                (selected_edges < edge_values.shape[0])
            )
            if not valid_edges.any():
                break
            valid_nodes = valid_nodes[valid_edges]
            selected_edges = selected_edges[valid_edges]
            selected_sum.index_add_(0, valid_nodes, edge_values[selected_edges])
            selected_count.index_add_(
                0, valid_nodes,
                torch.ones_like(valid_nodes, dtype=edge_values.dtype)
            )
            working_scores[selected_edges] = float('-inf')

        return selected_sum / selected_count.clamp(min=1.0).unsqueeze(-1)

    def _group_top1_select(self, edge_values, group_nodes, group_scores, num_nodes):
        max_scores, max_indices = scatter_max(
            group_scores, group_nodes, dim=0, dim_size=num_nodes
        )
        selected = torch.zeros(
            (num_nodes, edge_values.shape[-1]), device=edge_values.device
        )
        valid_groups = torch.isfinite(max_scores)
        if not valid_groups.any():
            return selected
        valid_nodes = valid_groups.nonzero(as_tuple=False).view(-1)
        selected_edges = max_indices[valid_nodes]
        valid_edges = (
            (selected_edges >= 0) &
            (selected_edges < edge_values.shape[0])
        )
        if not valid_edges.any():
            return selected
        valid_nodes = valid_nodes[valid_edges]
        selected_edges = selected_edges[valid_edges]
        selected[valid_nodes] = edge_values[selected_edges]
        return selected

    def _apply_edge_writeback(self, out, edge_state, src_nodes, dst_nodes,
                              node_type_tensor, batch, edge_weights=None):
        num_nodes = out.shape[0]
        incoming = torch.zeros((num_nodes, edge_state.shape[-1]), device=out.device)
        outgoing = torch.zeros_like(incoming)
        in_count = torch.zeros(num_nodes, device=out.device)
        out_count = torch.zeros_like(in_count)
        if edge_weights is None:
            edge_weights = torch.ones(src_nodes.shape[0], device=out.device)
        else:
            edge_weights = edge_weights.to(out.device)
        weighted_edge_state = edge_state * edge_weights.unsqueeze(-1)

        incoming.index_add_(0, dst_nodes, weighted_edge_state)
        outgoing.index_add_(0, src_nodes, weighted_edge_state)
        in_count.index_add_(0, dst_nodes, edge_weights)
        out_count.index_add_(0, src_nodes, edge_weights)

        incoming = incoming / in_count.clamp(min=1.0).unsqueeze(-1)
        outgoing = outgoing / out_count.clamp(min=1.0).unsqueeze(-1)
        if (
            self.directional_meanmax_writeback or
            self.directional_meanspike_writeback or
            self.directional_meantail_writeback or
            self.directional_meanmax_scaled_writeback or
            self.directional_meansoftmax_writeback or
            self.directional_meantopk_writeback or
            self.directional_meanwinner_writeback or
            self.directional_meanmaxwinnermix_writeback or
            self.directional_meanmaxwinnerplus_writeback or
            self.directional_meanmaxwinnerproj_writeback or
            self.directional_meanmaxwinnerprojline_writeback or
            self.directional_meanmaxtopkprojpluswinner_writeback or
            self.directional_meanmaxwinnerdecomp_writeback or
            self.directional_meanmaxwinnercohclip_writeback or
            self.directional_meanmaxwinnersoftclip_writeback or
            self.directional_meanmaxwinnerresid_writeback or
            self.directional_meanmaxwinnergap_writeback or
            self.directional_meanmaxmix_writeback or
            self.directional_meanmaxadd_writeback or
            self.directional_meanmaxcount_writeback or
            self.directional_meanmaxgap_writeback or
            self.directional_meanmaxspikeresid_writeback or
            self.directional_meanmaxplusspike_writeback or
            self.directional_meanmaxsoftclip_writeback or
            self.directional_meanmaxsoftmix_writeback or
            self.directional_dualgate_writeback
        ):
            if self.directional_meansoftmax_writeback:
                anomaly_scores = weighted_edge_state.norm(dim=-1)
                incoming_focus_weights = pyg_softmax(
                    anomaly_scores, dst_nodes, num_nodes=num_nodes
                )
                outgoing_focus_weights = pyg_softmax(
                    anomaly_scores, src_nodes, num_nodes=num_nodes
                )
                incoming_focus = torch.zeros_like(incoming)
                outgoing_focus = torch.zeros_like(outgoing)
                incoming_focus.index_add_(
                    0, dst_nodes,
                    weighted_edge_state * incoming_focus_weights.unsqueeze(-1)
                )
                outgoing_focus.index_add_(
                    0, src_nodes,
                    weighted_edge_state * outgoing_focus_weights.unsqueeze(-1)
                )
                edge_context = torch.cat(
                    (incoming, outgoing, incoming_focus, outgoing_focus), dim=-1
                )
            elif self.directional_meantopk_writeback:
                anomaly_scores = weighted_edge_state.norm(dim=-1)
                incoming_topk = self._group_topk_mean(
                    weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                )
                outgoing_topk = self._group_topk_mean(
                    weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                )
                edge_context = torch.cat(
                    (incoming, outgoing, incoming_topk, outgoing_topk), dim=-1
                )
            elif self.directional_meanwinner_writeback:
                anomaly_scores = weighted_edge_state.norm(dim=-1)
                incoming_winner = self._group_top1_select(
                    weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                )
                outgoing_winner = self._group_top1_select(
                    weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                )
                edge_context = torch.cat(
                    (incoming, outgoing, incoming_winner, outgoing_winner), dim=-1
                )
            else:
                incoming_max, _ = scatter_max(
                    edge_state, dst_nodes, dim=0, dim_size=num_nodes
                )
                outgoing_max, _ = scatter_max(
                    edge_state, src_nodes, dim=0, dim_size=num_nodes
                )
                incoming_max = torch.where(
                    torch.isfinite(incoming_max), incoming_max,
                    torch.zeros_like(incoming_max)
                )
                outgoing_max = torch.where(
                    torch.isfinite(outgoing_max), outgoing_max,
                    torch.zeros_like(outgoing_max)
                )
                if self.directional_meanmax_writeback:
                    edge_context = torch.cat(
                        (incoming, outgoing, incoming_max, outgoing_max), dim=-1
                    )
                elif self.directional_meanmaxwinnermix_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_mix = torch.sigmoid(self.writeback_winner_mix)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            winner_mix[0] * incoming_max +
                            (1.0 - winner_mix[0]) * incoming_winner,
                            winner_mix[1] * outgoing_max +
                            (1.0 - winner_mix[1]) * outgoing_winner,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnerplus_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_add = torch.sigmoid(self.writeback_winner_add)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + winner_add[0] * (incoming_winner - incoming),
                            outgoing_max + winner_add[1] * (outgoing_winner - outgoing),
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnerproj_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_proj_add = torch.sigmoid(self.writeback_winner_proj_add)
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    incoming_winner_residual = incoming_winner - incoming
                    outgoing_winner_residual = outgoing_winner - outgoing
                    incoming_proj_coeff = (
                        (incoming_winner_residual * incoming_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        incoming_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    outgoing_proj_coeff = (
                        (outgoing_winner_residual * outgoing_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        outgoing_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    incoming_proj = F.relu(incoming_proj_coeff) * incoming_spike
                    outgoing_proj = F.relu(outgoing_proj_coeff) * outgoing_spike
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + winner_proj_add[0] * incoming_proj,
                            outgoing_max + winner_proj_add[1] * outgoing_proj,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnerprojline_writeback:
                    line_prev_add = torch.sigmoid(self.writeback_line_prev_add)
                    line_next_add = torch.sigmoid(self.writeback_line_next_add)
                    line_edge_state = (
                        weighted_edge_state +
                        line_prev_add * incoming[src_nodes] +
                        line_next_add * outgoing[dst_nodes]
                    )
                    anomaly_scores = line_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        line_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        line_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_proj_add = torch.sigmoid(self.writeback_winner_proj_add)
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    incoming_winner_residual = incoming_winner - incoming
                    outgoing_winner_residual = outgoing_winner - outgoing
                    incoming_proj_coeff = (
                        (incoming_winner_residual * incoming_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        incoming_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    outgoing_proj_coeff = (
                        (outgoing_winner_residual * outgoing_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        outgoing_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    incoming_proj = F.relu(incoming_proj_coeff) * incoming_spike
                    outgoing_proj = F.relu(outgoing_proj_coeff) * outgoing_spike
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + winner_proj_add[0] * incoming_proj,
                            outgoing_max + winner_proj_add[1] * outgoing_proj,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxtopkprojpluswinner_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_focus = self._group_topk_mean(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_focus = self._group_topk_mean(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    focus_proj_add = torch.sigmoid(self.writeback_focus_proj_add)
                    winner_resid_add = torch.sigmoid(self.writeback_winner_resid_add)
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    incoming_focus_residual = incoming_focus - incoming
                    outgoing_focus_residual = outgoing_focus - outgoing
                    incoming_winner_residual = incoming_winner - incoming
                    outgoing_winner_residual = outgoing_winner - outgoing
                    incoming_focus_proj_coeff = (
                        (incoming_focus_residual * incoming_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        incoming_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    outgoing_focus_proj_coeff = (
                        (outgoing_focus_residual * outgoing_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        outgoing_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    incoming_winner_proj_coeff = (
                        (incoming_winner_residual * incoming_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        incoming_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    outgoing_winner_proj_coeff = (
                        (outgoing_winner_residual * outgoing_spike).sum(
                            dim=-1, keepdim=True
                        ) /
                        outgoing_spike.pow(2).sum(dim=-1, keepdim=True).clamp(
                            min=1e-6
                        )
                    )
                    incoming_focus_proj = (
                        F.relu(incoming_focus_proj_coeff) * incoming_spike
                    )
                    outgoing_focus_proj = (
                        F.relu(outgoing_focus_proj_coeff) * outgoing_spike
                    )
                    incoming_winner_proj = (
                        F.relu(incoming_winner_proj_coeff) * incoming_spike
                    )
                    outgoing_winner_proj = (
                        F.relu(outgoing_winner_proj_coeff) * outgoing_spike
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max +
                            focus_proj_add[0] * incoming_focus_proj +
                            winner_resid_add[0] * incoming_winner_proj,
                            outgoing_max +
                            focus_proj_add[1] * outgoing_focus_proj +
                            winner_resid_add[1] * outgoing_winner_proj,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnerdecomp_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    spike_add = torch.sigmoid(self.writeback_winner_spike_add)
                    resid_add = torch.sigmoid(self.writeback_winner_resid_add)
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    incoming_residual = incoming_winner - incoming_max
                    outgoing_residual = outgoing_winner - outgoing_max
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming +
                            (1.0 + spike_add[0]) * incoming_spike +
                            resid_add[0] * incoming_residual,
                            outgoing +
                            (1.0 + spike_add[1]) * outgoing_spike +
                            resid_add[1] * outgoing_residual,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnercohclip_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    spike_add = torch.sigmoid(self.writeback_winner_spike_add)
                    resid_add = torch.sigmoid(self.writeback_winner_resid_add)
                    winner_coh_tau = (
                        F.softplus(self.writeback_winner_coh_tau) + 1e-6
                    )
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    incoming_residual = incoming_winner - incoming_max
                    outgoing_residual = outgoing_winner - outgoing_max
                    incoming_cohclip = (
                        winner_coh_tau[0] *
                        torch.tanh(incoming_residual / winner_coh_tau[0])
                    )
                    outgoing_cohclip = (
                        winner_coh_tau[1] *
                        torch.tanh(outgoing_residual / winner_coh_tau[1])
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming +
                            (1.0 + spike_add[0]) * incoming_spike +
                            resid_add[0] * incoming_cohclip,
                            outgoing +
                            (1.0 + spike_add[1]) * outgoing_spike +
                            resid_add[1] * outgoing_cohclip,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnersoftclip_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_add = torch.sigmoid(self.writeback_winner_add)
                    winner_tau = F.softplus(self.writeback_winner_tau) + 1e-6
                    incoming_winner_residual = incoming_winner - incoming
                    outgoing_winner_residual = outgoing_winner - outgoing
                    incoming_winner_softclip = (
                        winner_tau[0] *
                        torch.tanh(incoming_winner_residual / winner_tau[0])
                    )
                    outgoing_winner_softclip = (
                        winner_tau[1] *
                        torch.tanh(outgoing_winner_residual / winner_tau[1])
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + winner_add[0] * incoming_winner_softclip,
                            outgoing_max + winner_add[1] * outgoing_winner_softclip,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnerresid_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_add = torch.sigmoid(self.writeback_winner_add)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + winner_add[0] * (incoming_winner - incoming),
                            outgoing_max + winner_add[1] * (outgoing_winner - outgoing),
                        ),
                        dim=-1
                    )
                    winner_context_raw = torch.cat(
                        (
                            incoming_winner - incoming_max,
                            outgoing_winner - outgoing_max,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxwinnergap_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_winner = self._group_top1_select(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_winner = self._group_top1_select(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    winner_gap_context = torch.stack(
                        (
                            torch.log1p((incoming_winner - incoming_max).norm(dim=-1)),
                            torch.log1p((outgoing_winner - outgoing_max).norm(dim=-1)),
                        ),
                        dim=-1
                    )
                    winner_gap_gate = torch.sigmoid(
                        winner_gap_context * self.writeback_winner_gap_scale +
                        self.writeback_winner_gap_bias
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max +
                            winner_gap_gate[:, 0].unsqueeze(-1) *
                            (incoming_winner - incoming),
                            outgoing_max +
                            winner_gap_gate[:, 1].unsqueeze(-1) *
                            (outgoing_winner - outgoing),
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxspikeresid_writeback:
                    edge_context = torch.cat(
                        (incoming, outgoing, incoming_max, outgoing_max), dim=-1
                    )
                    spike_context_raw = torch.cat(
                        (incoming_max - incoming, outgoing_max - outgoing), dim=-1
                    )
                elif self.directional_meanmaxplusspike_writeback:
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    spike_add = torch.sigmoid(self.writeback_spike_add)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming_max + spike_add[0] * incoming_spike,
                            outgoing_max + spike_add[1] * outgoing_spike,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxsoftclip_writeback:
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    softclip_tau = F.softplus(self.writeback_softclip_tau) + 1e-6
                    incoming_softclip = (
                        softclip_tau[0] *
                        torch.tanh(incoming_spike / softclip_tau[0])
                    )
                    outgoing_softclip = (
                        softclip_tau[1] *
                        torch.tanh(outgoing_spike / softclip_tau[1])
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming + incoming_softclip,
                            outgoing + outgoing_softclip,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxsoftmix_writeback:
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    softmix_tau = F.softplus(self.writeback_softmix_tau) + 1e-6
                    softmix_alpha = torch.sigmoid(self.writeback_softmix_alpha)
                    incoming_softclip = (
                        softmix_tau[0] *
                        torch.tanh(incoming_spike / softmix_tau[0])
                    )
                    outgoing_softclip = (
                        softmix_tau[1] *
                        torch.tanh(outgoing_spike / softmix_tau[1])
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            incoming + (
                                softmix_alpha[0] * incoming_spike +
                                (1.0 - softmix_alpha[0]) * incoming_softclip
                            ),
                            outgoing + (
                                softmix_alpha[1] * outgoing_spike +
                                (1.0 - softmix_alpha[1]) * outgoing_softclip
                            ),
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxcount_writeback:
                    count_context = torch.stack(
                        (
                            torch.log1p(in_count.clamp(min=0.0)),
                            torch.log1p(out_count.clamp(min=0.0)),
                        ),
                        dim=-1
                    )
                    count_gate = torch.sigmoid(
                        count_context * self.writeback_count_scale +
                        self.writeback_count_bias
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            count_gate[:, 0].unsqueeze(-1) * incoming_max,
                            count_gate[:, 1].unsqueeze(-1) * outgoing_max,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxgap_writeback:
                    gap_context = torch.stack(
                        (
                            torch.log1p((incoming_max - incoming).norm(dim=-1)),
                            torch.log1p((outgoing_max - outgoing).norm(dim=-1)),
                        ),
                        dim=-1
                    )
                    gap_gate = torch.sigmoid(
                        gap_context * self.writeback_gap_scale +
                        self.writeback_gap_bias
                    )
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            gap_gate[:, 0].unsqueeze(-1) * incoming_max,
                            gap_gate[:, 1].unsqueeze(-1) * outgoing_max,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxmix_writeback:
                    anomaly_scores = weighted_edge_state.norm(dim=-1)
                    incoming_topk = self._group_topk_mean(
                        weighted_edge_state, dst_nodes, anomaly_scores, num_nodes
                    )
                    outgoing_topk = self._group_topk_mean(
                        weighted_edge_state, src_nodes, anomaly_scores, num_nodes
                    )
                    anomaly_mix = torch.sigmoid(self.writeback_anomaly_mix)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            anomaly_mix[0] * incoming_max +
                            (1.0 - anomaly_mix[0]) * incoming_topk,
                            anomaly_mix[1] * outgoing_max +
                            (1.0 - anomaly_mix[1]) * outgoing_topk,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmaxadd_writeback:
                    mean_context_raw = torch.cat((incoming, outgoing), dim=-1)
                    anomaly_context_raw = torch.cat(
                        (incoming_max, outgoing_max), dim=-1
                    )
                elif self.directional_meanmax_scaled_writeback:
                    anomaly_scale = torch.sigmoid(self.writeback_anomaly_scale)
                    edge_context = torch.cat(
                        (
                            incoming,
                            outgoing,
                            anomaly_scale[0] * incoming_max,
                            anomaly_scale[1] * outgoing_max,
                        ),
                        dim=-1
                    )
                elif self.directional_meanmax_dualgate_writeback:
                    mean_context_raw = torch.cat((incoming, outgoing), dim=-1)
                    anomaly_context_raw = torch.cat(
                        (incoming_max, outgoing_max), dim=-1
                    )
                elif self.directional_meanspike_dualgate_writeback:
                    mean_context_raw = torch.cat((incoming, outgoing), dim=-1)
                    anomaly_context_raw = torch.cat(
                        (incoming_max - incoming, outgoing_max - outgoing), dim=-1
                    )
                elif self.directional_meanspike_writeback:
                    # Encode how much each direction deviates from its typical edge state.
                    incoming_spike = incoming_max - incoming
                    outgoing_spike = outgoing_max - outgoing
                    edge_context = torch.cat(
                        (incoming, outgoing, incoming_spike, outgoing_spike), dim=-1
                    )
                else:
                    incoming_tail = torch.zeros_like(incoming)
                    outgoing_tail = torch.zeros_like(outgoing)
                    incoming_tail_edges = F.relu(edge_state - incoming[dst_nodes])
                    outgoing_tail_edges = F.relu(edge_state - outgoing[src_nodes])
                    if edge_weights is not None:
                        incoming_tail_edges = (
                            incoming_tail_edges * edge_weights.unsqueeze(-1)
                        )
                        outgoing_tail_edges = (
                            outgoing_tail_edges * edge_weights.unsqueeze(-1)
                        )
                    incoming_tail.index_add_(0, dst_nodes, incoming_tail_edges)
                    outgoing_tail.index_add_(0, src_nodes, outgoing_tail_edges)
                    incoming_tail = incoming_tail / in_count.clamp(min=1.0).unsqueeze(-1)
                    outgoing_tail = outgoing_tail / out_count.clamp(min=1.0).unsqueeze(-1)
                    edge_context = torch.cat(
                        (incoming, outgoing, incoming_tail, outgoing_tail), dim=-1
                    )
        else:
            edge_context = torch.cat((incoming, outgoing), dim=-1)

        out_with_writeback = out.clone()
        for idx, node_type in enumerate(batch.node_types):
            mask = node_type_tensor == idx
            if not mask.any():
                continue
            node_out = out[mask]
            if self.directional_meanmaxspikeresid_writeback:
                node_context = self.activation(
                    self.writeback_update[node_type](edge_context[mask])
                )
                spike_residual = self.activation(
                    self.writeback_spike_update[node_type](
                        spike_context_raw[mask]
                    )
                )
                node_context = node_context + (
                    torch.sigmoid(self.writeback_spike_residual) * spike_residual
                )
            elif self.directional_meanmaxwinnerresid_writeback:
                node_context = self.activation(
                    self.writeback_update[node_type](edge_context[mask])
                )
                winner_residual = self.activation(
                    self.writeback_winner_update[node_type](
                        winner_context_raw[mask]
                    )
                )
                node_context = node_context + (
                    torch.sigmoid(self.writeback_winner_residual) *
                    winner_residual
                )
            elif self.directional_meanmaxadd_writeback:
                mean_context = self.activation(
                    self.writeback_mean_update[node_type](mean_context_raw[mask])
                )
                anomaly_context = self.activation(
                    self.writeback_anomaly_update[node_type](
                        anomaly_context_raw[mask]
                    )
                )
                node_context = mean_context + (
                    torch.sigmoid(self.writeback_anomaly_add) * anomaly_context
                )
            elif self.directional_dualgate_writeback:
                mean_context = self.activation(
                    self.writeback_mean_update[node_type](
                        mean_context_raw[mask]
                    )
                )
                anomaly_context = self.activation(
                    self.writeback_anomaly_update[node_type](
                        anomaly_context_raw[mask]
                    )
                )
                anomaly_gate = torch.sigmoid(
                    self.writeback_anomaly_gate[node_type](
                        torch.cat(
                            (node_out, mean_context, anomaly_context), dim=-1
                        )
                    )
                )
                node_context = mean_context + anomaly_gate * anomaly_context
            else:
                node_context = self.activation(
                    self.writeback_update[node_type](edge_context[mask])
                )
            node_context = self.writeback_dropout(node_context)
            node_gate = torch.sigmoid(
                self.writeback_gate[node_type](
                    torch.cat((node_out, node_context), dim=-1)
                )
            )
            out_with_writeback[mask] = node_out + node_gate * node_context
        return out_with_writeback


    def forward(self, batch):
        has_edge_attr = False
        if isinstance(batch, HeteroData):
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
            if sum(batch.num_edge_features.values()):
                edge_attr_dict = batch.collect('edge_attr')
                has_edge_attr = True
        else:
            h_dict = {'node_type': batch.x}
            edge_index_dict = {('node_type', 'edge_type', 'node_type'): batch.edge_index}
            if sum(batch.num_edge_features.values()):
                edge_attr_dict = {('node_type', 'edge_type', 'node_type'): batch.edge_attr}
                has_edge_attr = True
        h_in_dict = h_dict#.copy()
        if has_edge_attr:
            edge_attr_in_dict = edge_attr_dict.copy()

        h_out_dict_list = {node_type: [] for node_type in h_dict}
        runtime_stats_cuda.start_region("gt-layer")

        if self.global_model_type != 'None':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm1_global[node_type](h_dict[node_type])
                    for node_type in batch.node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
                        for edge_type in batch.edge_types
                    }
            

            h_attn_dict_list = {node_type: [] for node_type in h_dict}
            if self.global_model_type == 'TorchTransformer':
                D = self.dim_h

                homo_data = batch.to_homogeneous()
                h = homo_data.x
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
                
                L = h.shape[0]
                S = h.shape[0]
                q = h.view(1, -1, D)
                k = h.view(1, -1, D)
                v = h.view(1, -1, D)

                if cfg.gt.attn_mask in ['Edge', 'kHop']:
                    attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                    if cfg.gt.attn_mask == 'kHop':
                        with torch.no_grad():
                            ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                            edge_index_list = [edge_index]
                            edge_index_k = edge_index
                            for i in range(1, self.kHop):
                                # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                    edge_index, ones, 
                                                                    L, L, L, True)
                                edge_index_list.append(edge_index_k)
                        
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                    else:
                        # Avoid the nan from attention mask
                        attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                
                elif cfg.gt.attn_mask == 'Bias':
                    attn_mask = batch.attn_bi[self.index, :, :, :]
                else:
                    attn_mask = None

                h, A = self.attn(q, k, v,
                            attn_mask=attn_mask,
                            need_weights=True)
                            # average_attn_weights=False)

                # attn_weights = A.detach().cpu()
                h = h.view(1, -1, D)
                for idx, node_type in enumerate(batch.node_types):
                    out_type = h[:, node_type_tensor == idx, :]
                    h_attn_dict_list[node_type].append(out_type.squeeze())

            elif self.global_model_type == 'SparseNodeTransformer':
                # Test if Signed attention is beneficial
                # st = time.time()
                H, D = self.num_heads, self.dim_h // self.num_heads
                homo_data = batch.to_homogeneous()
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
                q = torch.empty((homo_data.num_nodes, self.dim_h), device=homo_data.x.device)
                k = torch.empty((homo_data.num_nodes, self.dim_h), device=homo_data.x.device)
                v = torch.empty((homo_data.num_nodes, self.dim_h), device=homo_data.x.device)
                edge_attr = torch.empty((homo_data.num_edges, self.dim_h), device=homo_data.x.device)
                edge_gate = torch.empty((homo_data.num_edges, self.dim_h), device=homo_data.x.device)
                for idx, node_type in enumerate(batch.node_types):
                    mask = node_type_tensor == idx
                    q[mask] = self.q_lin[node_type](h_dict[node_type])
                    k[mask] = self.k_lin[node_type](h_dict[node_type])
                    v[mask] = self.v_lin[node_type](h_dict[node_type])
                for idx, edge_type_tuple in enumerate(batch.edge_types):
                    edge_type = '__'.join(edge_type_tuple)
                    mask = edge_type_tensor == idx
                    edge_attr[mask] = self.e_lin[edge_type](edge_attr_dict[edge_type_tuple])
                    edge_gate[mask] = self.g_lin[edge_type](edge_attr_dict[edge_type_tuple])
                src_nodes, dst_nodes = edge_index
                num_edges = edge_index.shape[1]
                L = homo_data.x.shape[0]
                S = homo_data.x.shape[0]
                temporal_delta = None

                if has_edge_attr:
                    # src_nodes, dst_nodes = edge_index
                    # edge_attr = edge_attr_dict[edge_type_tuple]
                    # edge_attr = self.e_lin[edge_type](edge_attr).view(-1, H, D)
                    edge_attr = edge_attr.view(-1, H, D)
                    # edge_attr = self.e_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_attr), dim=-1)).view(-1, H, D)
                    edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)

                    # edge_gate = edge_attr_dict[edge_type_tuple]
                    # edge_gate = self.g_lin[edge_type](edge_gate).view(-1, H, D)
                    edge_gate = edge_gate.view(-1, H, D)
                    # edge_gate = self.g_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_gate), dim=-1)).view(-1, H, D)
                    edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

                q = q.view(-1, H, D)
                k = k.view(-1, H, D)
                v = v.view(-1, H, D)

                # transpose to get dimensions h * sl * d_model
                q = q.transpose(0,1)
                k = k.transpose(0,1)
                v = v.transpose(0,1)
                writeback_weights = None

                if cfg.gt.attn_mask in ['Edge', 'kHop']:
                    if cfg.gt.attn_mask in ['kHop']:
                        with torch.no_grad():
                            edge_index_list = [edge_index]
                            edge_index_k = torch.cat(edge_index_list, dim=1)

                            # ones = torch.ones(edge_index.shape[1], device=edge_index.device)
                            # edge_index_list = [edge_index]
                            # edge_index_k = edge_index
                            # for i in range(1, self.kHop):
                            #     # print(edge_index_k.shape, int(edge_index_k.max()), L)
                            #     edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                            #                                         edge_index, ones, 
                            #                                         L, L, L, True)
                            #     edge_index_list.append(edge_index_k)
                        
                        attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        src_nodes, dst_nodes = edge_index_k
                        num_edges = edge_index_k.shape[1]
                    else:
                        src_nodes, dst_nodes = edge_index
                        num_edges = edge_index.shape[1]
                    if (self.temporal_bias_enabled or self.temporal_gate_enabled) and cfg.gt.attn_mask == 'Edge':
                        edge_timestamps = self._collect_edge_timestamps(
                            batch, edge_type_tensor, q.device
                        )
                        temporal_delta = self._compute_temporal_delta(
                            dst_nodes, edge_timestamps, L
                        )
                    # Compute query and key for each edge
                    edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # num_heads * num_edges * d_k
                    edge_k = k[:, src_nodes, :]  # Keys for source nodes
                    edge_v = v[:, src_nodes, :]

                    if hasattr(self, 'edge_weights'):
                        edge_weight = self.edge_weights[edge_type_tensor]  # (num_edges, num_heads, d_k, d_k)

                        edge_weight = edge_weight.transpose(0, 1)  # Transpose for batch matrix multiplication: (num_heads, num_edges, d_k, d_k)
                        # edge_k = edge_k.transpose(0, 1)  # Transpose to (num_edges, num_heads, d_k)
                        edge_k = edge_k.unsqueeze(-1) # Add dimension for matrix multiplication (num_heads, num_edges, d_k, 1)

                        # print(edge_weight.shape, edge_k.shape)
                        edge_k = torch.matmul(edge_weight, edge_k)  # (num_heads, num_edges, d_k, 1)
                        edge_k = edge_k.squeeze(-1)  # Remove the extra dimension (num_heads, num_edges, d_k)
                    # edge_k = edge_k.transpose(0, 1)  # Transpose back (num_edges, num_heads, d_k)

                    # Apply weight matrix to keys
                    # edge_k = torch.einsum('ehij,hej->hei', edge_weight, edge_k)
                    # msg_weight = self.msg_weights[edge_type_tensor]
                    # edge_v = torch.einsum('ehij,hej->hei', msg_weight, edge_v)

                    # Compute attention scores
                    edge_scores = edge_q * edge_k
                    if has_edge_attr:
                        edge_scores = edge_scores + edge_attr
                        edge_v = edge_v * F.sigmoid(edge_gate)
                        edge_attr = edge_scores
                    if temporal_delta is not None and self.temporal_gate_enabled:
                        temporal_gate = torch.exp(
                            -F.softplus(self.temporal_gate_alpha).unsqueeze(-1) *
                            temporal_delta.unsqueeze(0)
                        )
                        edge_v = edge_v * temporal_gate.unsqueeze(-1)
                        if has_edge_attr:
                            edge_attr = edge_attr * temporal_gate.unsqueeze(-1)
                    
                    edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # num_heads * num_edges
                    if temporal_delta is not None and self.temporal_bias_enabled:
                        edge_scores = edge_scores - (
                            F.softplus(self.temporal_alpha).unsqueeze(-1) *
                            temporal_delta.unsqueeze(0)
                        )
                    edge_scores = torch.clamp(edge_scores, min=-5, max=5)
                    if cfg.gt.attn_mask in ['kHop']:
                        edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]

                    expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                    
                    # Step 2: Calculate max for each destination node per head using scatter_max
                    max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                    max_scores = max_scores.gather(1, expanded_dst_nodes)

                    # Step 3: Exponentiate scores and sum
                    exp_scores = torch.exp(edge_scores - max_scores)
                    sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                    sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
                    # sum_exp_scores.clamp_(min=1e-9)

                    # Step 4: Apply softmax
                    edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                    writeback_weights = None
                    if cfg.gt.edge_writeback == 'attn_mean':
                        writeback_weights = edge_scores.mean(dim=0)
                    edge_scores = edge_scores.unsqueeze(-1)
                    edge_scores = self.dropout_attn(edge_scores)

                    out = torch.zeros((H, L, D), device=q.device)
                    out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                    scores = F.softmax(scores, dim=-1)
                    scores = self.dropout_attn(scores)
                    
                    out = torch.matmul(scores, v)

                out = out.transpose(0,1).contiguous().view(-1, H * D)
                if has_edge_attr:
                    edge_state = edge_attr.transpose(0,1).contiguous().view(-1, H * D)
                    if self.edge_writeback_enabled:
                        out = self._apply_edge_writeback(
                            out, edge_state, src_nodes, dst_nodes,
                            node_type_tensor, batch,
                            edge_weights=writeback_weights
                        )

                for idx, node_type in enumerate(batch.node_types):
                    mask = node_type_tensor == idx
                    out_type = self.o_lin[node_type](out[mask, :])
                    h_attn_dict_list[node_type].append(out_type.squeeze())
                if has_edge_attr:
                    for idx, edge_type_tuple in enumerate(batch.edge_types):
                        edge_type = '__'.join(edge_type_tuple)
                        mask = edge_type_tensor == idx
                        out_type = self.oe_lin[edge_type](edge_state[mask, :])
                        edge_attr_dict[edge_type_tuple] = out_type

            h_attn_dict = {}
            for node_type in h_attn_dict_list:
                # h_attn_dict[node_type] = torch.zeros_like(h_in_dict[node_type])
                h_attn_dict[node_type] = torch.sum(torch.stack(h_attn_dict_list[node_type], dim=0), dim=0)
                h_attn_dict[node_type] = self.dropout_global(h_attn_dict[node_type])

            if cfg.gt.residual == 'Fixed':
                h_attn_dict = {
                    node_type: h_attn_dict[node_type] + h_in_dict[node_type]
                    for node_type in batch.node_types
                }

                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + edge_attr_in_dict[edge_type]
                        for edge_type in batch.edge_types
                    }
            elif cfg.gt.residual == 'Learn':
                alpha_dict = {
                    node_type: self.skip_global[node_type].sigmoid() for node_type in batch.node_types
                }
                h_attn_dict = {
                    node_type: alpha_dict[node_type] * h_attn_dict[node_type] + \
                        (1 - alpha_dict[node_type]) * h_in_dict[node_type]
                    for node_type in batch.node_types
                }
            elif cfg.gt.residual != 'none':
                raise ValueError(
                    f"Invalid attention residual option {cfg.gt.residual}"
                )
            
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_attn_dict = {
            #         node_type: self.norm1_global[node_type](h_attn_dict[node_type])
            #         for node_type in batch.node_types
            #     }
            #     if has_edge_attr:
            #         edge_attr_dict = {
            #             edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
            #             for edge_type in batch.edge_types
            #         }

            
            # Concat output
            h_out_dict_list = {
                node_type: h_out_dict_list[node_type] + [h_attn_dict[node_type]] for node_type in batch.node_types
            }

        # Combine global information
        h_dict = {
            node_type: sum(h_out_dict_list[node_type]) for node_type in batch.node_types
        }
        if cfg.gt.ffn != 'none':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm2_ffn[node_type](h_dict[node_type])
                    for node_type in batch.node_types
                }
            
            if cfg.gt.ffn == 'Type':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block_type(h_dict[node_type], node_type)
                    for node_type in batch.node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + self._ff_block_edge_type(edge_attr_dict[edge_type], edge_type)
                        for edge_type in batch.edge_types
                    }
            elif cfg.gt.ffn == 'Single':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block(h_dict[node_type])
                    for node_type in batch.node_types
                }
            else:
                raise ValueError(
                    f"Invalid GT FFN option {cfg.gt.ffn}"
                )
                
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_dict = {
            #         node_type: self.norm2_ffn[node_type](h_dict[node_type])
            #         for node_type in batch.node_types
            #     }
        
        if cfg.gt.residual == 'Concat':
            h_dict = {
                node_type: torch.cat((h_in_dict[node_type], h_dict[node_type]), dim=1)
                for node_type in batch.node_types
            }

        runtime_stats_cuda.end_region("gt-layer")

        if isinstance(batch, HeteroData):
            for node_type in batch.node_types:
                batch[node_type].x = h_dict[node_type]
            if has_edge_attr:
                for edge_type in batch.edge_types:
                    batch[edge_type].edge_attr = edge_attr_dict[edge_type]
        else:
            batch.x = h_dict['node_type']

        if self.return_attention:
            return batch, saved_scores
        return batch
    
    def _ff_block_type(self, x, node_type):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_type[node_type](x)))
        return self.ff_dropout2(self.ff_linear2_type[node_type](x))
    
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    def _ff_block_edge_type(self, x, edge_type):
        """Feed Forward block.
        """
        edge_type = "__".join(edge_type)
        x = self.ff_dropout1(self.activation(self.ff_linear1_edge_type[edge_type](x)))
        return self.ff_dropout2(self.ff_linear2_edge_type[edge_type](x))

    # def __repr__(self):
    #     return '{}({}, {})'.format(self.__class__.__name__, self.dim_h,
    #                                self.dim_h)
