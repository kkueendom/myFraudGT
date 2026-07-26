import math

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

from fraudGT.cdvt.event_graph import CausalEventGraphBatch


class TemporalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        frequencies = torch.exp(torch.linspace(
            math.log(1.0), math.log(1e-4), hidden_dim // 2))
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(2 * (hidden_dim // 2), hidden_dim)

    def forward(self, target_delta):
        scaled = torch.log1p(target_delta.float().clamp_min(0)).unsqueeze(-1)
        angles = scaled * self.frequencies.unsqueeze(0)
        return self.projection(torch.cat((angles.sin(), angles.cos()), -1))


class RelationAwareTemporalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        edge_dim,
        num_relations,
        dropout,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_key = nn.Embedding(num_relations, hidden_dim)
        self.relation_value = nn.Embedding(num_relations, hidden_dim)
        self.edge_key = nn.Linear(edge_dim, hidden_dim)
        self.edge_value = nn.Linear(edge_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, node_states, edge_index, edge_attr, edge_relation):
        if edge_index.numel() == 0:
            attended = node_states.new_zeros(node_states.shape)
            attention = node_states.new_empty((0, self.num_heads))
        else:
            source, destination = edge_index
            query = self.query(node_states[destination]).view(
                -1, self.num_heads, self.head_dim)
            key = (
                self.key(node_states[source])
                + self.relation_key(edge_relation)
                + self.edge_key(edge_attr)
            ).view(-1, self.num_heads, self.head_dim)
            value = (
                self.value(node_states[source])
                + self.relation_value(edge_relation)
                + self.edge_value(edge_attr)
            ).view(-1, self.num_heads, self.head_dim)
            scores = (query * key).sum(-1) / math.sqrt(self.head_dim)
            attention = pyg_softmax(
                scores, destination, num_nodes=node_states.size(0))
            messages = value * self.attention_dropout(
                attention).unsqueeze(-1)
            attended = scatter(
                messages,
                destination,
                dim=0,
                dim_size=node_states.size(0),
                reduce="sum",
            ).reshape(-1, self.hidden_dim)
        states = self.attention_norm(
            node_states + self.residual_dropout(self.output(attended)))
        states = self.ffn_norm(
            states + self.residual_dropout(self.ffn(states)))
        return states, attention


class CausalEventTransformer(nn.Module):
    """Encode an explicit transaction-event DAG with causal message passing."""

    def __init__(
        self,
        num_currencies,
        num_payment_formats,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        edge_dim=10,
        num_relations=4,
        dropout=0.2,
    ):
        super().__init__()
        if num_currencies < 1 or num_payment_formats < 1:
            raise ValueError("categorical cardinalities must be positive")
        category_dim = max(4, hidden_dim // 8)
        self.currency_embedding = nn.Embedding(
            num_currencies, category_dim)
        self.payment_embedding = nn.Embedding(
            num_payment_formats, category_dim)
        self.numeric_projection = nn.Linear(3, hidden_dim)
        self.category_projection = nn.Linear(2 * category_dim, hidden_dim)
        self.target_embedding = nn.Embedding(2, hidden_dim)
        self.temporal_position = TemporalPositionEncoding(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            RelationAwareTemporalLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_dim=edge_dim,
                num_relations=num_relations,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def _category_ids(self, values, cardinality):
        return values.round().long().clamp(0, cardinality - 1)

    def _encode_nodes(self, graph):
        raw = graph.node_raw
        numeric = torch.stack((
            torch.log1p(raw[:, 0].float().clamp_min(0)) / 20.0,
            raw[:, 1].float().clamp(-10.0, 10.0) / 5.0,
            torch.log1p(graph.node_target_delta.clamp_min(0)) / 20.0,
        ), dim=-1)
        categories = torch.cat((
            self.currency_embedding(self._category_ids(
                raw[:, 2], self.currency_embedding.num_embeddings)),
            self.payment_embedding(self._category_ids(
                raw[:, 3], self.payment_embedding.num_embeddings)),
        ), dim=-1)
        is_target = torch.zeros(
            raw.size(0), device=raw.device, dtype=torch.long)
        is_target[graph.target_nodes] = 1
        states = (
            self.numeric_projection(numeric)
            + self.category_projection(categories)
            + self.target_embedding(is_target)
            + self.temporal_position(graph.node_target_delta)
        )
        return self.input_dropout(self.input_norm(torch.nn.functional.gelu(
            states)))

    def forward(self, graph: CausalEventGraphBatch):
        states = self._encode_nodes(graph)
        attentions = []
        for layer in self.layers:
            states, attention = layer(
                states,
                graph.edge_index,
                graph.edge_attr,
                graph.edge_relation,
            )
            attentions.append(attention)
        diagnostics = {
            "event_norm": states.norm(dim=-1),
            "target_event_norm": states[graph.target_nodes].norm(dim=-1),
            "transition_attention": attentions,
            "event_count": graph.graph_ptr[1:] - graph.graph_ptr[:-1],
        }
        return states, states[graph.target_nodes], diagnostics

