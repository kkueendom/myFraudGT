import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from fraudGT.cdvt.temporal_transformer import CausalEventTransformer


class DualViewFusionClassifier(nn.Module):
    VARIANTS = {"event_only", "dual_view"}

    def __init__(
        self,
        account_dim,
        num_currencies,
        num_payment_formats,
        variant="dual_view",
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown CDVT variant: {variant}")
        self.variant = variant
        self.event_encoder = CausalEventTransformer(
            num_currencies=num_currencies,
            num_payment_formats=num_payment_formats,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if variant == "dual_view":
            self.account_projection = nn.Sequential(
                nn.Linear(account_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_projection = nn.Linear(hidden_dim, hidden_dim)
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, account_features, event_graph):
        event_states, target_events, event_diagnostics = (
            self.event_encoder(event_graph))
        if self.variant == "event_only":
            logits = self.event_classifier(target_events).squeeze(-1)
            diagnostics = dict(event_diagnostics)
            diagnostics.update({
                "account_norm": torch.zeros_like(logits),
                "fusion_norm": target_events.norm(dim=-1),
                "fusion_gain_norm": torch.zeros_like(logits),
                "cross_attention": event_states.new_zeros(
                    (event_graph.num_graphs, 1, 1)),
            })
            return logits, diagnostics

        if account_features.dim() != 2 or (
            account_features.size(0) != event_graph.num_graphs
        ):
            raise ValueError("account and event graph rows must align")
        account = self.account_projection(account_features)
        dense_events, event_mask = to_dense_batch(
            event_states, event_graph.node_graph)
        context, attention = self.cross_attention(
            account.unsqueeze(1),
            dense_events,
            dense_events,
            key_padding_mask=~event_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.cross_projection(context.squeeze(1))
        fused = self.fusion_norm(account + context)
        logits = self.classifier(fused).squeeze(-1)
        diagnostics = dict(event_diagnostics)
        diagnostics.update({
            "account_norm": account.norm(dim=-1),
            "fusion_norm": fused.norm(dim=-1),
            "fusion_gain_norm": (fused - account).norm(dim=-1),
            "cross_attention": attention,
        })
        return logits, diagnostics
