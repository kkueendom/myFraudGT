import torch
import torch.nn as nn

from fraudGT.cdvt.event_graph import CausalEventGraphIndex
from fraudGT.cdvt.fusion import DualViewFusionClassifier
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_network
from fraudGT.network.gt_model import GTModel


@register_network("CDVTModel")
class CDVTModel(nn.Module):
    """End-to-end account/event dual-view transaction classifier."""

    TASK = ("node", "to", "node")

    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        if not getattr(cfg.dataset, "tier_evidence", False):
            raise ValueError("CDVT requires dataset.tier_evidence=True")
        self.account_encoder = GTModel(dim_in, dim_out, dataset)
        full_store = dataset["test"][self.TASK]
        if not hasattr(full_store, "raw_edge_attr"):
            raise ValueError("CDVT requires immutable raw_edge_attr")
        self.event_index = CausalEventGraphIndex(
            edge_index=full_store.edge_index,
            timestamps=full_store.timestamps,
            raw_edge_attr=full_store.raw_edge_attr,
        )
        raw = full_store.raw_edge_attr
        account_dim = int(cfg.gt.dim_hidden) * 3
        if cfg.gt.jumping_knowledge:
            account_dim = (
                int(cfg.gt.dim_hidden) * (int(cfg.gt.layers) + 1) * 2
                + int(cfg.gt.dim_hidden)
            )
        self.dual_view = DualViewFusionClassifier(
            account_dim=account_dim,
            num_currencies=int(raw[:, 2].max().item()) + 1,
            num_payment_formats=int(raw[:, 3].max().item()) + 1,
            variant=cfg.cdvt.variant,
            hidden_dim=int(cfg.cdvt.hidden_dim),
            num_heads=int(cfg.cdvt.num_heads),
            num_layers=int(cfg.cdvt.num_layers),
            dropout=float(cfg.cdvt.dropout),
        )
        self.last_diagnostics = None

    def _event_graph(self, edge_ids, condition):
        time_window = int(cfg.cdvt.time_window)
        graph = self.event_index.query(
            edge_ids.detach().cpu(),
            k=int(cfg.cdvt.history_k),
            hops=int(cfg.cdvt.history_hops),
            max_events=int(cfg.cdvt.max_events),
            time_window=None if time_window < 0 else time_window,
        ).to(edge_ids.device)
        if condition == "normal":
            return graph
        if condition == "shuffled":
            return graph.shuffled()
        if condition == "off":
            return graph.off()
        raise ValueError(f"unknown event condition: {condition}")

    def forward_details(self, batch, condition="normal"):
        encoded = self.account_encoder.encode_batch(batch)
        head = self.account_encoder.post_gt
        mask = head._edge_mask(encoded)
        edge_ids = encoded[self.TASK].e_id[mask]
        account_features, labels = head._apply_index(encoded)
        if edge_ids.numel() != labels.numel():
            raise AssertionError("target edge IDs and labels are not aligned")
        graph = self._event_graph(edge_ids, condition)
        logits, diagnostics = self.dual_view(account_features, graph)
        diagnostics = dict(diagnostics)
        diagnostics["target_edge_ids"] = edge_ids
        diagnostics["event_condition"] = condition
        return logits, labels, diagnostics

    def forward(self, batch):
        logits, labels, diagnostics = self.forward_details(batch)
        self.last_diagnostics = diagnostics
        return logits, labels

