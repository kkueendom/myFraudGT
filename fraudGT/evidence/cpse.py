"""Causal predictive-surprise evidence without fraud-label supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fraudGT.evidence.tier import EvidenceBatch


class CausalPredictiveSurpriseEncoder(nn.Module):
    RAW_DIM = 4
    ROLE_DIM = 5
    MOTIF_DIM = 3
    FEATURE_DIM = 29

    def __init__(
        self,
        num_currencies,
        num_payment_formats,
        hidden_dim=48,
        dropout=0.15,
    ):
        super().__init__()
        category_dim = max(4, hidden_dim // 8)
        self.currency_embedding = nn.Embedding(
            num_currencies, category_dim)
        self.payment_embedding = nn.Embedding(
            num_payment_formats, category_dim)
        numeric_dim = 3 + self.ROLE_DIM + self.MOTIF_DIM
        self.token_encoder = nn.Sequential(
            nn.Linear(numeric_dim + 2 * category_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.support_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.amount_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.currency_head = nn.Linear(hidden_dim, num_currencies)
        self.payment_head = nn.Linear(hidden_dim, num_payment_formats)
        self.compact_head = nn.Linear(hidden_dim, 16)

    @staticmethod
    def _category_ids(values, cardinality):
        return values.round().long().clamp(0, cardinality - 1)

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (
            (values * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

    def forward(self, evidence: EvidenceBatch):
        tokens = evidence.tokens
        if tokens.ndim != 3 or tokens.size(-1) != 13:
            raise ValueError("CPSE evidence tokens must have shape [B,T,13]")
        raw = tokens[..., :self.RAW_DIM]
        delta = tokens[..., self.RAW_DIM]
        roles = tokens[
            ..., self.RAW_DIM + 1:self.RAW_DIM + 1 + self.ROLE_DIM]
        motifs = tokens[..., -self.MOTIF_DIM:]
        currency = self._category_ids(
            raw[..., 2], self.currency_embedding.num_embeddings)
        payment = self._category_ids(
            raw[..., 3], self.payment_embedding.num_embeddings)
        numeric = torch.cat(
            (
                (raw[..., 1].float().clamp(-10, 10) / 5).unsqueeze(-1),
                (torch.log1p(raw[..., 0].float().clamp_min(0)) / 20)
                .unsqueeze(-1),
                (delta.float() / 20).unsqueeze(-1),
                roles.float(),
                motifs.float(),
            ),
            dim=-1,
        )
        encoded = self.token_encoder(torch.cat(
            (
                numeric,
                self.currency_embedding(currency),
                self.payment_embedding(payment),
            ),
            dim=-1,
        ))
        mask = evidence.mask.bool()
        encoded = encoded * mask.unsqueeze(-1)
        source_mask = mask & roles[..., :2].bool().any(dim=-1)
        destination_mask = mask & roles[..., 2:4].bool().any(dim=-1)
        source = self._masked_mean(encoded, source_mask)
        destination = self._masked_mean(encoded, destination_mask)
        global_history = self._masked_mean(encoded, mask)
        support = self.support_encoder(
            torch.log1p(evidence.support.float().clamp_min(0)))
        context = self.context(torch.cat(
            (source, destination, global_history, support), dim=-1))
        return {
            "context": context,
            "amount": self.amount_head(context).squeeze(-1),
            "delta": F.softplus(self.delta_head(context).squeeze(-1)),
            "currency": self.currency_head(context),
            "payment": self.payment_head(context),
            "compact": torch.tanh(self.compact_head(context)),
            "has_history": mask.any(dim=1),
        }

    @staticmethod
    def targets(evidence, target_raw):
        mask = evidence.mask.bool()
        context_time = evidence.tokens[..., 0].float().masked_fill(
            ~mask, -torch.inf)
        latest = context_time.max(dim=1).values
        target_time = target_raw[:, 0].float()
        delta = torch.where(
            mask.any(dim=1),
            torch.log1p((target_time - latest).clamp_min(0)) / 20,
            torch.zeros_like(target_time),
        )
        return {
            "amount": target_raw[:, 1].float().clamp(-10, 10) / 5,
            "delta": delta,
            "currency": target_raw[:, 2].round().long(),
            "payment": target_raw[:, 3].round().long(),
        }

    def self_supervised_loss(self, prediction, evidence, target_raw):
        target = self.targets(evidence, target_raw)
        active = prediction["has_history"]
        if not active.any():
            return prediction["amount"].sum() * 0.0
        amount = F.smooth_l1_loss(
            prediction["amount"][active], target["amount"][active])
        delta = F.smooth_l1_loss(
            prediction["delta"][active], target["delta"][active])
        currency = F.cross_entropy(
            prediction["currency"][active], target["currency"][active])
        payment = F.cross_entropy(
            prediction["payment"][active], target["payment"][active])
        return amount + delta + 0.5 * (currency + payment)

    def surprise_features(self, prediction, evidence, target_raw):
        target = self.targets(evidence, target_raw)
        amount_residual = target["amount"] - prediction["amount"]
        delta_residual = target["delta"] - prediction["delta"]
        currency_nll = F.cross_entropy(
            prediction["currency"], target["currency"], reduction="none")
        payment_nll = F.cross_entropy(
            prediction["payment"], target["payment"], reduction="none")
        currency_confidence = torch.softmax(
            prediction["currency"], dim=-1).max(dim=-1).values
        payment_confidence = torch.softmax(
            prediction["payment"], dim=-1).max(dim=-1).values
        support = torch.log1p(
            evidence.support.float().clamp_min(0)) / 8.0
        features = torch.cat(
            (
                amount_residual[:, None],
                amount_residual.abs()[:, None],
                delta_residual[:, None],
                delta_residual.abs()[:, None],
                currency_nll[:, None],
                payment_nll[:, None],
                currency_confidence[:, None],
                payment_confidence[:, None],
                support,
                prediction["has_history"].float()[:, None],
                prediction["compact"],
            ),
            dim=-1,
        )
        if features.size(1) != self.FEATURE_DIM:
            raise AssertionError("unexpected CPSE feature dimension")
        return features

