import math
from typing import Dict

import torch
import torch.nn as nn

from fraudGT.evidence.tier import EvidenceBatch


EVIDENCE_FAMILIES = {"all", "structure", "temporal", "flow_role"}
EVIDENCE_SUPPORT_MASKS = {
    "all": (1, 1, 1, 1, 1, 1),
    "structure": (1, 1, 0, 1, 1, 1),
    "temporal": (1, 0, 1, 0, 0, 0),
    "flow_role": (1, 1, 0, 0, 0, 0),
}


def evidence_family_channels(family: str) -> Dict[str, bool]:
    if family not in EVIDENCE_FAMILIES:
        raise ValueError(f"unknown evidence family: {family}")
    return {
        "absolute_time": family in {"all", "temporal"},
        "flow_attributes": family in {"all", "flow_role"},
        "relative_time": family in {"all", "temporal"},
        "roles": family in {"all", "structure", "flow_role"},
        "motifs": family in {"all", "structure"},
    }


def evidence_family_support_mask(family: str):
    if family not in EVIDENCE_FAMILIES:
        raise ValueError(f"unknown evidence family: {family}")
    return EVIDENCE_SUPPORT_MASKS[family]


class TransactionEvidenceEncoder(nn.Module):
    """Evidence-only classifier with no FraudGT hidden-state inputs."""

    RAW_DIM = 4
    ROLE_DIM = 5
    MOTIF_DIM = 3
    TOKEN_DIM = RAW_DIM + 1 + ROLE_DIM + MOTIF_DIM

    def __init__(
        self,
        num_currencies: int,
        num_payment_formats: int,
        family: str = "all",
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_currencies < 1 or num_payment_formats < 1:
            raise ValueError("categorical cardinalities must be positive")
        self.family = family
        self.channels = evidence_family_channels(family)
        self.register_buffer(
            "support_mask",
            torch.tensor(
                evidence_family_support_mask(family),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        category_dim = max(4, hidden_dim // 8)
        self.currency_embedding = nn.Embedding(
            num_currencies, category_dim)
        self.payment_embedding = nn.Embedding(
            num_payment_formats, category_dim)

        token_numeric_dim = 3 + self.ROLE_DIM + self.MOTIF_DIM
        self.token_numeric = nn.Linear(token_numeric_dim, hidden_dim)
        self.token_category = nn.Linear(2 * category_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)

        self.target_numeric = nn.Linear(2, hidden_dim)
        self.target_category = nn.Linear(2 * category_dim, hidden_dim)
        self.target_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.support_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    @staticmethod
    def _scaled_time(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(value.float().clamp_min(0)) / 20.0

    @staticmethod
    def _scaled_amount(value: torch.Tensor) -> torch.Tensor:
        return value.float().clamp(-10.0, 10.0) / 5.0

    @staticmethod
    def _category_ids(
        values: torch.Tensor,
        cardinality: int,
    ) -> torch.Tensor:
        return values.round().long().clamp(0, cardinality - 1)

    def _category_features(self, raw: torch.Tensor) -> torch.Tensor:
        currency = self._category_ids(
            raw[..., 2], self.currency_embedding.num_embeddings)
        payment = self._category_ids(
            raw[..., 3], self.payment_embedding.num_embeddings)
        return torch.cat(
            (
                self.currency_embedding(currency),
                self.payment_embedding(payment),
            ),
            dim=-1,
        )

    def _encode_target(self, target_raw: torch.Tensor) -> torch.Tensor:
        if target_raw.dim() != 2 or target_raw.size(1) != self.RAW_DIM:
            raise ValueError("target_raw must have shape [batch, 4]")
        numeric = torch.stack(
            (
                self._scaled_time(target_raw[:, 0]),
                self._scaled_amount(target_raw[:, 1]),
            ),
            dim=-1,
        )
        encoded = (
            self.target_numeric(numeric)
            + self.target_category(self._category_features(target_raw))
        )
        return self.target_norm(self.activation(encoded))

    def _encode_tokens(self, evidence: EvidenceBatch):
        tokens = evidence.tokens
        if tokens.dim() != 3 or tokens.size(-1) != self.TOKEN_DIM:
            raise ValueError(
                "evidence tokens must have shape [batch, tokens, 13]")
        raw = tokens[..., :self.RAW_DIM]
        delta = tokens[..., self.RAW_DIM]
        roles = tokens[
            ..., self.RAW_DIM + 1:self.RAW_DIM + 1 + self.ROLE_DIM]
        motifs = tokens[..., -self.MOTIF_DIM:]

        numeric = torch.cat(
            (
                self._scaled_time(raw[..., 0]).unsqueeze(-1),
                self._scaled_amount(raw[..., 1]).unsqueeze(-1),
                (delta.float() / 20.0).unsqueeze(-1),
                roles.float(),
                motifs.float(),
            ),
            dim=-1,
        )
        if not self.channels["absolute_time"]:
            numeric[..., 0] = 0
        if not self.channels["flow_attributes"]:
            numeric[..., 1] = 0
        if not self.channels["relative_time"]:
            numeric[..., 2] = 0
        if not self.channels["roles"]:
            numeric[..., 3:3 + self.ROLE_DIM] = 0
        if not self.channels["motifs"]:
            numeric[..., -self.MOTIF_DIM:] = 0

        category = self._category_features(raw)
        if not self.channels["flow_attributes"]:
            category = torch.zeros_like(category)
        encoded = self.token_numeric(numeric) + self.token_category(category)
        encoded = self.token_norm(self.activation(encoded))
        return encoded, evidence.mask.bool()

    def forward(
        self,
        evidence: EvidenceBatch,
        target_raw: torch.Tensor,
    ):
        target = self._encode_target(target_raw)
        tokens, mask = self._encode_tokens(evidence)
        if tokens.size(0) != target.size(0):
            raise ValueError("evidence and target batch sizes differ")

        has_evidence = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_tokens = tokens
        if safe_mask.size(1):
            safe_mask[~has_evidence, 0] = True
            safe_tokens = safe_tokens.clone()
            safe_tokens[~has_evidence, 0] = 0
        encoded = self.context_encoder(
            safe_tokens, src_key_padding_mask=~safe_mask)
        query = self.query_projection(target)
        scores = (encoded * query.unsqueeze(1)).sum(-1)
        scores = scores / math.sqrt(encoded.size(-1))
        scores = scores.masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled * has_evidence.unsqueeze(-1)

        support_values = (
            evidence.support.float().clamp_min(0)
            * self.support_mask.view(1, -1)
        )
        support = torch.log1p(support_values)
        support = self.support_encoder(support)
        support = support * has_evidence.unsqueeze(-1)
        features = torch.cat(
            (target, pooled, support, target * pooled), dim=-1)
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        diagnostics = {
            "has_evidence": has_evidence,
            "attention": weights * safe_mask,
            "pooled_norm": pooled.norm(dim=-1),
        }
        return logits, diagnostics
