import math

import torch
import torch.nn as nn

from fraudGT.evidence.tier import EvidenceBatch


class CausalTemporalSubgraphEncoder(nn.Module):
    """Encode target-admissible transaction history independently of FraudGT."""

    RAW_DIM = 4
    ROLE_DIM = 5
    MOTIF_DIM = 3
    TOKEN_DIM = RAW_DIM + 1 + ROLE_DIM + MOTIF_DIM

    def __init__(
        self,
        num_currencies,
        num_payment_formats,
        hidden_dim=64,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        if hidden_dim < 8 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_currencies < 1 or num_payment_formats < 1:
            raise ValueError("categorical cardinalities must be positive")
        category_dim = max(4, hidden_dim // 8)
        self.currency_embedding = nn.Embedding(
            num_currencies, category_dim)
        self.payment_embedding = nn.Embedding(
            num_payment_formats, category_dim)
        self.token_numeric = nn.Linear(
            3 + self.ROLE_DIM + self.MOTIF_DIM, hidden_dim)
        self.token_category = nn.Linear(2 * category_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.target_numeric = nn.Linear(2, hidden_dim)
        self.target_category = nn.Linear(2 * category_dim, hidden_dim)
        self.target_norm = nn.LayerNorm(hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.source_query = nn.Linear(hidden_dim, hidden_dim)
        self.destination_query = nn.Linear(hidden_dim, hidden_dim)
        self.global_query = nn.Linear(hidden_dim, hidden_dim)
        self.support_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    @staticmethod
    def _scaled_time(values):
        return torch.log1p(values.float().clamp_min(0)) / 20.0

    @staticmethod
    def _scaled_amount(values):
        return values.float().clamp(-10.0, 10.0) / 5.0

    @staticmethod
    def _category_ids(values, cardinality):
        return values.round().long().clamp(0, cardinality - 1)

    def _category_features(self, raw):
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

    def _encode_target(self, target_raw):
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

    def _encode_tokens(self, evidence):
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
        encoded = (
            self.token_numeric(numeric)
            + self.token_category(self._category_features(raw))
        )
        return self.token_norm(self.activation(encoded)), roles

    @staticmethod
    def _safe_history(tokens, mask):
        has_history = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_tokens = tokens
        if safe_mask.size(1):
            safe_mask[~has_history, 0] = True
            safe_tokens = safe_tokens.clone()
            safe_tokens[~has_history, 0] = 0
        return safe_tokens, safe_mask, has_history

    @staticmethod
    def _attention_pool(tokens, mask, query):
        has_values = mask.any(dim=1)
        safe_mask = mask.clone()
        if safe_mask.size(1):
            safe_mask[~has_values, 0] = True
        scores = (tokens * query.unsqueeze(1)).sum(-1)
        scores = scores / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled * has_values.unsqueeze(-1)
        return pooled, weights * mask

    def forward(self, evidence: EvidenceBatch, target_raw):
        target = self._encode_target(target_raw)
        tokens, roles = self._encode_tokens(evidence)
        if tokens.size(0) != target.size(0):
            raise ValueError("evidence and target batch sizes differ")
        mask = evidence.mask.bool()
        safe_tokens, safe_mask, has_history = self._safe_history(
            tokens, mask)
        encoded = self.history_encoder(
            safe_tokens, src_key_padding_mask=~safe_mask)
        encoded = encoded * mask.unsqueeze(-1)

        source_mask = mask & roles[..., :2].bool().any(dim=-1)
        destination_mask = mask & roles[..., 2:4].bool().any(dim=-1)
        source, source_attention = self._attention_pool(
            encoded, source_mask, self.source_query(target))
        destination, destination_attention = self._attention_pool(
            encoded, destination_mask, self.destination_query(target))
        global_history, global_attention = self._attention_pool(
            encoded, mask, self.global_query(target))
        support = self.support_encoder(
            torch.log1p(evidence.support.float().clamp_min(0)))
        support = support * has_history.unsqueeze(-1)
        history_features = torch.cat(
            (source, destination, global_history, support), dim=-1)
        representation = self.evidence_projection(
            torch.cat((target, history_features), dim=-1))
        diagnostics = {
            "has_history": has_history,
            "source_attention": source_attention,
            "destination_attention": destination_attention,
            "global_attention": global_attention,
            "history_norm": history_features.norm(dim=-1),
            "evidence_norm": representation.norm(dim=-1),
            "support_count": evidence.support[:, 0],
        }
        return representation, encoded, mask, diagnostics


class CETFusionClassifier(nn.Module):
    """Fuse frozen FraudGT edge representations with causal history tokens."""

    VARIANTS = {"encoder_only", "fusion"}

    def __init__(
        self,
        base_feature_dim,
        num_currencies,
        num_payment_formats,
        variant="fusion",
        hidden_dim=64,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown CET variant: {variant}")
        self.variant = variant
        self.evidence_encoder = CausalTemporalSubgraphEncoder(
            num_currencies=num_currencies,
            num_payment_formats=num_payment_formats,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.base_projection = nn.Sequential(
            nn.Linear(base_feature_dim, hidden_dim),
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
        self.evidence_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    @staticmethod
    def _safe_tokens(tokens, mask):
        has_history = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_tokens = tokens
        if safe_mask.size(1):
            safe_mask[~has_history, 0] = True
            safe_tokens = safe_tokens.clone()
            safe_tokens[~has_history, 0] = 0
        return safe_tokens, safe_mask, has_history

    def forward(self, base_features, evidence, target_raw):
        evidence_repr, tokens, mask, evidence_diagnostics = (
            self.evidence_encoder(evidence, target_raw))
        evidence_logits = self.evidence_classifier(
            evidence_repr).squeeze(-1)
        if self.variant == "encoder_only":
            diagnostics = dict(evidence_diagnostics)
            diagnostics.update({
                "base_norm": torch.zeros_like(evidence_logits),
                "fusion_norm": evidence_repr.norm(dim=-1),
                "fusion_gain_norm": torch.zeros_like(evidence_logits),
                "cross_attention": torch.zeros_like(mask, dtype=torch.float32),
                "evidence_logits": evidence_logits,
            })
            return evidence_logits, diagnostics

        if (
            base_features.dim() != 2
            or base_features.size(0) != evidence_repr.size(0)
        ):
            raise ValueError("base features must align with evidence rows")
        base = self.base_projection(base_features)
        safe_tokens, safe_mask, has_history = self._safe_tokens(tokens, mask)
        context, attention = self.cross_attention(
            base.unsqueeze(1),
            safe_tokens,
            safe_tokens,
            key_padding_mask=~safe_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1) * has_history.unsqueeze(-1)
        fused = self.fusion_norm(
            base + self.cross_projection(context))
        features = torch.cat(
            (
                fused,
                evidence_repr,
                (fused - evidence_repr).abs(),
                fused * evidence_repr,
            ),
            dim=-1,
        )
        logits = self.fusion_classifier(features).squeeze(-1)
        diagnostics = dict(evidence_diagnostics)
        diagnostics.update({
            "base_norm": base.norm(dim=-1),
            "fusion_norm": fused.norm(dim=-1),
            "fusion_gain_norm": (fused - base).norm(dim=-1),
            "cross_attention": attention.squeeze(1) * mask,
            "evidence_logits": evidence_logits,
        })
        return logits, diagnostics
