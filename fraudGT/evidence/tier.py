from dataclasses import dataclass
from typing import Optional

import torch


def build_raw_edge_attributes(
    timestamps: torch.Tensor,
    amounts: torch.Tensor,
    currencies: torch.Tensor,
    payment_formats: torch.Tensor,
    train_end: int,
) -> torch.Tensor:
    """Build the immutable TIER input attributes using train-only statistics."""
    tensors = (timestamps, amounts, currencies, payment_formats)
    if any(tensor.dim() != 1 for tensor in tensors):
        raise ValueError("raw edge attribute inputs must be one-dimensional")
    if len({int(tensor.numel()) for tensor in tensors}) != 1:
        raise ValueError("raw edge attribute inputs must have equal lengths")
    if train_end < 1 or train_end > timestamps.numel():
        raise ValueError("train_end must select a non-empty prefix")
    if any(not torch.isfinite(tensor.float()).all() for tensor in tensors):
        raise ValueError("raw edge attribute inputs must be finite")
    if (amounts < 0).any():
        raise ValueError("amounts must be non-negative")
    if (currencies < 0).any() or (payment_formats < 0).any():
        raise ValueError("categorical IDs must be non-negative")

    amount = amounts.float()
    train_amount = amount[:train_end]
    amount_mean = train_amount.mean()
    amount_std = train_amount.std(unbiased=False).clamp_min(1e-6)
    normalized_amount = (amount - amount_mean) / amount_std

    return torch.stack(
        (
            timestamps.float(),
            normalized_amount,
            currencies.float(),
            payment_formats.float(),
        ),
        dim=-1,
    )


def _fit_affine_from_overlap(
    target_train: torch.Tensor,
    source_train: torch.Tensor,
):
    target = target_train.detach().cpu().double()
    source = source_train.detach().cpu().double()
    if target.shape != source.shape or target.dim() != 1:
        raise ValueError("affine overlap inputs must be equal 1D tensors")
    source_scale = source.std(unbiased=False)
    target_scale = target.std(unbiased=False)
    if source_scale <= 1e-12 or target_scale <= 1e-12:
        raise ValueError("affine overlap feature must have nonzero variance")
    slope = target_scale / source_scale
    intercept = target.mean() - slope * source.mean()
    recovered = slope * source + intercept
    error = recovered - target
    rmse = error.square().mean().sqrt()
    max_error = error.abs().max()
    if rmse > 1e-5 or max_error > 5e-4:
        raise ValueError(
            "legacy split features are not affine-compatible: "
            f"rmse={float(rmse):.6g}, max={float(max_error):.6g}"
        )
    return slope, intercept


def _recover_category_codes(values: torch.Tensor) -> torch.Tensor:
    unique_values = torch.unique(values).sort().values
    if unique_values.numel() == 1:
        return torch.zeros_like(values)
    spacing = torch.median(unique_values[1:] - unique_values[:-1])
    if spacing <= 1e-12:
        raise ValueError("category values must have positive spacing")
    codes = torch.round((values - unique_values[0]) / spacing)
    reconstructed = unique_values[0] + codes * spacing
    if (reconstructed - values).abs().max() > 5e-4:
        raise ValueError("legacy category values are not evenly spaced")
    if (codes < 0).any():
        raise ValueError("recovered category codes must be non-negative")
    return codes.float()


def recover_train_normalized_raw_edge_attributes(
    timestamps: torch.Tensor,
    train_edge_attr: torch.Tensor,
    full_edge_attr: torch.Tensor,
) -> torch.Tensor:
    """Recover train-only transaction fields from legacy split z-scores.

    Historical AML caches z-normalized each prefix separately. Those cached
    columns are affine transforms of the same pre-encoder transaction fields.
    The overlap with the train prefix therefore identifies an exact affine map
    from the full-prefix cache to the train-prefix scale without labels or
    validation/test statistics in the resulting representation.
    """
    if train_edge_attr.dim() != 2 or full_edge_attr.dim() != 2:
        raise ValueError("legacy edge attributes must be two-dimensional")
    if train_edge_attr.size(1) < 4 or full_edge_attr.size(1) < 4:
        raise ValueError("legacy edge attributes must contain four columns")
    train_end = int(train_edge_attr.size(0))
    num_edges = int(full_edge_attr.size(0))
    if train_end < 2 or train_end > num_edges:
        raise ValueError("legacy train prefix must contain at least two edges")
    if timestamps.dim() != 1 or timestamps.numel() != num_edges:
        raise ValueError("timestamps must contain one value per full edge")
    if not torch.isfinite(train_edge_attr[:, :4]).all():
        raise ValueError("legacy train edge attributes must be finite")
    if not torch.isfinite(full_edge_attr[:, :4]).all():
        raise ValueError("legacy full edge attributes must be finite")

    recovered_columns = []
    for column in (1, 2, 3):
        slope, intercept = _fit_affine_from_overlap(
            train_edge_attr[:, column],
            full_edge_attr[:train_end, column],
        )
        recovered_columns.append(
            slope * full_edge_attr[:, column].detach().cpu().double()
            + intercept
        )

    # Historical z_norm uses the unbiased sample standard deviation. Convert
    # its amount channel to the population z-score used by the TIER contract.
    amount_scale = (train_end / (train_end - 1.0)) ** 0.5
    normalized_amount = (recovered_columns[0] * amount_scale).float()
    currency_codes = _recover_category_codes(recovered_columns[1])
    payment_codes = _recover_category_codes(recovered_columns[2])
    return torch.stack(
        (
            timestamps.detach().cpu().float(),
            normalized_amount,
            currency_codes,
            payment_codes,
        ),
        dim=-1,
    ).contiguous()


@dataclass(frozen=True)
class EvidenceBatch:
    tokens: torch.Tensor
    mask: torch.Tensor
    context_edge_ids: torch.Tensor
    support: torch.Tensor

    def shuffled(self, permutation: torch.Tensor) -> "EvidenceBatch":
        if permutation.dim() != 1 or permutation.numel() != self.tokens.size(0):
            raise ValueError("permutation must contain one index per target")
        if permutation.dtype != torch.long:
            raise ValueError("permutation must use torch.long indices")
        expected = torch.arange(
            permutation.numel(), device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("permutation must be a bijection")
        return EvidenceBatch(
            tokens=self.tokens[permutation],
            mask=self.mask[permutation],
            context_edge_ids=self.context_edge_ids[permutation],
            support=self.support[permutation],
        )

    def off(self) -> "EvidenceBatch":
        return EvidenceBatch(
            tokens=torch.zeros_like(self.tokens),
            mask=torch.zeros_like(self.mask),
            context_edge_ids=torch.full_like(self.context_edge_ids, -1),
            support=torch.zeros_like(self.support),
        )


class TemporalIncidentIndex:
    """Read-only incident-event index for transaction-centered evidence.

    Edge IDs are the deterministic tie-breaker for transactions sharing a
    timestamp. A context edge is admissible when it is earlier than the target
    or has the same timestamp and a smaller global edge ID.
    """

    ROLE_DIM = 5
    MOTIF_DIM = 3
    SUPPORT_DIM = 6

    def __init__(
        self,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
        raw_edge_attr: torch.Tensor,
    ) -> None:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        num_edges = int(edge_index.size(1))
        if timestamps.dim() != 1 or timestamps.numel() != num_edges:
            raise ValueError("timestamps must contain one value per edge")
        if raw_edge_attr.dim() != 2 or raw_edge_attr.size(0) != num_edges:
            raise ValueError("raw_edge_attr must contain one row per edge")
        if num_edges and (edge_index < 0).any():
            raise ValueError("node IDs must be non-negative")
        if timestamps.numel() > 1 and (timestamps[1:] < timestamps[:-1]).any():
            raise ValueError("edges must be sorted by non-decreasing timestamp")
        if not torch.isfinite(raw_edge_attr.float()).all():
            raise ValueError("raw_edge_attr must be finite")

        self.edge_index = edge_index.detach().cpu().long().contiguous()
        self.timestamps = timestamps.detach().cpu().long().contiguous()
        self.raw_edge_attr = (
            raw_edge_attr.detach().cpu().float().contiguous())
        self.num_edges = num_edges
        self.num_nodes = (
            int(self.edge_index.max().item()) + 1 if num_edges else 0)

        edge_ids = torch.arange(num_edges, dtype=torch.long)
        incident_nodes = torch.cat(
            (self.edge_index[0], self.edge_index[1]), dim=0)
        incident_edges = torch.cat((edge_ids, edge_ids), dim=0)
        if incident_nodes.numel():
            key = incident_nodes * (num_edges + 1) + incident_edges
            order = torch.argsort(key)
            self.incident_edge_ids = incident_edges[order]
            counts = torch.bincount(
                incident_nodes[order], minlength=self.num_nodes)
        else:
            self.incident_edge_ids = torch.empty(0, dtype=torch.long)
            counts = torch.empty(0, dtype=torch.long)
        self.offsets = torch.zeros(self.num_nodes + 1, dtype=torch.long)
        if counts.numel():
            self.offsets[1:] = counts.cumsum(0)

    @property
    def token_dim(self) -> int:
        return int(self.raw_edge_attr.size(1)) + 1 + self.ROLE_DIM + self.MOTIF_DIM

    def _node_history(
        self,
        node_id: int,
        target_time: int,
        target_edge_id: int,
    ) -> torch.Tensor:
        start = int(self.offsets[node_id])
        stop = int(self.offsets[node_id + 1])
        edge_ids = self.incident_edge_ids[start:stop]
        if not edge_ids.numel():
            return edge_ids
        times = self.timestamps[edge_ids]
        admissible = (times < target_time) | (
            (times == target_time) & (edge_ids < target_edge_id))
        return edge_ids[admissible]

    @staticmethod
    def _mark_pair(
        pair_to_positions,
        pair,
        flags: torch.Tensor,
        column: int,
    ) -> None:
        for position in pair_to_positions.get(pair, ()):
            flags[position, column] = 1.0

    def _motif_flags(
        self,
        context_edge_ids: torch.Tensor,
        target_src: int,
        target_dst: int,
    ) -> torch.Tensor:
        count = int(context_edge_ids.numel())
        flags = torch.zeros((count, self.MOTIF_DIM), dtype=torch.float32)
        if not count:
            return flags

        context_edges = self.edge_index[:, context_edge_ids]
        pair_to_positions = {}
        for position, (src, dst) in enumerate(context_edges.t().tolist()):
            pair_to_positions.setdefault((src, dst), []).append(position)

        self._mark_pair(
            pair_to_positions, (target_dst, target_src), flags, 0)

        for middle in {
                int(dst) for src, dst in pair_to_positions if src == target_src}:
            if (middle, target_dst) in pair_to_positions:
                self._mark_pair(
                    pair_to_positions, (target_src, middle), flags, 1)
                self._mark_pair(
                    pair_to_positions, (middle, target_dst), flags, 1)

        for middle in {
                int(dst) for src, dst in pair_to_positions if src == target_dst}:
            if (middle, target_src) in pair_to_positions:
                self._mark_pair(
                    pair_to_positions, (target_dst, middle), flags, 2)
                self._mark_pair(
                    pair_to_positions, (middle, target_src), flags, 2)
        return flags

    def _query_one(
        self,
        target_edge_id: int,
        max_tokens: int,
        time_window: Optional[int],
    ):
        target_src = int(self.edge_index[0, target_edge_id])
        target_dst = int(self.edge_index[1, target_edge_id])
        target_time = int(self.timestamps[target_edge_id])
        histories = (
            self._node_history(target_src, target_time, target_edge_id),
            self._node_history(target_dst, target_time, target_edge_id),
        )
        context_edge_ids = torch.unique(torch.cat(histories))
        if time_window is not None and context_edge_ids.numel():
            deltas = target_time - self.timestamps[context_edge_ids]
            context_edge_ids = context_edge_ids[deltas <= time_window]
        if context_edge_ids.numel():
            recency_key = (
                self.timestamps[context_edge_ids] * (self.num_edges + 1) +
                context_edge_ids
            )
            order = torch.argsort(recency_key, descending=True)
            context_edge_ids = context_edge_ids[order[:max_tokens]]

        count = int(context_edge_ids.numel())
        tokens = torch.zeros((max_tokens, self.token_dim), dtype=torch.float32)
        mask = torch.zeros(max_tokens, dtype=torch.bool)
        padded_ids = torch.full((max_tokens,), -1, dtype=torch.long)
        support = torch.zeros(self.SUPPORT_DIM, dtype=torch.float32)
        if not count:
            return tokens, mask, padded_ids, support

        context_edges = self.edge_index[:, context_edge_ids]
        context_src, context_dst = context_edges
        roles = torch.stack(
            (
                context_dst == target_src,
                context_src == target_src,
                context_dst == target_dst,
                context_src == target_dst,
                (context_src == target_dst) & (context_dst == target_src),
            ),
            dim=-1,
        ).float()
        motif = self._motif_flags(
            context_edge_ids, target_src, target_dst)
        deltas = (
            target_time - self.timestamps[context_edge_ids]).float()
        delta_feature = torch.log1p(deltas).unsqueeze(-1)
        selected_tokens = torch.cat(
            (
                self.raw_edge_attr[context_edge_ids],
                delta_feature,
                roles,
                motif,
            ),
            dim=-1,
        )
        tokens[:count] = selected_tokens
        mask[:count] = True
        padded_ids[:count] = context_edge_ids

        time_span = (
            self.timestamps[context_edge_ids].max() -
            self.timestamps[context_edge_ids].min()
        ).float()
        support[:] = torch.tensor(
            (
                float(count),
                float((roles[:, :4].sum(0) > 0).sum()),
                float(torch.log1p(time_span)),
                float(motif[:, 0].sum()),
                float(motif[:, 1].sum() / 2.0),
                float(motif[:, 2].sum() / 2.0),
            ),
            dtype=torch.float32,
        )
        return tokens, mask, padded_ids, support

    def query(
        self,
        target_edge_ids: torch.Tensor,
        max_tokens: int,
        time_window: Optional[int] = None,
    ) -> EvidenceBatch:
        if target_edge_ids.dim() != 1:
            raise ValueError("target_edge_ids must be one-dimensional")
        if target_edge_ids.dtype != torch.long:
            raise ValueError("target_edge_ids must use torch.long")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if time_window is not None and time_window < 0:
            raise ValueError("time_window must be non-negative")
        if target_edge_ids.numel() and (
                (target_edge_ids < 0).any() or
                (target_edge_ids >= self.num_edges).any()):
            raise IndexError("target edge ID is out of range")

        output_device = target_edge_ids.device
        target_ids_cpu = target_edge_ids.detach().cpu()
        rows = [
            self._query_one(int(edge_id), max_tokens, time_window)
            for edge_id in target_ids_cpu.tolist()
        ]
        if rows:
            tokens, masks, context_ids, supports = zip(*rows)
            batch = EvidenceBatch(
                tokens=torch.stack(tokens),
                mask=torch.stack(masks),
                context_edge_ids=torch.stack(context_ids),
                support=torch.stack(supports),
            )
        else:
            batch = EvidenceBatch(
                tokens=torch.empty((0, max_tokens, self.token_dim)),
                mask=torch.empty((0, max_tokens), dtype=torch.bool),
                context_edge_ids=torch.empty(
                    (0, max_tokens), dtype=torch.long),
                support=torch.empty((0, self.SUPPORT_DIM)),
            )
        return EvidenceBatch(
            tokens=batch.tokens.to(output_device),
            mask=batch.mask.to(output_device),
            context_edge_ids=batch.context_edge_ids.to(output_device),
            support=batch.support.to(output_device),
        )
