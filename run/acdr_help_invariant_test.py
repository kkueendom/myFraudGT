#!/usr/bin/env python3
"""CPU-only invariants for the ACDR-Help inference route."""

import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead


def make_head(dim_in, hidden=32, radius=1.0):
    head = HeteroGNNEdgeHead.__new__(HeteroGNNEdgeHead)
    nn.Module.__init__(head)
    head.acdr_help_num_signals = 10
    head.acdr_help_route_radius = radius
    head.acdr_help_router = head._make_acdr_help_router(
        dim_in + head.acdr_help_num_signals, hidden)
    return head


def route(head, tensors, index=None):
    selected = tensors if index is None else tuple(item[index] for item in tensors)
    return head._acdr_help_route(*selected)


def main():
    torch.manual_seed(20260718)
    dim_in = 8
    sample_count = 11
    head = make_head(dim_in)
    head.eval()
    assert not any(
        isinstance(module, nn.Dropout) for module in head.acdr_help_router)
    assert "labels" not in inspect.signature(
        head._acdr_help_route).parameters

    h = torch.randn(sample_count, dim_in, device="cpu")
    z_base = torch.randn(sample_count, 2, device="cpu")
    delta = torch.tanh(torch.randn(sample_count, 2, device="cpu"))
    pos_sim = torch.rand(sample_count, 1, device="cpu")
    neg_sim = torch.rand(sample_count, 1, device="cpu")
    proto_margin = pos_sim - neg_sim
    ready = torch.randint(0, 2, (sample_count, 1)).float()
    tensors = (h, z_base, delta, pos_sim, neg_sim, proto_margin, ready)

    _, probability, dose = route(head, tensors)
    assert torch.equal(probability, torch.full_like(probability, 0.5))
    assert torch.equal(dose, torch.ones_like(dose))
    beta = torch.tensor(0.37)
    a2_logits = z_base + beta * ready * delta
    acdr_logits = z_base + beta * ready * dose * delta
    fallback_error = (a2_logits - acdr_logits).abs().max().item()
    assert fallback_error == 0.0

    # Give the final layer a deterministic nonzero mapping so independence is
    # tested on varying doses rather than only on the neutral constant route.
    with torch.no_grad():
        head.acdr_help_router[-1].weight.copy_(
            torch.linspace(-0.2, 0.2, head.acdr_help_router[-1].weight.numel())
            .view_as(head.acdr_help_router[-1].weight))
        head.acdr_help_router[-1].bias.fill_(0.07)

    full_score, full_probability, full_dose = route(head, tensors)
    split = 4
    first = route(head, tensors, slice(None, split))
    second = route(head, tensors, slice(split, None))
    chunk_score = torch.cat([first[0], second[0]], dim=0)
    chunk_probability = torch.cat([first[1], second[1]], dim=0)
    chunk_dose = torch.cat([first[2], second[2]], dim=0)

    permutation = torch.randperm(sample_count)
    inverse = torch.argsort(permutation)
    permuted = route(head, tensors, permutation)
    permutation_dose = permuted[2][inverse]

    chunk_error = max(
        (full_score - chunk_score).abs().max().item(),
        (full_probability - chunk_probability).abs().max().item(),
        (full_dose - chunk_dose).abs().max().item(),
    )
    permutation_error = (
        full_dose - permutation_dose).abs().max().item()
    # Different CPU GEMM batch shapes can differ by one float32 rounding step.
    assert chunk_error < 1e-6
    assert permutation_error < 1e-6

    # An unrelated row cannot influence an existing sample.
    changed = list(tensors)
    changed[0] = h.clone()
    changed[0][-1].mul_(100.0)
    changed_dose = route(head, tuple(changed))[2]
    unrelated_row_error = (
        full_dose[:-1] - changed_dose[:-1]).abs().max().item()
    assert unrelated_row_error < 1e-6

    print(
        "ACDR_HELP_INVARIANTS=PASS "
        f"fallback_error={fallback_error:.1e} "
        f"chunk_error={chunk_error:.1e} "
        f"permutation_error={permutation_error:.1e} "
        f"unrelated_row_error={unrelated_row_error:.1e}")


if __name__ == "__main__":
    main()
