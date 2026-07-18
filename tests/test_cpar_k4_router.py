import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead


class CparK4RouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim_in = 6
        self.head = HeteroGNNEdgeHead.__new__(HeteroGNNEdgeHead)
        nn.Module.__init__(self.head)
        self.head.register_buffer(
            'cpar_action_doses',
            torch.tensor([0.0, 0.5, 1.0, 1.5]))
        self.head.cpar_router = nn.Sequential(
            nn.Linear(self.dim_in + 9, 8),
            nn.GELU(),
            nn.Linear(8, 4),
        )
        nn.init.zeros_(self.head.cpar_router[-1].weight)
        nn.init.zeros_(self.head.cpar_router[-1].bias)

    def inputs(self, count=11):
        h = torch.randn(count, self.dim_in)
        z_base = torch.randn(count, 2)
        residual = 0.2 * torch.randn(count, 2)
        pos_sim = torch.rand(count, 1)
        neg_sim = torch.rand(count, 1)
        proto_margin = pos_sim - neg_sim
        ready = torch.randint(0, 2, (count, 1)).float()
        return (
            h, z_base, residual, pos_sim, neg_sim, proto_margin, ready)

    def route(self, inputs):
        return self.head._cpar_route(*inputs)

    def test_uniform_router_is_exact_a2_fallback(self):
        inputs = self.inputs()
        _, action_prob, confidence, dose = self.route(inputs)
        expected_prob = torch.full_like(action_prob, 0.25)
        self.assertTrue(torch.equal(action_prob, expected_prob))
        self.assertTrue(torch.equal(confidence, torch.zeros_like(confidence)))
        self.assertTrue(torch.equal(dose, torch.ones_like(dose)))

        z_base = inputs[1]
        residual = inputs[2]
        self.assertTrue(torch.equal(
            z_base + dose * residual,
            z_base + residual))

    def test_route_is_independent_of_batch_partition_and_order(self):
        nn.init.normal_(self.head.cpar_router[-1].weight, std=0.2)
        nn.init.normal_(self.head.cpar_router[-1].bias, std=0.1)
        inputs = self.inputs(count=13)
        full = self.route(inputs)

        split_at = 5
        left = self.route(tuple(item[:split_at] for item in inputs))
        right = self.route(tuple(item[split_at:] for item in inputs))
        for full_tensor, left_tensor, right_tensor in zip(full, left, right):
            partitioned = torch.cat([left_tensor, right_tensor], dim=0)
            self.assertTrue(torch.allclose(
                full_tensor, partitioned, atol=1e-7, rtol=1e-6))

        permutation = torch.randperm(inputs[0].size(0))
        inverse = torch.argsort(permutation)
        permuted = self.route(tuple(item[permutation] for item in inputs))
        for full_tensor, permuted_tensor in zip(full, permuted):
            self.assertTrue(torch.allclose(
                full_tensor, permuted_tensor[inverse],
                atol=1e-7, rtol=1e-6))

    def test_router_contains_no_dropout(self):
        self.assertFalse(any(
            isinstance(module, nn.Dropout)
            for module in self.head.cpar_router.modules()))


if __name__ == '__main__':
    unittest.main()
