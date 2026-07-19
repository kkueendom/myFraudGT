import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead


class DabrRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.dim_in = 6
        self.head = HeteroGNNEdgeHead.__new__(HeteroGNNEdgeHead)
        nn.Module.__init__(self.head)
        self.head.dabr_route_strength = 0.25
        self.head.register_buffer(
            'dabr_action_doses', torch.tensor([0.75, 1.0, 1.25]))
        self.head.dabr_router = HeteroGNNEdgeHead._dabr_make_router(
            self.dim_in, 8)
        self.head.dabr_gap_floor = 1e-4
        self.head.dabr_gap_threshold = 0.005

    def inputs(self, count=13):
        h = torch.randn(count, self.dim_in)
        z_base = torch.randn(count, 2)
        residual = 0.2 * torch.randn(count, 2)
        pos_sim = torch.rand(count, 1)
        neg_sim = torch.rand(count, 1)
        proto_margin = pos_sim - neg_sim
        ready = torch.ones(count, 1)
        return (
            h, z_base, residual, pos_sim, neg_sim, proto_margin, ready)

    def route(self, inputs):
        return self.head._dabr_route(*inputs)

    def test_uniform_router_is_exact_a2_fallback(self):
        inputs = self.inputs()
        _, probability, confidence, dose = self.route(inputs)
        self.assertTrue(torch.equal(
            probability, torch.full_like(probability, 1.0 / 3.0)))
        self.assertTrue(torch.equal(confidence, torch.zeros_like(confidence)))
        self.assertTrue(torch.equal(dose, torch.ones_like(dose)))
        self.assertTrue(torch.equal(
            inputs[1] + dose * inputs[2], inputs[1] + inputs[2]))

    def test_route_is_batch_partition_and_order_independent(self):
        nn.init.normal_(self.head.dabr_router[-1].weight, std=0.2)
        nn.init.normal_(self.head.dabr_router[-1].bias, std=0.1)
        inputs = self.inputs()
        full = self.route(inputs)
        split = 5
        left = self.route(tuple(item[:split] for item in inputs))
        right = self.route(tuple(item[split:] for item in inputs))
        for full_item, left_item, right_item in zip(full, left, right):
            self.assertTrue(torch.allclose(
                full_item, torch.cat([left_item, right_item]),
                atol=1e-7, rtol=1e-6))
        permutation = torch.randperm(inputs[0].size(0))
        inverse = torch.argsort(permutation)
        permuted = self.route(tuple(item[permutation] for item in inputs))
        for full_item, permuted_item in zip(full, permuted):
            self.assertTrue(torch.allclose(
                full_item, permuted_item[inverse],
                atol=1e-7, rtol=1e-6))

    def test_counterfactual_targets_use_endpoints_and_weak_neutral(self):
        count = 8
        z_base = torch.zeros(count, 2)
        residual = torch.zeros_like(z_base)
        residual[:, 1] = 0.5
        labels = torch.tensor([1, 0] * (count // 2))
        (_, oracle, _, strong, target) = (
            self.head._dabr_counterfactual_targets(
                z_base, residual, labels))
        self.assertTrue(strong.all())
        self.assertTrue(torch.equal(oracle[labels == 1],
                                    torch.full_like(oracle[labels == 1], 2)))
        self.assertTrue(torch.equal(oracle[labels == 0],
                                    torch.zeros_like(oracle[labels == 0])))
        self.assertTrue(torch.equal(target, oracle))

        zero_residual = torch.zeros_like(residual)
        (_, _, gap, strong, target) = (
            self.head._dabr_counterfactual_targets(
                z_base, zero_residual, labels))
        self.assertTrue(torch.equal(gap, torch.zeros_like(gap)))
        self.assertFalse(strong.any())
        self.assertTrue(torch.equal(target, torch.ones_like(target)))

    def test_action_groups_have_equal_supervision_mass(self):
        target = torch.tensor([0, 0, 1, 1, 1, 2])
        sample_weight = torch.tensor([1.0, 6.0, 1.0, 2.0, 6.0, 6.0])
        balanced = self.head._dabr_group_balanced_weights(
            target, sample_weight)
        mass = torch.stack([
            balanced[target == idx].sum() for idx in range(3)
        ])
        self.assertAlmostEqual(balanced.mean().item(), 1.0, places=6)
        self.assertTrue(torch.allclose(
            mass, torch.full_like(mass, 2.0), atol=1e-6))

    def test_router_construction_preserves_global_rng(self):
        torch.manual_seed(211)
        expected = torch.rand(7)
        torch.manual_seed(211)
        router = HeteroGNNEdgeHead._dabr_make_router(self.dim_in, 8)
        actual = torch.rand(7)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(
            router[-1].weight, torch.zeros_like(router[-1].weight)))


if __name__ == '__main__':
    unittest.main()
