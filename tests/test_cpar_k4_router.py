import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead
from fraudGT.train.custom_train import (
    _clip_gradients,
    _cpar_training_anchor,
)


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
        self.head.cpar_gap_floor = 1e-4
        self.head.cpar_gap_threshold = 0.005

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

    def test_learned_neutral_action_is_exact_a2_fallback(self):
        with torch.no_grad():
            self.head.cpar_router[-1].weight.zero_()
            self.head.cpar_router[-1].bias.copy_(
                torch.tensor([-12.0, -12.0, 12.0, -12.0]))
        inputs = self.inputs()
        _, action_prob, confidence, dose = self.route(inputs)
        self.assertTrue(torch.equal(
            action_prob.argmax(dim=-1),
            torch.full((inputs[0].size(0),), 2, dtype=torch.long)))
        self.assertTrue((confidence > 0.99).all())
        self.assertTrue(torch.equal(dose, torch.ones_like(dose)))
        self.assertTrue(torch.equal(
            inputs[1] + dose * inputs[2],
            inputs[1] + inputs[2]))
        dose.sum().backward()
        self.assertGreater(
            self.head.cpar_router[-1].bias.grad.abs().sum().item(), 0.0)

    def test_all_weak_gaps_abstain_to_neutral(self):
        count = 8
        z_base = torch.randn(count, 2)
        zero_residual = torch.zeros_like(z_base)
        labels = torch.arange(count) % 2
        (action_losses, _, normalized_gap,
         strong_mask, target_action) = (
            self.head._cpar_counterfactual_targets(
                z_base, zero_residual, labels))
        self.assertEqual(tuple(action_losses.shape), (count, 4))
        self.assertTrue(torch.equal(
            normalized_gap, torch.zeros_like(normalized_gap)))
        self.assertFalse(strong_mask.any())
        self.assertTrue(torch.equal(
            target_action, torch.full_like(target_action, 2)))

    def test_tiny_nonzero_oracle_gap_still_abstains(self):
        count = 6
        z_base = torch.zeros(count, 2)
        tiny_residual = torch.zeros_like(z_base)
        tiny_residual[:, 1] = 1e-4
        labels = torch.ones(count, dtype=torch.long)
        (_, oracle_action, normalized_gap,
         strong_mask, target_action) = (
            self.head._cpar_counterfactual_targets(
                z_base, tiny_residual, labels))
        self.assertTrue(torch.equal(
            oracle_action, torch.full_like(oracle_action, 3)))
        self.assertTrue((normalized_gap > 0.0).all())
        self.assertTrue(
            (normalized_gap < self.head.cpar_gap_threshold).all())
        self.assertFalse(strong_mask.any())
        self.assertTrue(torch.equal(
            target_action, torch.full_like(target_action, 2)))

    def test_final_action_groups_have_equal_supervision_mass(self):
        target_action = torch.tensor([2, 2, 2, 2, 0, 0, 3, 3, 3])
        sample_weights = torch.tensor([
            1.0, 6.0, 1.0, 6.0, 1.0, 6.0, 1.0, 2.0, 6.0])
        balanced = self.head._cpar_group_balanced_weights(
            target_action, sample_weights)
        self.assertAlmostEqual(balanced.mean().item(), 1.0, places=6)
        group_mass = torch.stack([
            balanced[target_action == idx].sum()
            for idx in (0, 2, 3)
        ])
        self.assertTrue(torch.allclose(
            group_mass, torch.full_like(group_mass, 3.0), atol=1e-6))
        self.assertAlmostEqual(
            (balanced[1] / balanced[0]).item(),
            (sample_weights[1] / sample_weights[0]).item(),
            places=6)

    def test_router_contains_no_dropout(self):
        self.assertFalse(any(
            isinstance(module, nn.Dropout)
            for module in self.head.cpar_router.modules()))

    def test_router_construction_preserves_global_rng(self):
        torch.manual_seed(101)
        expected = torch.rand(8)
        torch.manual_seed(101)
        router = HeteroGNNEdgeHead._cpar_make_router(self.dim_in, 8)
        actual = torch.rand(8)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(
            router[-1].weight, torch.zeros_like(router[-1].weight)))

    def test_anchor_lookup_and_separate_gradient_clipping(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor_weight = nn.Parameter(torch.tensor([3.0]))
                self.cpar_router = nn.Linear(1, 1, bias=False)
                self._cpar_anchor_logits = self.anchor_weight.view(1, 1)

        model = DummyModel()
        self.assertIs(_cpar_training_anchor(model),
                      model._cpar_anchor_logits)
        model.anchor_weight.grad = torch.tensor([6.0])
        model.cpar_router.weight.grad = torch.tensor([[8.0]])
        _clip_gradients(model, 1.0, separate_cpar=True)
        self.assertAlmostEqual(
            model.anchor_weight.grad.norm().item(), 1.0, places=6)
        self.assertAlmostEqual(
            model.cpar_router.weight.grad.norm().item(), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
