import copy
import inspect
import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead


class CostarRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.dim_in = 6
        self.head = self.make_head()

    def make_head(self):
        head = HeteroGNNEdgeHead.__new__(HeteroGNNEdgeHead)
        nn.Module.__init__(head)
        head.costar_router = nn.Sequential(
            nn.Linear(self.dim_in + 8, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(head.costar_router[-1].weight)
        nn.init.zeros_(head.costar_router[-1].bias)
        head.costar_router_ema = copy.deepcopy(head.costar_router)
        for parameter in head.costar_router_ema.parameters():
            parameter.requires_grad_(False)
        head.register_buffer('costar_center_numerator', torch.zeros(1))
        head.register_buffer('costar_center_denominator', torch.zeros(1))
        head.register_buffer(
            'costar_center_updates', torch.zeros((), dtype=torch.long))
        head.register_buffer(
            'costar_ema_updates', torch.zeros((), dtype=torch.long))
        head.costar_ema_decay = 0.9
        head.costar_center_decay = 0.9
        head.costar_consistency_tau = 0.25
        head.costar_soft_f1_temperature = 0.1
        head.costar_threshold_perturb = 0.1
        head.costar_platform_tolerance = 0.01
        head.costar_cvar_fraction = 0.5
        head.costar_adapter_weight = 0.05
        head.costar_rank_weight = 0.25
        head.costar_safe_weight = 1.0
        head.costar_orth_weight = 0.1
        head.costar_time_weight = 0.05
        head.costar_safe_margin = 0.0
        head.costar_rank_safe_margin = 0.0
        head.costar_start_epoch = 10
        head.costar_fallback_confidence = 0.25
        head.costar_fallback_delta = 1e-4
        head._eg_cur_epoch = 20
        head._costar_diag = None
        return head

    def inputs(self, count=14, requires_grad=False):
        h = torch.randn(count, self.dim_in, requires_grad=requires_grad)
        z_base = torch.randn(count, 2, requires_grad=requires_grad)
        effective_residual = (
            0.2 * torch.randn(count, 2)).requires_grad_(requires_grad)
        z_anchor = z_base + effective_residual
        pos_sim = torch.rand(count, 1)
        neg_sim = torch.rand(count, 1)
        proto_margin = pos_sim - neg_sim
        ready = torch.ones(count, 1)
        labels = torch.arange(count) % 2
        return (
            h, z_base, z_anchor, effective_residual,
            pos_sim, neg_sim, proto_margin, ready, labels)

    def route(self, inputs):
        return self.head._costar_route(*inputs[:-1])

    def randomize_routers(self):
        with torch.no_grad():
            nn.init.normal_(self.head.costar_router[-1].weight, std=0.2)
            nn.init.normal_(self.head.costar_router[-1].bias, std=0.1)
            nn.init.normal_(self.head.costar_router_ema[-1].weight, std=0.15)
            nn.init.normal_(self.head.costar_router_ema[-1].bias, std=0.05)

    def test_zero_router_is_exact_a2_fallback_and_has_no_dropout(self):
        inputs = self.inputs()
        route = self.route(inputs)
        self.assertTrue(torch.equal(route['logits'], inputs[2]))
        self.assertTrue(torch.equal(
            route['correction_margin'],
            torch.zeros_like(route['correction_margin'])))
        self.assertEqual(route['fallback_ratio'].item(), 1.0)
        self.assertTrue(route['fallback_mask'].all())
        self.assertTrue(torch.equal(
            route['correction_margin'],
            route['residual_weight'] * route['evidence_margin']))
        self.assertFalse(any(
            isinstance(module, nn.Dropout)
            for module in self.head.costar_router.modules()))
        self.assertNotIn(
            'labels', inspect.signature(self.head._costar_route).parameters)

    def test_inference_is_independent_of_batch_partition_and_order(self):
        self.randomize_routers()
        self.head.costar_center_numerator.fill_(0.02)
        self.head.costar_center_denominator.fill_(0.20)
        self.head.costar_center_updates.fill_(3)
        self.head.eval()
        inputs = self.inputs(count=15)
        full = self.route(inputs)

        split = 6
        left = self.route(tuple(item[:split] for item in inputs))
        right = self.route(tuple(item[split:] for item in inputs))
        for key in ('logits', 'current_value', 'ema_value', 'consistency',
                    'residual_weight', 'correction_margin'):
            partitioned = torch.cat([left[key], right[key]], dim=0)
            self.assertTrue(torch.allclose(
                full[key], partitioned, atol=1e-7, rtol=1e-6), key)

        permutation = torch.randperm(inputs[0].size(0))
        inverse = torch.argsort(permutation)
        permuted = self.route(tuple(item[permutation] for item in inputs))
        for key in ('logits', 'current_value', 'ema_value', 'consistency',
                    'residual_weight', 'correction_margin'):
            self.assertTrue(torch.allclose(
                full[key], permuted[key][inverse],
                atol=1e-7, rtol=1e-6), key)
        self.assertTrue(torch.equal(
            full['fallback_mask'],
            torch.cat([
                left['fallback_mask'], right['fallback_mask']], dim=0)))
        self.assertTrue(torch.equal(
            full['fallback_mask'], permuted['fallback_mask'][inverse]))

    def test_adapter_loss_does_not_update_a2_inputs(self):
        self.randomize_routers()
        inputs = self.inputs(requires_grad=True)
        route = self.route(inputs)
        loss = self.head._costar_adapter_objective(
            inputs[2], route, inputs[-1])
        loss.backward()

        self.assertIsNone(inputs[0].grad)
        self.assertIsNone(inputs[1].grad)
        self.assertIsNone(inputs[3].grad)
        router_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.head.costar_router.parameters()
            if parameter.grad is not None)
        self.assertGreater(router_grad, 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_ema_and_center_are_checkpoint_state(self):
        self.randomize_routers()
        self.head.costar_center_numerator.fill_(0.12)
        self.head.costar_center_denominator.fill_(0.34)
        self.head.costar_center_updates.fill_(7)
        self.head.costar_update_ema()
        state = copy.deepcopy(self.head.state_dict())

        restored = self.make_head()
        restored.load_state_dict(state)
        self.assertEqual(restored.costar_ema_updates.item(), 1)
        self.assertEqual(restored.costar_center_updates.item(), 7)
        self.assertTrue(torch.equal(
            restored.costar_center_numerator,
            self.head.costar_center_numerator))
        for left, right in zip(
                restored.costar_router_ema.parameters(),
                self.head.costar_router_ema.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertIn('costar_router_ema.0.weight', state)
        self.assertIn('costar_center_numerator', state)

    def test_exact_train_moment_makes_correction_orthogonal(self):
        self.randomize_routers()
        inputs = self.inputs(count=18)
        features, evidence = self.head._costar_router_features(
            *inputs[:-1])
        current, ema = self.head._costar_router_values(features)
        consistency = torch.exp(
            -(current - ema).abs() / self.head.costar_consistency_tau)
        consensus = consistency * 0.5 * (current + ema)
        evidence_square = evidence.square()
        self.head.costar_center_numerator.copy_(
            (consensus * evidence_square).mean().detach().view(1))
        self.head.costar_center_denominator.copy_(
            evidence_square.mean().detach().view(1))
        self.head.costar_center_updates.fill_(1)

        route = self.route(inputs)
        inner = (
            route['correction_margin'] * route['evidence_margin']).mean()
        self.assertLess(inner.abs().item(), 1e-7)
        self.assertLess(route['orth_error'].item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
