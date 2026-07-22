import copy
import inspect
import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import HeteroGNNEdgeHead


class _DummyCPTR(HeteroGNNEdgeHead):
    def __init__(self):
        nn.Module.__init__(self)
        self.cptr_max_correction = 0.10
        self.cptr_boundary_temperature = 0.10
        self.cptr_soft_f1_temperature = 0.10
        self.cptr_adapter_weight = 0.05
        self.cptr_safe_weight = 1.0
        self.cptr_safe_margin = 0.0
        self.cptr_threshold_decay = 0.9
        self.cptr_uplift_decay = 0.9
        self.cptr_uplift_lcb_z = 1.0
        self.cptr_gate_scale = 0.01
        self.cptr_min_env_count = 4
        self.cptr_min_threshold_count = 4
        self.cptr_start_epoch = 0
        self._eg_cur_epoch = 20
        self.cptr_adapter = nn.Sequential(nn.Linear(2, 1))
        with torch.no_grad():
            self.cptr_adapter[0].weight.copy_(torch.tensor([[0.7, -0.3]]))
            self.cptr_adapter[0].bias.fill_(0.2)
        self.register_buffer(
            'cptr_env_boundaries', torch.tensor([10, 20]))
        self.register_buffer(
            'cptr_threshold_ema', torch.tensor([0.25]))
        self.register_buffer(
            'cptr_threshold_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('cptr_uplift_ema', torch.zeros(3))
        self.register_buffer('cptr_uplift_variance', torch.zeros(3))
        self.register_buffer(
            'cptr_uplift_count', torch.zeros(3, dtype=torch.long))
        self._cptr_diag = None

    def make_ready(self, uplift=0.04):
        self.cptr_threshold_count.fill_(8)
        self.cptr_uplift_count.fill_(8)
        self.cptr_uplift_ema.fill_(uplift)
        self.cptr_uplift_variance.fill_(1e-6)


class CPTRTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = _DummyCPTR()
        self.features = torch.randn(9, 2)
        margin = torch.linspace(-0.5, 0.7, 9)
        self.anchor = torch.stack([-0.5 * margin, 0.5 * margin], dim=-1)
        self.edge_ids = torch.tensor([2, 5, 9, 11, 15, 19, 21, 25, 29])

    def test_exact_a2_fallback_until_ready_and_for_nonpositive_lcb(self):
        self.head.cptr_uplift_count.fill_(8)
        self.head.cptr_uplift_ema.fill_(0.04)
        route = self.head._cptr_route(
            self.features, self.anchor, self.edge_ids)
        self.assertTrue(torch.equal(route['logits'], self.anchor))
        self.assertEqual(torch.count_nonzero(route['gate']), 0)

        self.head.cptr_threshold_count.fill_(8)
        self.head.cptr_uplift_ema.fill_(-0.01)
        route = self.head._cptr_route(
            self.features, self.anchor, self.edge_ids)
        self.assertTrue(torch.equal(route['logits'], self.anchor))
        self.assertEqual(torch.count_nonzero(route['gate']), 0)

    def test_boundary_weight_is_localized_at_training_threshold(self):
        margins = torch.tensor([[0.25], [-1.75], [2.25]])
        weights = self.head._cptr_boundary_weight(
            margins, torch.tensor(0.25)).view(-1)
        self.assertEqual(weights[0].item(), 1.0)
        self.assertLess(weights[1].item(), 1e-6)
        self.assertLess(weights[2].item(), 1e-6)

    def test_deployment_gate_has_no_label_or_batch_statistic_input(self):
        route_parameters = inspect.signature(
            HeteroGNNEdgeHead._cptr_route).parameters
        gate_parameters = inspect.signature(
            HeteroGNNEdgeHead._cptr_deploy_gate).parameters
        self.assertNotIn('labels', route_parameters)
        self.assertEqual(set(gate_parameters), {'self', 'edge_ids'})

        self.head.make_ready()
        before = self.head._cptr_route(
            self.features, self.anchor, self.edge_ids)['logits']
        self.head._cptr_update_train_buffers(
            [torch.tensor(0.5)], [(0, torch.tensor(-0.2))])
        after = self.head._cptr_route(
            self.features, self.anchor, self.edge_ids)['logits']
        self.assertFalse(torch.equal(before, after))

        self.head.eval()
        threshold_before = self.head.cptr_threshold_ema.clone()
        uplift_before = self.head.cptr_uplift_ema.clone()
        self.head._cptr_update_train_buffers(
            [torch.tensor(-5.0)], [(1, torch.tensor(-5.0))])
        self.assertTrue(torch.equal(
            threshold_before, self.head.cptr_threshold_ema))
        self.assertTrue(torch.equal(uplift_before, self.head.cptr_uplift_ema))

    def test_inference_is_invariant_to_order_and_partition(self):
        self.head.make_ready()
        whole = self.head._cptr_route(
            self.features, self.anchor, self.edge_ids)['logits']

        permutation = torch.tensor([5, 1, 8, 0, 4, 2, 7, 3, 6])
        permuted = self.head._cptr_route(
            self.features[permutation], self.anchor[permutation],
            self.edge_ids[permutation])['logits']
        restored = torch.empty_like(permuted)
        restored[permutation] = permuted
        self.assertTrue(torch.allclose(whole, restored, atol=1e-7, rtol=0.0))

        pieces = []
        for indices in (slice(0, 2), slice(2, 6), slice(6, 9)):
            pieces.append(self.head._cptr_route(
                self.features[indices], self.anchor[indices],
                self.edge_ids[indices])['logits'])
        self.assertTrue(torch.allclose(
            whole, torch.cat(pieces), atol=1e-7, rtol=0.0))

    def test_checkpoint_restores_adapter_and_all_train_buffers(self):
        self.head.make_ready()
        self.head.cptr_threshold_ema.fill_(0.37)
        self.head.cptr_uplift_ema.copy_(torch.tensor([0.02, 0.03, 0.04]))
        self.head.cptr_uplift_variance.copy_(
            torch.tensor([0.001, 0.002, 0.003]))
        state = copy.deepcopy(self.head.state_dict())
        restored = _DummyCPTR()
        restored.load_state_dict(state)

        for key in (
                'cptr_adapter.0.weight', 'cptr_env_boundaries',
                'cptr_threshold_ema', 'cptr_threshold_count',
                'cptr_uplift_ema', 'cptr_uplift_variance',
                'cptr_uplift_count'):
            self.assertIn(key, state)
            self.assertTrue(torch.equal(state[key], restored.state_dict()[key]))

    def test_stratified_split_is_order_invariant_and_bidirectional(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        edge_ids = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1])
        split_a, split_b, valid = self.head._cptr_stratified_masks(
            labels, edge_ids)
        self.assertTrue(valid)
        self.assertFalse(torch.any(split_a & split_b))
        self.assertTrue(torch.all(split_a | split_b))
        self.assertEqual(labels[split_a].unique().numel(), 2)
        self.assertEqual(labels[split_b].unique().numel(), 2)

        permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
        pa, pb, pvalid = self.head._cptr_stratified_masks(
            labels[permutation], edge_ids[permutation])
        self.assertTrue(pvalid)
        self.assertEqual(
            set(edge_ids[split_a].tolist()),
            set(edge_ids[permutation][pa].tolist()))
        self.assertEqual(
            set(edge_ids[split_b].tolist()),
            set(edge_ids[permutation][pb].tolist()))

    def test_bidirectional_transfer_trains_adapter_and_updates_two_thresholds(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        edge_ids = torch.tensor([1, 2, 4, 5, 11, 12, 15, 16, 21, 22])
        features = torch.randn(10, 2)
        margin = torch.linspace(-0.4, 0.5, 10)
        anchor = torch.stack([-0.5 * margin, 0.5 * margin], dim=-1)
        route = self.head._cptr_route(features, anchor, edge_ids)
        loss = self.head._cptr_adapter_objective(route, labels, edge_ids)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            self.head.cptr_adapter[0].weight.grad.abs().sum().item(), 0.0)
        self.assertEqual(self.head.cptr_threshold_count.item(), 2)


if __name__ == '__main__':
    unittest.main()
