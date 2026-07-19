import unittest

import torch
import torch.nn as nn

from fraudGT.graphgym.config import cfg
from fraudGT.head.hetero_edge import HeteroGNNEdgeHead, TDSCAREditor
from fraudGT.train.custom_train import _td_scar_aux_loss


class TDSCAREditorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.repr_dim = 4
        self.editor = TDSCAREditor(
            self.repr_dim, 1, hidden_dim=7, correction_bound=0.5)

    def inputs(self, count=9):
        return (
            torch.randn(count, self.repr_dim),
            torch.randn(count, 1),
            torch.randn(count, 1),
            torch.ones(count, 1),
            torch.ones(count, 1),
            torch.randn(count, 1),
            torch.randn(count, 1),
        )

    def test_construction_preserves_global_rng(self):
        torch.manual_seed(223)
        expected = torch.rand(8)
        torch.manual_seed(223)
        TDSCAREditor(self.repr_dim, 1, hidden_dim=7)
        actual = torch.rand(8)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_initialization_is_exact_anchor_fallback(self):
        inputs = self.inputs()
        output = self.editor(*inputs)
        self.assertTrue(torch.equal(output['final_logits'], inputs[-2]))
        self.assertTrue((output['route_index'] == 0).all())

    def test_signed_directions_are_norm_bounded(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.stable_direction.output.bias.fill_(3.0)
            self.editor.recent_direction.output.bias.fill_(-3.0)
        output = self.editor(*inputs)
        self.assertTrue((output['stable_direction'] > 0).all())
        self.assertTrue((output['recent_direction'] < 0).all())
        self.assertTrue(
            (output['stable_direction'].norm(dim=-1) <= 0.500001).all())
        self.assertTrue(
            (output['recent_direction'].norm(dim=-1) <= 0.500001).all())

    def test_router_selects_one_candidate_without_mixing(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.stable_direction.output.bias.fill_(0.2)
            self.editor.router.output.bias.copy_(torch.tensor([-2.0, 3.0, -2.0]))
        output = self.editor(*inputs)
        expected = inputs[-2] + output['stable_direction']
        self.assertTrue(torch.equal(output['final_logits'], expected))
        self.assertTrue((output['route_index'] == 1).all())

    def _memory_head(self):
        head = HeteroGNNEdgeHead.__new__(HeteroGNNEdgeHead)
        nn.Module.__init__(head)
        head.td_scar_recent_tau = 10.0
        head.td_scar_evidence_scale = 2.0
        head._td_scar_cur_epoch = 0
        head.register_buffer('td_scar_recent_bank', torch.zeros(2, 2))
        head.register_buffer('td_scar_recent_ready', torch.zeros(2))
        head.register_buffer('td_scar_epoch_sum', torch.zeros(2, 2))
        head.register_buffer('td_scar_epoch_weight', torch.zeros(2))
        head.register_buffer(
            'td_scar_epoch_max_time', torch.tensor(float('-inf')))
        head.register_buffer('td_scar_memory_epoch', torch.tensor(-1))
        return head

    def test_recent_memory_is_visible_only_next_epoch(self):
        head = self._memory_head()
        head._td_scar_prepare_epoch()
        representations = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        labels = torch.tensor([1, 0])
        timestamps = torch.tensor([10.0, 5.0])
        head._td_scar_accumulate_recent(representations, labels, timestamps)
        _, current_ready = head._td_scar_recent_evidence(representations)
        self.assertEqual(current_ready.max().item(), 0.0)
        head._td_scar_cur_epoch = 1
        head._td_scar_prepare_epoch()
        evidence, next_ready = head._td_scar_recent_evidence(
            representations[:1])
        self.assertEqual(next_ready.min().item(), 1.0)
        self.assertGreater(evidence.item(), 0.0)

    def test_stable_evidence_retains_sign(self):
        head = self._memory_head()
        head.num_class_proto_slots = 2
        head.support_class_proto_temperature = nn.Parameter(torch.tensor(0.0))
        head.register_buffer('support_class_proto_bank', torch.tensor([
            [[-1.0, 0.0], [-1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]))
        head.register_buffer('support_class_proto_ready', torch.ones(2, 2))
        positive, ready = head._td_scar_stable_evidence(
            torch.tensor([[1.0, 0.0]]))
        negative, _ = head._td_scar_stable_evidence(
            torch.tensor([[-1.0, 0.0]]))
        self.assertEqual(ready.item(), 1.0)
        self.assertGreater(positive.item(), 0.0)
        self.assertLess(negative.item(), 0.0)

    def test_auxiliary_loss_cannot_update_anchor(self):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor([
                    [0.1, -0.1], [-0.2, 0.2]]))
                self.stable = nn.Parameter(torch.full((2, 2), 0.05))
                self.recent = nn.Parameter(torch.full((2, 2), -0.04))
                self.route = nn.Parameter(torch.zeros(2, 3))
                self._td_scar_stable_logits = self.anchor.detach() + self.stable
                self._td_scar_recent_logits = self.anchor.detach() + self.recent
                self._td_scar_labels = torch.tensor([1, 0])
                self._td_scar_route_logits = self.route
                self._td_scar_route_target = torch.tensor([1, 2])
                self._td_scar_stable_direction = self.stable
                self._td_scar_recent_direction = self.recent
                self.td_scar_direction_loss_weight = 0.25
                self.td_scar_router_loss_weight = 0.10
                self.td_scar_norm_loss_weight = 1e-4

        holder = Holder()
        old_loss = cfg.model.loss_fun
        had_weight = hasattr(cfg.model, 'loss_fun_weight')
        old_weight = cfg.model.loss_fun_weight if had_weight else None
        old_device = cfg.device
        try:
            cfg.model.loss_fun = 'weighted_cross_entropy'
            cfg.model.loss_fun_weight = [1, 6]
            cfg.device = 'cpu'
            _td_scar_aux_loss(holder).backward()
        finally:
            cfg.model.loss_fun = old_loss
            if had_weight:
                cfg.model.loss_fun_weight = old_weight
            else:
                cfg.model.pop('loss_fun_weight', None)
            cfg.device = old_device
        self.assertIsNone(holder.anchor.grad)
        self.assertGreater(holder.stable.grad.abs().sum().item(), 0.0)
        self.assertGreater(holder.recent.grad.abs().sum().item(), 0.0)
        self.assertGreater(holder.route.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
