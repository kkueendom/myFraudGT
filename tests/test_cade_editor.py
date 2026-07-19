import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import CADEEditor
from fraudGT.graphgym.config import cfg
from fraudGT.train.custom_train import _cade_aux_loss


class CadeEditorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.repr_dim = 5
        self.editor = CADEEditor(
            self.repr_dim, 1, hidden_dim=8, correction_bound=0.5)

    def inputs(self, count=11):
        h = torch.randn(count, self.repr_dim)
        pos = torch.randn(count, self.repr_dim)
        neg = torch.randn(count, self.repr_dim)
        pos_sim = torch.rand(count, 1)
        neg_sim = torch.rand(count, 1)
        margin = pos_sim - neg_sim
        ready = torch.ones(count, 1)
        anchor = torch.randn(count, 1)
        base = torch.randn(count, 1)
        return h, pos, neg, pos_sim, neg_sim, margin, ready, anchor, base

    def test_construction_preserves_global_rng(self):
        torch.manual_seed(211)
        expected = torch.rand(9)
        torch.manual_seed(211)
        CADEEditor(self.repr_dim, 1, hidden_dim=8, correction_bound=0.5)
        actual = torch.rand(9)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_initialization_is_exact_a2_fallback(self):
        inputs = self.inputs()
        output = self.editor(*inputs)
        self.assertTrue(torch.equal(output['final_logits'], inputs[-2]))
        self.assertTrue(torch.equal(
            output['direction'], torch.zeros_like(output['direction'])))
        self.assertTrue(torch.equal(
            output['quality_weight'],
            torch.zeros_like(output['quality_weight'])))

    def test_direction_is_signed_and_norm_bounded(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(2.0)
        positive = self.editor(*inputs)['direction']
        self.assertTrue((positive > 0).all())
        self.assertTrue((positive.norm(dim=-1) <= 0.500001).all())
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(-2.0)
        negative = self.editor(*inputs)['direction']
        self.assertTrue((negative < 0).all())
        self.assertTrue((negative.norm(dim=-1) <= 0.500001).all())

    def test_batch_partition_and_order_do_not_change_outputs(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.direction.output.weight.normal_(std=0.1)
            self.editor.quality.output.weight.normal_(std=0.1)
        full = self.editor(*inputs)['final_logits']
        split = 4
        partitioned = torch.cat([
            self.editor(*(item[:split] for item in inputs))['final_logits'],
            self.editor(*(item[split:] for item in inputs))['final_logits'],
        ])
        self.assertTrue(torch.allclose(full, partitioned, atol=1e-7, rtol=1e-6))
        permutation = torch.randperm(full.size(0))
        inverse = torch.argsort(permutation)
        permuted = self.editor(
            *(item[permutation] for item in inputs))['final_logits']
        self.assertTrue(torch.allclose(
            full, permuted[inverse], atol=1e-7, rtol=1e-6))

    def test_direction_branch_detaches_anchor_features(self):
        inputs = list(self.inputs())
        inputs[0].requires_grad_(True)
        inputs[-2].requires_grad_(True)
        direction = self.editor(*inputs)['direction']
        direction.sum().backward()
        self.assertIsNone(inputs[0].grad)
        self.assertIsNone(inputs[-2].grad)

    def test_negative_quality_is_exact_fallback(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(0.25)
            self.editor.quality.output.bias.fill_(-10.0)
        output = self.editor(*inputs)
        self.assertTrue(torch.equal(output['final_logits'], inputs[-2]))

    def test_auxiliary_loss_cannot_update_anchor_parameter(self):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor([[0.2], [-0.1]]))
                self.branch = nn.Parameter(torch.tensor([[0.05], [-0.05]]))
                self.quality = nn.Parameter(torch.tensor([[0.3], [-0.3]]))
                self._cade_aux_logits = self.anchor.detach() + self.branch
                self._cade_aux_labels = torch.tensor([1, 0])
                self._cade_quality_logit = self.quality
                self._cade_quality_target = torch.tensor([[0.8], [0.2]])
                self._cade_direction = self.branch
                self.cade_direction_loss_weight = 0.25
                self.cade_quality_loss_weight = 0.10
                self.cade_norm_loss_weight = 1e-4

        holder = Holder()
        old_loss = cfg.model.loss_fun
        old_weight = cfg.model.loss_fun_weight
        old_device = cfg.device
        try:
            cfg.model.loss_fun = 'weighted_cross_entropy'
            cfg.model.loss_fun_weight = [1, 6]
            cfg.device = 'cpu'
            loss = _cade_aux_loss(holder)
            loss.backward()
        finally:
            cfg.model.loss_fun = old_loss
            cfg.model.loss_fun_weight = old_weight
            cfg.device = old_device
        self.assertIsNone(holder.anchor.grad)
        self.assertIsNotNone(holder.branch.grad)
        self.assertGreater(holder.branch.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(holder.quality.grad)


if __name__ == '__main__':
    unittest.main()
