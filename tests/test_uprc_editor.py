import unittest

import torch
import torch.nn as nn

from fraudGT.head.hetero_edge import UPRCEditor
from fraudGT.train.custom_train import (
    _uprc_aux_loss, _uprc_hard_pair_rank_loss)


class UPRCEditorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.repr_dim = 5
        self.editor = UPRCEditor(
            self.repr_dim, 1, hidden_dim=8, correction_bound=0.25,
            locality_floor=0.10)

    def inputs(self, count=10):
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
        torch.manual_seed(229)
        expected = torch.rand(8)
        torch.manual_seed(229)
        UPRCEditor(self.repr_dim, 1, hidden_dim=8)
        actual = torch.rand(8)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_initialization_is_exact_a2_fallback(self):
        inputs = self.inputs()
        output = self.editor(*inputs)
        self.assertTrue(torch.equal(output['final_logits'], inputs[-2]))
        self.assertTrue(torch.equal(
            output['correction'], torch.zeros_like(output['correction'])))

    def test_signed_correction_is_bounded(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(3.0)
        positive = self.editor(*inputs)['correction']
        self.assertTrue((positive > 0).all())
        self.assertTrue((positive.norm(dim=-1) <= 0.250001).all())
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(-3.0)
        negative = self.editor(*inputs)['correction']
        self.assertTrue((negative < 0).all())
        self.assertTrue((negative.norm(dim=-1) <= 0.250001).all())

    def test_smooth_bound_retains_gradient_for_large_direction(self):
        inputs = list(self.inputs())
        inputs[-2] = torch.zeros_like(inputs[-2])
        with torch.no_grad():
            self.editor.direction.output.bias.fill_(100.0)
        correction = self.editor(*inputs)['correction']
        correction.sum().backward()
        gradient = self.editor.direction.output.bias.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().item(), 0.0)

    def test_partition_and_order_do_not_change_outputs(self):
        inputs = self.inputs()
        with torch.no_grad():
            self.editor.direction.output.weight.normal_(std=0.1)
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

    def test_pairwise_loss_pushes_positive_up_and_negative_down(self):
        anchor = torch.tensor([[-0.2], [-0.1], [0.1], [0.2]],
                              requires_grad=True)
        correction = nn.Parameter(torch.zeros_like(anchor))
        candidate = anchor.detach() + correction
        labels = torch.tensor([1, 1, 0, 0])
        loss = _uprc_hard_pair_rank_loss(
            anchor, candidate, labels, pair_limit=4, temperature=0.25)
        loss.backward()
        self.assertIsNone(anchor.grad)
        self.assertTrue((correction.grad[:2] < 0).all())
        self.assertTrue((correction.grad[2:] > 0).all())

    def test_auxiliary_loss_cannot_update_anchor(self):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor([
                    [-0.2], [-0.1], [0.1], [0.2]]))
                self.correction = nn.Parameter(torch.zeros(4, 1))
                self._uprc_anchor_logits = self.anchor
                self._uprc_candidate_logits = (
                    self.anchor.detach() + self.correction)
                self._uprc_labels = torch.tensor([1, 1, 0, 0])
                self._uprc_correction = self.correction
                self.uprc_pair_limit = 4
                self.uprc_rank_temperature = 0.25
                self.uprc_rank_loss_weight = 1.0
                self.uprc_balanced_loss_weight = 0.25
                self.uprc_center_loss_weight = 0.01
                self.uprc_norm_loss_weight = 1e-3

        holder = Holder()
        _uprc_aux_loss(holder).backward()
        self.assertIsNone(holder.anchor.grad)
        self.assertGreater(holder.correction.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
