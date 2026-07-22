import unittest

import torch

from fraudGT.train.custom_train import (
    _epra_hard_pair_rank_loss,
    _epra_proto_alignment_loss,
)


class EPRALossTest(unittest.TestCase):
    def test_rank_loss_pushes_hard_pairs_in_correct_direction(self):
        logits = torch.nn.Parameter(torch.zeros(4, 2))
        labels = torch.tensor([1, 1, 0, 0])

        loss = _epra_hard_pair_rank_loss(
            logits, labels, pair_limit=4, temperature=0.25,
            target_margin=0.5)
        loss.backward()

        margin_grad = logits.grad[:, 1] - logits.grad[:, 0]
        self.assertTrue((margin_grad[:2] < 0.0).all())
        self.assertTrue((margin_grad[2:] > 0.0).all())

    def test_rank_loss_has_finite_margin_stop(self):
        labels = torch.tensor([1, 1, 0, 0])
        weak = torch.tensor([[0.0, 0.0], [0.0, 0.0],
                             [0.0, 0.0], [0.0, 0.0]])
        strong = torch.tensor([[0.0, 2.0], [0.0, 2.0],
                               [0.0, -2.0], [0.0, -2.0]])

        weak_loss = _epra_hard_pair_rank_loss(
            weak, labels, 4, 0.25, 0.5)
        strong_loss = _epra_hard_pair_rank_loss(
            strong, labels, 4, 0.25, 0.5)

        self.assertGreater(weak_loss.item(), strong_loss.item() * 100.0)

    def test_bpra_hinge_has_exact_finite_stop(self):
        labels = torch.tensor([1, 1, 0, 0])
        logits = torch.nn.Parameter(torch.tensor([
            [0.0, 2.0], [0.0, 2.0], [0.0, -2.0], [0.0, -2.0]]))
        loss = _epra_hard_pair_rank_loss(
            logits, labels, 4, 0.25, 0.5, finite_hinge=True)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_rank_loss_is_scale_invariant_for_nontrivial_scores(self):
        labels = torch.tensor([1, 1, 0, 0])
        logits = torch.tensor([[0.0, 4.0], [0.0, 2.0],
                               [0.0, -2.0], [0.0, -4.0]])
        base = _epra_hard_pair_rank_loss(
            logits, labels, 4, 0.25, 0.5)
        scaled = _epra_hard_pair_rank_loss(
            logits * 10.0, labels, 4, 0.25, 0.5)
        self.assertAlmostEqual(base.item(), scaled.item(), places=6)

    def test_single_class_batch_adds_no_auxiliary_gradient(self):
        logits = torch.nn.Parameter(torch.zeros(4, 2))
        labels = torch.zeros(4, dtype=torch.long)
        rank_loss = _epra_hard_pair_rank_loss(
            logits, labels, 4, 0.25, 0.5)
        rank_loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

        edge_repr = torch.nn.Parameter(torch.randn(4, 3))
        proto_loss = _epra_proto_alignment_loss(
            edge_repr, torch.randn(4, 3), torch.randn(4, 3),
            torch.ones(4, 1), labels, 0.2)
        proto_loss.backward()
        self.assertTrue(torch.equal(
            edge_repr.grad, torch.zeros_like(edge_repr)))

    def test_prototype_alignment_rewards_correct_class_geometry(self):
        labels = torch.tensor([1, 1, 0, 0])
        edge_repr = torch.tensor([[1.0, 0.0], [1.0, 0.1],
                                  [-1.0, 0.0], [-1.0, 0.1]])
        pos_proto = torch.tensor([[1.0, 0.0]]).expand_as(edge_repr)
        neg_proto = torch.tensor([[-1.0, 0.0]]).expand_as(edge_repr)
        ready = torch.ones(4, 1)

        aligned = _epra_proto_alignment_loss(
            edge_repr, pos_proto, neg_proto, ready, labels, 0.2)
        reversed_loss = _epra_proto_alignment_loss(
            -edge_repr, pos_proto, neg_proto, ready, labels, 0.2)

        self.assertLess(aligned.item(), reversed_loss.item())

    def test_bpra_single_class_batch_uses_ready_prototypes(self):
        labels = torch.zeros(4, dtype=torch.long)
        edge_repr = torch.nn.Parameter(torch.randn(4, 3))
        proto_loss = _epra_proto_alignment_loss(
            edge_repr, torch.randn(4, 3), torch.randn(4, 3),
            torch.ones(4, 1), labels, 0.2, allow_single_class=True)
        proto_loss.backward()
        self.assertGreater(proto_loss.item(), 0.0)
        self.assertGreater(edge_repr.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
