import unittest

import torch

from fraudGT.cdvt.event_graph import CausalEventGraphIndex
from fraudGT.cdvt.fusion import DualViewFusionClassifier


class CDVTModelComponentsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        edges = torch.tensor([
            [0, 1, 0, 2, 1, 0, 3],
            [1, 2, 2, 0, 0, 3, 0],
        ])
        times = torch.tensor([1, 2, 3, 3, 4, 5, 6])
        raw = torch.tensor([
            [1, 1.0, 0, 0],
            [2, 2.0, 0, 0],
            [3, 3.0, 1, 0],
            [3, 4.0, 1, 1],
            [4, 5.0, 0, 1],
            [5, 6.0, 0, 0],
            [6, 7.0, 1, 1],
        ])
        self.graph = CausalEventGraphIndex(
            edges, times, raw).query(torch.tensor([4, 6]), k=3, hops=2)

    def test_dual_view_fuses_before_classification_and_backpropagates(self):
        model = DualViewFusionClassifier(
            account_dim=48,
            num_currencies=2,
            num_payment_formats=2,
            hidden_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        account = torch.randn(2, 48, requires_grad=True)
        logits, diagnostics = model(account, self.graph)
        self.assertEqual(logits.shape, (2,))
        self.assertGreater(float(diagnostics["fusion_gain_norm"].sum()), 0)
        logits.sum().backward()
        self.assertIsNotNone(account.grad)
        self.assertIsNotNone(model.cross_projection.weight.grad)
        self.assertIsNotNone(
            model.event_encoder.layers[0].edge_key.weight.grad)

    def test_event_only_is_independent_of_account_representation(self):
        model = DualViewFusionClassifier(
            account_dim=48,
            num_currencies=2,
            num_payment_formats=2,
            variant="event_only",
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()
        self.assertFalse(hasattr(model, "account_projection"))
        self.assertFalse(hasattr(model, "cross_attention"))
        first, _ = model(torch.randn(2, 48), self.graph)
        second, _ = model(torch.randn(2, 48), self.graph)
        self.assertTrue(torch.equal(first, second))

    def test_normal_shuffled_and_off_are_distinct_interventions(self):
        model = DualViewFusionClassifier(
            account_dim=48,
            num_currencies=2,
            num_payment_formats=2,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()
        account = torch.randn(2, 48)
        normal, _ = model(account, self.graph)
        torch.manual_seed(7)
        shuffled, _ = model(account, self.graph.shuffled())
        off, diagnostics = model(account, self.graph.off())
        self.assertFalse(torch.equal(normal, shuffled))
        self.assertFalse(torch.equal(normal, off))
        self.assertTrue(torch.equal(
            diagnostics["event_count"], torch.ones(2, dtype=torch.long)))

    def test_all_message_edges_follow_causal_order(self):
        source, destination = self.graph.edge_index
        source_delta = self.graph.node_target_delta[source]
        destination_delta = self.graph.node_target_delta[destination]
        self.assertTrue((source_delta >= destination_delta).all())
        tied = source_delta == destination_delta
        self.assertTrue((
            self.graph.node_edge_ids[source[tied]]
            < self.graph.node_edge_ids[destination[tied]]
        ).all())

    def test_event_encoder_outputs_are_finite(self):
        model = DualViewFusionClassifier(
            account_dim=48,
            num_currencies=2,
            num_payment_formats=2,
            variant="event_only",
            hidden_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        model.eval()
        logits, diagnostics = model(torch.zeros(2, 48), self.graph)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(
            diagnostics["target_event_norm"]).all())


if __name__ == "__main__":
    unittest.main()
