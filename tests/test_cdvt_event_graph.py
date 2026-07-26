import unittest

import torch

from fraudGT.cdvt.event_graph import CausalEventGraphIndex


class CausalEventGraphIndexTest(unittest.TestCase):
    def setUp(self):
        self.edges = torch.tensor([
            [0, 1, 0, 2, 1, 0, 3],
            [1, 2, 2, 0, 0, 3, 0],
        ])
        self.times = torch.tensor([1, 2, 3, 3, 4, 5, 6])
        self.raw = torch.tensor([
            [1, 1.0, 0, 0],
            [2, 2.0, 0, 0],
            [3, 3.0, 1, 0],
            [3, 4.0, 1, 1],
            [4, 5.0, 0, 1],
            [5, 6.0, 0, 0],
            [6, 7.0, 1, 1],
        ])
        self.index = CausalEventGraphIndex(
            self.edges, self.times, self.raw)

    def test_target_receives_only_causal_predecessors(self):
        graph = self.index.query(torch.tensor([4]), k=3, hops=2)
        target = int(graph.target_nodes[0])
        incoming = graph.edge_index[0, graph.edge_index[1] == target]
        predecessor_ids = graph.node_edge_ids[incoming]
        self.assertTrue((self.times[predecessor_ids] <= 4).all())
        self.assertNotIn(4, predecessor_ids.tolist())
        self.assertNotIn(5, graph.node_edge_ids.tolist())
        self.assertNotIn(6, graph.node_edge_ids.tolist())

    def test_equal_timestamp_uses_edge_id_tie_break(self):
        earlier = self.index.query(torch.tensor([2]), k=4, hops=1)
        later = self.index.query(torch.tensor([3]), k=4, hops=1)
        self.assertNotIn(3, earlier.node_edge_ids.tolist())
        self.assertIn(2, later.node_edge_ids.tolist())

    def test_latest_k_is_applied_per_account(self):
        graph = self.index.query(
            torch.tensor([6]), k=1, hops=1, max_events=20)
        context = set(graph.node_edge_ids.tolist()) - {6}
        self.assertEqual(context, {5})

    def test_relations_and_transition_features_are_explicit(self):
        graph = self.index.query(torch.tensor([4]), k=4, hops=2)
        self.assertEqual(graph.edge_attr.size(1), 10)
        self.assertTrue((graph.edge_relation >= 0).all())
        self.assertTrue((graph.edge_relation < 4).all())
        self.assertTrue(torch.isfinite(graph.edge_attr).all())
        self.assertGreater(int(torch.unique(graph.edge_relation).numel()), 1)

    def test_off_keeps_only_target_query_nodes(self):
        graph = self.index.query(torch.tensor([4, 6]), k=2, hops=2)
        off = graph.off()
        self.assertEqual(off.node_raw.shape, (2, 4))
        self.assertEqual(off.edge_index.numel(), 0)
        self.assertTrue(torch.equal(
            off.node_edge_ids, torch.tensor([4, 6])))

    def test_labels_are_not_an_input(self):
        graph = self.index.query(torch.tensor([4]), k=2, hops=2)
        self.assertEqual(graph.node_raw.size(1), 4)
        self.assertFalse(hasattr(graph, "labels"))


if __name__ == "__main__":
    unittest.main()
