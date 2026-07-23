import unittest

import torch

from fraudGT.evidence.tier import (
    TemporalIncidentIndex,
    build_raw_edge_attributes,
    recover_train_normalized_raw_edge_attributes,
)


class RawEdgeAttributeTest(unittest.TestCase):
    def test_amount_normalization_uses_train_prefix_only(self):
        timestamps = torch.tensor([0, 1, 2, 3])
        amounts = torch.tensor([1.0, 3.0, 100.0, 1000.0])
        currencies = torch.tensor([0, 1, 1, 0])
        formats = torch.tensor([2, 2, 3, 3])
        raw = build_raw_edge_attributes(
            timestamps, amounts, currencies, formats, train_end=2)

        expected = amounts
        train_mean = expected[:2].mean()
        train_std = expected[:2].std(unbiased=False)
        self.assertTrue(torch.allclose(
            raw[:, 1], (expected - train_mean) / train_std))
        self.assertTrue(torch.equal(raw[:, 2], currencies.float()))
        self.assertTrue(torch.equal(raw[:, 3], formats.float()))

    def test_recovers_train_only_scale_from_legacy_split_zscores(self):
        timestamps = torch.arange(6)
        amount = torch.tensor([1.0, 3.0, 7.0, 9.0, 100.0, 1000.0])
        currency = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        payment = torch.tensor([0.0, 1.0, 0.0, 2.0, 1.0, 2.0])
        raw = torch.stack(
            (timestamps.float(), amount, currency, payment), dim=-1)

        def legacy_zscore(values):
            return (
                (values - values.mean(0, keepdim=True))
                / values.std(0, unbiased=True, keepdim=True)
            )

        recovered = recover_train_normalized_raw_edge_attributes(
            timestamps=timestamps,
            train_edge_attr=legacy_zscore(raw[:4]),
            full_edge_attr=legacy_zscore(raw),
        )
        expected_amount = (
            (amount - amount[:4].mean())
            / amount[:4].std(unbiased=False)
        )
        self.assertTrue(torch.allclose(
            recovered[:, 1], expected_amount, atol=1e-5))
        self.assertTrue(torch.equal(recovered[:, 2], currency))
        self.assertTrue(torch.equal(recovered[:, 3], payment))


class TemporalIncidentIndexTest(unittest.TestCase):
    def setUp(self):
        # Edges 0/1 share a timestamp. Edge 1 may use edge 0 as context, but
        # edge 0 may not use edge 1. Edge 6 is the target u=0 -> v=1.
        self.edge_index = torch.tensor([
            [0, 2, 1, 3, 1, 4, 0],
            [2, 1, 3, 0, 4, 0, 1],
        ])
        self.timestamps = torch.tensor([1, 1, 2, 2, 3, 3, 4])
        attrs = torch.stack(
            (
                self.timestamps.float(),
                torch.arange(7).float(),
                torch.zeros(7),
                torch.ones(7),
            ),
            dim=-1,
        )
        self.index = TemporalIncidentIndex(
            self.edge_index, self.timestamps, attrs)

    def test_future_and_self_edges_are_excluded(self):
        evidence = self.index.query(torch.tensor([0]), max_tokens=8)
        self.assertFalse(evidence.mask.any())
        self.assertEqual(float(evidence.support.sum()), 0.0)

        evidence = self.index.query(torch.tensor([6]), max_tokens=8)
        context = evidence.context_edge_ids[0][evidence.mask[0]]
        self.assertNotIn(6, context.tolist())
        self.assertTrue(
            (self.timestamps[context] <= self.timestamps[6]).all())

    def test_equal_time_uses_global_edge_id_tie_break(self):
        evidence = self.index.query(torch.tensor([1]), max_tokens=8)
        context = evidence.context_edge_ids[0][evidence.mask[0]]
        self.assertIn(0, context.tolist())
        self.assertNotIn(1, context.tolist())

    def test_relay_and_cycle_flags_are_detected(self):
        evidence = self.index.query(torch.tensor([6]), max_tokens=8)
        context = evidence.context_edge_ids[0][evidence.mask[0]]
        token = evidence.tokens[0][evidence.mask[0]]
        motif = token[:, -3:]
        by_edge = {
            int(edge_id): motif[position]
            for position, edge_id in enumerate(context.tolist())
        }
        self.assertEqual(float(by_edge[0][1]), 1.0)
        self.assertEqual(float(by_edge[1][1]), 1.0)
        self.assertEqual(float(by_edge[2][2]), 1.0)
        self.assertEqual(float(by_edge[3][2]), 1.0)

    def test_duplicate_directed_edges_keep_distinct_ids(self):
        edge_index = torch.tensor([[0, 0, 0], [1, 1, 1]])
        timestamps = torch.tensor([1, 2, 3])
        attrs = torch.zeros((3, 4))
        index = TemporalIncidentIndex(edge_index, timestamps, attrs)
        evidence = index.query(torch.tensor([2]), max_tokens=4)
        context = evidence.context_edge_ids[0][evidence.mask[0]]
        self.assertEqual(set(context.tolist()), {0, 1})

    def test_shuffle_and_off_preserve_shapes(self):
        evidence = self.index.query(torch.tensor([5, 6]), max_tokens=4)
        shuffled = evidence.shuffled(torch.tensor([1, 0]))
        self.assertTrue(torch.equal(shuffled.tokens[0], evidence.tokens[1]))
        self.assertTrue(torch.equal(
            shuffled.context_edge_ids[0], evidence.context_edge_ids[1]))

        off = evidence.off()
        self.assertEqual(off.tokens.shape, evidence.tokens.shape)
        self.assertFalse(off.mask.any())
        self.assertTrue((off.context_edge_ids == -1).all())
        self.assertEqual(float(off.support.sum()), 0.0)

    def test_time_window_and_empty_batch(self):
        evidence = self.index.query(
            torch.tensor([6]), max_tokens=8, time_window=1)
        context = evidence.context_edge_ids[0][evidence.mask[0]]
        self.assertTrue(
            ((self.timestamps[6] - self.timestamps[context]) <= 1).all())

        empty = self.index.query(
            torch.empty(0, dtype=torch.long), max_tokens=3)
        self.assertEqual(empty.tokens.shape, (0, 3, self.index.token_dim))
        self.assertEqual(empty.support.shape, (0, self.index.SUPPORT_DIM))

    def test_vectorized_query_matches_reference(self):
        targets = torch.arange(self.edge_index.size(1))
        for max_tokens in (1, 3, 8):
            for time_window in (None, 0, 1, 10):
                expected = self.index.query_reference(
                    targets, max_tokens=max_tokens,
                    time_window=time_window)
                actual = self.index.query(
                    targets, max_tokens=max_tokens,
                    time_window=time_window)
                self.assertTrue(torch.equal(
                    actual.context_edge_ids, expected.context_edge_ids))
                self.assertTrue(torch.equal(actual.mask, expected.mask))
                self.assertTrue(torch.equal(actual.tokens, expected.tokens))
                self.assertTrue(torch.equal(actual.support, expected.support))

    def test_vectorized_query_matches_reference_on_random_graphs(self):
        generator = torch.Generator().manual_seed(20260723)
        for _ in range(12):
            num_nodes = 8
            num_edges = 40
            edge_index = torch.randint(
                num_nodes, (2, num_edges), generator=generator)
            timestamps = torch.randint(
                0, 8, (num_edges,), generator=generator).sort().values
            raw = torch.randn(
                (num_edges, 4), generator=generator)
            raw[:, 0] = timestamps.float()
            index = TemporalIncidentIndex(
                edge_index, timestamps, raw)
            targets = torch.randint(
                num_edges, (17,), generator=generator)
            for max_tokens in (1, 4, 9):
                for time_window in (None, 0, 2):
                    expected = index.query_reference(
                        targets, max_tokens, time_window)
                    actual = index.query(
                        targets, max_tokens, time_window)
                    self.assertTrue(torch.equal(
                        actual.context_edge_ids,
                        expected.context_edge_ids,
                    ))
                    self.assertTrue(torch.equal(
                        actual.mask, expected.mask))
                    self.assertTrue(torch.equal(
                        actual.tokens, expected.tokens))
                    self.assertTrue(torch.equal(
                        actual.support, expected.support))

    def test_role_motif_selection_can_retain_older_reciprocal_edge(self):
        edge_index = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 2, 3, 4, 5, 1],
        ])
        timestamps = torch.arange(6)
        raw = torch.zeros((6, 4))
        raw[:, 0] = timestamps
        index = TemporalIncidentIndex(edge_index, timestamps, raw)

        recent = index.query(
            torch.tensor([5]), max_tokens=2, selection="recent")
        selected = index.query(
            torch.tensor([5]),
            max_tokens=2,
            selection="role_motif",
            selection_pool_factor=3,
        )

        self.assertNotIn(0, recent.context_edge_ids[0].tolist())
        self.assertIn(0, selected.context_edge_ids[0].tolist())
        self.assertGreater(selected.tokens[0, :, -3].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
