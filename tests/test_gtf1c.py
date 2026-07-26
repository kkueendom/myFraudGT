import math
import unittest

import torch

from fraudGT.evidence.gtf1c import (
    graph_time_groups,
    grouped_break_upper,
    paired_f1_lcb,
    paired_policy_statistics,
)


class GTF1CTest(unittest.TestCase):
    def test_positive_row_net_can_reduce_f1(self):
        labels = torch.tensor(
            [True] * 10 + [False] * 30 + [True] * 20)
        base = torch.tensor(
            [True] * 10 + [True] * 20 + [False] * 10
            + [False] * 20)
        routed = base.clone()
        routed[10:30] = False
        routed[:5] = False
        stats = paired_policy_statistics(labels, base, routed)
        self.assertEqual(stats["corrected"], 20)
        self.assertEqual(stats["broken"], 5)
        self.assertGreater(stats["net"], 0)
        self.assertLess(stats["paired_f1_delta"], 0)

    def test_group_dependence_widens_paired_f1_interval(self):
        labels = torch.tensor(
            [1, 1, 0, 0], dtype=torch.bool).repeat_interleave(50)
        base = torch.tensor(
            [1, 0, 1, 0], dtype=torch.bool).repeat_interleave(50)
        routed = labels.clone()
        iid = paired_f1_lcb(labels, base, routed)
        clustered = paired_f1_lcb(
            labels, base, routed, torch.arange(200) // 50)
        self.assertGreater(
            clustered["standard_error"], iid["standard_error"])
        self.assertLess(
            clustered["lower_bound"], iid["lower_bound"])

    def test_shared_entity_is_grouped_inside_time_block(self):
        groups = graph_time_groups(
            torch.tensor([1, 2, 8]),
            torch.tensor([2, 3, 9]),
            torch.tensor([1, 1, 1]),
            block_count=1,
        )
        self.assertEqual(groups[0].item(), groups[1].item())
        self.assertNotEqual(groups[1].item(), groups[2].item())

    def test_empty_break_policy_is_not_qualified(self):
        active = torch.zeros(8, dtype=torch.bool)
        broken = torch.zeros(8, dtype=torch.bool)
        self.assertEqual(
            grouped_break_upper(active, broken, torch.arange(8)), 1.0)

    def test_more_groups_tighten_break_bound(self):
        active = torch.ones(32, dtype=torch.bool)
        broken = torch.zeros(32, dtype=torch.bool)
        many = grouped_break_upper(
            active, broken, torch.arange(32))
        few = grouped_break_upper(
            active, broken, torch.arange(32) // 16)
        self.assertLess(many, few)
        self.assertTrue(math.isfinite(many))


if __name__ == "__main__":
    unittest.main()
