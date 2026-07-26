import unittest

import torch

from run.cpse_phase0b_utility_audit import (
    graph_time_groups,
    grouped_upper,
)


class CPSEUtilityAuditTest(unittest.TestCase):
    def test_shared_entity_in_same_time_block_is_grouped(self):
        groups = graph_time_groups(
            torch.tensor([1, 2, 8]),
            torch.tensor([2, 3, 9]),
            torch.tensor([1, 1, 1]),
            block_count=1,
        )
        self.assertEqual(groups[0].item(), groups[1].item())
        self.assertNotEqual(groups[1].item(), groups[2].item())

    def test_grouped_upper_rejects_no_intervention(self):
        active = torch.zeros(4, dtype=torch.bool)
        broken = torch.zeros(4, dtype=torch.bool)
        groups = torch.arange(4)
        self.assertEqual(grouped_upper(active, broken, groups), 1.0)

    def test_more_independent_groups_tighten_zero_break_bound(self):
        active = torch.ones(16, dtype=torch.bool)
        broken = torch.zeros(16, dtype=torch.bool)
        many = grouped_upper(active, broken, torch.arange(16))
        few = grouped_upper(
            active, broken, torch.arange(16) // 8)
        self.assertLess(many, few)

    def test_vectorized_group_rates_match_manual_reference(self):
        active = torch.tensor([True, True, False, True, True, True])
        broken = torch.tensor([True, False, True, False, True, True])
        groups = torch.tensor([4, 4, 4, 9, 9, 12])
        expected_mean = (0.5 + 0.5 + 1.0) / 3
        expected = min(
            1.0,
            expected_mean
            + (torch.log(torch.tensor(64 / 0.05)) / 6).sqrt().item(),
        )
        self.assertAlmostEqual(
            grouped_upper(active, broken, groups), expected, places=6)


if __name__ == "__main__":
    unittest.main()
