import unittest

import torch

from fraudGT.evidence.gtprc import (
    gather_policy,
    grouped_empirical_bernstein_upper,
    row_wilson_upper,
    select_max_coverage_policy,
)
from run.summarize_gtprc_phase0a import (
    build_aggregate,
)


class GTPRCTest(unittest.TestCase):
    def test_row_upper_rejects_empty_policy(self):
        selected = torch.zeros((1, 2, 8), dtype=torch.bool)
        broken = torch.zeros((1, 8), dtype=torch.bool)
        upper = row_wilson_upper(selected, broken, 2, 0.05)
        self.assertTrue(torch.equal(upper, torch.ones_like(upper)))

    def test_group_bound_is_stricter_with_fewer_groups(self):
        selected = torch.ones((1, 1, 1024), dtype=torch.bool)
        broken = torch.zeros((1, 1024), dtype=torch.bool)
        small_groups = grouped_empirical_bernstein_upper(
            selected, broken, 2, 1, 0.05)
        large_groups = grouped_empirical_bernstein_upper(
            selected, broken, 64, 1, 0.05)
        self.assertGreater(
            large_groups.item(), small_groups.item())

    def test_policy_selection_prefers_coverage_under_constraints(self):
        coverage = torch.tensor([[0.1, 0.3, 0.2]])
        net = torch.tensor([[0.1, 0.1, -0.1]])
        upper = torch.tensor([[0.2, 0.3, 0.1]])
        chosen = select_max_coverage_policy(
            coverage, net, upper, 0.4, 0.01)
        self.assertEqual(chosen.item(), 1)
        gathered = gather_policy(coverage, chosen)
        self.assertAlmostEqual(gathered.item(), 0.3, places=6)

    def test_aggregate_requires_all_seven_regimes(self):
        with self.assertRaises(ValueError):
            build_aggregate([])


if __name__ == "__main__":
    unittest.main()
