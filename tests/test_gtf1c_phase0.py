import unittest

import torch

from run.gtf1c_phase0_real_graph import (
    REGIMES,
    candidate_policy_thresholds,
    generate_scores,
    select_policy,
)


def toy_view():
    labels = torch.tensor(
        [1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
    base = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bool)
    return {
        "labels": labels,
        "base": base,
        "source": torch.arange(8),
        "destination": torch.arange(8) + 8,
        "timestamps": torch.arange(8),
        "edge_ids": torch.arange(8),
        "time_groups": torch.arange(8) // 2,
        "graph_groups": torch.arange(8) // 2,
        "degree_signal": torch.zeros(8),
    }


class GTF1CPhase0Test(unittest.TestCase):
    def test_policy_family_has_registered_size(self):
        view = toy_view()
        generator = torch.Generator().manual_seed(9)
        scores = generate_scores(
            view, REGIMES["iid_positive"], "selection", generator)
        add, remove = candidate_policy_thresholds(
            view, scores, torch.linspace(0.5, 0.99, 16),
            REGIMES["iid_positive"])
        self.assertEqual(add.numel(), 48)
        self.assertEqual(remove.numel(), 48)

    def test_disabled_add_direction_has_infinite_threshold(self):
        view = toy_view()
        config = REGIMES["cpse_remove_fragility"]
        generator = torch.Generator().manual_seed(10)
        scores = generate_scores(
            view, config, "selection", generator)
        add, _ = candidate_policy_thresholds(
            view, scores, torch.linspace(0.5, 0.99, 16), config)
        self.assertTrue(torch.isinf(add[:16]).all())

    def test_selection_obeys_minimum_change_count(self):
        view = toy_view()
        generator = torch.Generator().manual_seed(11)
        scores = generate_scores(
            view, REGIMES["iid_positive"], "selection", generator)
        add, remove = candidate_policy_thresholds(
            view, scores, torch.linspace(0.5, 0.99, 16),
            REGIMES["iid_positive"])
        self.assertEqual(
            select_policy(view, scores, add, remove, min_changes=20),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
