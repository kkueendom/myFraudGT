import unittest

import torch

from run.tier_phase2b_oof_utility_audit import (
    condition_features,
    direction_candidates,
    normal_utility_targets,
    policy_statistics,
    probe_scores,
    split_assignments,
    train_probe,
)


class TierPhaseTwoBOOFUtilityAuditTest(unittest.TestCase):
    def rows(self):
        return {
            "edge_ids": torch.arange(10),
            "labels": torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 1, 0]),
            "a2_scores": torch.tensor(
                [0.2, 0.8, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1, 0.9]
            ),
            "evidence_scores": torch.tensor(
                [0.8, 0.2, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2]
            ),
            "shuffled_scores": torch.full((10,), 0.5),
            "off_scores": torch.full((10,), 0.4),
            "support_count": torch.arange(1, 11).float(),
            "a2_thresholds": torch.full((10,), 0.5),
            "evidence_thresholds": torch.full((10,), 0.5),
        }

    def test_three_way_split_is_disjoint_and_complete(self):
        edge_ids = torch.arange(100)
        split = split_assignments(edge_ids, 42)
        self.assertTrue(((split >= 0) & (split < 5)).all())
        masks = [split == bucket for bucket in range(5)]
        self.assertTrue(torch.equal(
            torch.stack(masks).sum(0), torch.ones(100)))
        self.assertTrue(all(int(mask.sum()) == 20 for mask in masks))

    def test_direction_candidates_and_utility_targets(self):
        rows = self.rows()
        add = direction_candidates(rows, "normal", "add")
        remove = direction_candidates(rows, "normal", "remove")
        self.assertTrue(torch.equal(add, torch.tensor(
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)))
        self.assertTrue(torch.equal(remove, ~add))
        add_targets = normal_utility_targets(rows, "add")
        self.assertEqual(int(add_targets[add].sum()), 4)
        self.assertEqual(int(add.sum()), 5)

    def test_features_do_not_depend_on_labels(self):
        rows = self.rows()
        features = condition_features(rows, "normal")
        changed = dict(rows)
        changed["labels"] = 1 - rows["labels"]
        self.assertEqual(features.shape, (10, 7))
        self.assertTrue(torch.equal(
            features, condition_features(changed, "normal")))

    def test_policy_statistics_counts_corrections_and_breaks(self):
        rows = self.rows()
        active = torch.ones(10, dtype=torch.bool)
        stats = policy_statistics(
            rows,
            "normal",
            active,
            torch.ones(10, dtype=torch.bool),
        )
        self.assertEqual(
            stats["changed_predictions"], 10)
        self.assertEqual(
            stats["corrected_predictions"] + stats["broken_predictions"],
            10,
        )

    def test_probe_training_is_seed_deterministic(self):
        rows = self.rows()
        split = torch.zeros(10, dtype=torch.long)
        first, first_diagnostics = train_probe(
            rows, "add", split, torch.device("cpu"), 4201, epochs=5)
        second, second_diagnostics = train_probe(
            rows, "add", split, torch.device("cpu"), 4201, epochs=5)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertTrue(torch.equal(
            probe_scores(first, rows, "normal", torch.device("cpu")),
            probe_scores(second, rows, "normal", torch.device("cpu")),
        ))


if __name__ == "__main__":
    unittest.main()
