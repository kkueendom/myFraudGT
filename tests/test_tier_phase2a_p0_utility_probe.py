import json
import unittest
from pathlib import Path

import torch

from run.tier_phase2a_p0_utility_probe import (
    aggregate_unique_rows,
    deterministic_calibration_mask,
    direction_candidates,
    router_features,
    utility_targets,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "run" / "tier_phase2a_p0_utility_spec.json"


class TierPhaseTwoAP0UtilityProbeTest(unittest.TestCase):
    def test_registered_matrix_is_complete(self):
        spec = json.loads(SPEC.read_text())
        keys = {
            (row["dataset"], row["family"], row["selection"])
            for row in spec["tasks"]
        }
        self.assertEqual(len(spec["tasks"]), 12)
        self.assertEqual(len(keys), 12)
        self.assertEqual(spec["sampling_protocol"], "dynamic_random")

    def test_directional_candidates_and_targets(self):
        rows = {
            "labels": torch.tensor([1, 0, 0, 1]),
            "a2_scores": torch.tensor([0.2, 0.8, 0.2, 0.8]),
            "evidence_scores": torch.tensor([0.8, 0.2, 0.8, 0.2]),
            "support_count": torch.ones(4),
        }
        add = direction_candidates(rows, 0.5, 0.5, "add")
        remove = direction_candidates(rows, 0.5, 0.5, "remove")
        self.assertEqual(add.tolist(), [True, False, True, False])
        self.assertEqual(remove.tolist(), [False, True, False, True])
        self.assertEqual(
            utility_targets(rows, add, 0.5).tolist(), [1.0, 0.0])
        self.assertEqual(
            utility_targets(rows, remove, 0.5).tolist(), [1.0, 0.0])

    def test_unique_aggregation_does_not_mix_edge_ids(self):
        rows = {
            "target_edge_ids": torch.tensor([3, 3, 8]),
            "labels": torch.tensor([1, 1, 0]),
            "evidence_scores": torch.tensor([0.6, 0.8, 0.2]),
            "a2_scores": torch.tensor([0.4, 0.6, 0.1]),
            "support_count": torch.tensor([2.0, 4.0, 1.0]),
        }
        unique = aggregate_unique_rows(rows)
        self.assertEqual(unique["target_edge_ids"].tolist(), [3, 8])
        self.assertTrue(torch.allclose(
            unique["evidence_scores"], torch.tensor([0.7, 0.2])))
        self.assertTrue(torch.allclose(
            unique["support_count"], torch.tensor([3.0, 1.0])))

    def test_calibration_split_is_edge_deterministic(self):
        edge_ids = torch.arange(20)
        first = deterministic_calibration_mask(edge_ids, 42, 5, 0)
        second = deterministic_calibration_mask(edge_ids, 42, 5, 0)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(first.sum()), 4)
        self.assertEqual(int((~first).sum()), 16)

    def test_router_features_exclude_labels(self):
        rows = {
            "labels": torch.tensor([0, 1]),
            "a2_scores": torch.tensor([0.2, 0.8]),
            "evidence_scores": torch.tensor([0.7, 0.3]),
            "support_count": torch.tensor([1.0, 4.0]),
        }
        features = router_features(rows, 0.5, 0.5)
        flipped = dict(rows)
        flipped["labels"] = 1 - rows["labels"]
        self.assertEqual(features.shape, (2, 7))
        self.assertTrue(torch.equal(
            features, router_features(flipped, 0.5, 0.5)))


if __name__ == "__main__":
    unittest.main()
