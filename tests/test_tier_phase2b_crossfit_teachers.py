import json
import unittest
from pathlib import Path

import torch

from run.tier_phase2b_crossfit_teachers import (
    aggregate_unique,
    fold_ids,
    fold_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "run" / "tier_phase2b_crossfit_spec.json"


class TierPhaseTwoBCrossFitTeacherTest(unittest.TestCase):
    def test_registered_matrix_has_three_folds_per_dataset(self):
        spec = json.loads(SPEC.read_text())
        keys = {
            (row["dataset"], row["fold"]) for row in spec["tasks"]
        }
        self.assertEqual(len(spec["tasks"]), 6)
        self.assertEqual(len(keys), 6)
        self.assertEqual(spec["num_folds"], 3)

    def test_fold_assignment_partitions_edges(self):
        edge_ids = torch.arange(300)
        assignments = fold_ids(edge_ids, 42, 3)
        masks = [
            fold_mask(edge_ids, 42, 3, fold, held_out=True)
            for fold in range(3)
        ]
        self.assertTrue(torch.equal(
            torch.stack(masks).sum(0), torch.ones(300)))
        self.assertTrue(all(int(mask.sum()) == 100 for mask in masks))
        self.assertTrue(torch.equal(
            assignments, torch.stack(masks).long().argmax(0)))

    def test_fit_mask_is_exact_held_out_complement(self):
        edge_ids = torch.arange(50)
        for fold in range(3):
            held_out = fold_mask(
                edge_ids, 44, 3, fold, held_out=True)
            fit = fold_mask(edge_ids, 44, 3, fold, held_out=False)
            self.assertTrue(torch.equal(fit, ~held_out))

    def test_unique_oof_aggregation_preserves_perturbations(self):
        values = {
            "edge_ids": torch.tensor([2, 2, 9]),
            "labels": torch.tensor([1, 1, 0]),
            "a2_scores": torch.tensor([0.2, 0.4, 0.8]),
            "evidence_scores": torch.tensor([0.7, 0.9, 0.1]),
            "shuffled_scores": torch.tensor([0.3, 0.5, 0.6]),
            "off_scores": torch.tensor([0.1, 0.1, 0.4]),
            "support_count": torch.tensor([2.0, 4.0, 1.0]),
        }
        output = aggregate_unique(values)
        self.assertEqual(output["edge_ids"].tolist(), [2, 9])
        self.assertTrue(torch.allclose(
            output["evidence_scores"], torch.tensor([0.8, 0.1])))
        self.assertTrue(torch.allclose(
            output["shuffled_scores"], torch.tensor([0.4, 0.6])))
        self.assertTrue(torch.allclose(
            output["support_count"], torch.tensor([3.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
