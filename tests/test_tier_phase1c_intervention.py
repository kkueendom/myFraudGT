import json
import unittest
from pathlib import Path

import torch

from run.tier_phase1c_intervention_diagnostic import (
    intervention_statistics,
    passes_gate,
    policy_mask,
    select_policy,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "run" / "tier_phase1c_intervention_spec.json"
GATE = {
    "min_correction_rate_on_a2_errors": 0.10,
    "min_corrected_to_broken_ratio": 1.5,
    "min_changed_predictions": 2,
    "min_changed_fraction_of_a2_errors": 0.10,
}


class TierPhaseOneCInterventionTest(unittest.TestCase):
    def test_registered_matrix_has_twelve_unique_tasks(self):
        spec = json.loads(SPEC.read_text())
        tasks = {
            (row["dataset"], row["family"], row["selection"])
            for row in spec["tasks"]
        }
        self.assertEqual(len(spec["tasks"]), 12)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(spec["sampling_protocol"], "dynamic_random")

    def test_policy_mask_respects_direction_and_support(self):
        rows = {
            "a2_scores": torch.tensor([0.4, 0.6, 0.4]),
            "evidence_scores": torch.tensor([0.8, 0.2, 0.8]),
            "support_count": torch.tensor([4.0, 4.0, 1.0]),
        }
        policy = {
            "direction": "evidence_positive",
            "a2_max_margin": 0.2,
            "evidence_min_margin": 0.2,
            "support_min": 2.0,
        }
        self.assertEqual(
            policy_mask(rows, policy, 0.5, 0.5).tolist(),
            [True, False, False],
        )

    def test_no_break_policy_passes_ratio_gate(self):
        rows = {
            "labels": torch.tensor([1, 1, 0, 0]),
            "a2_scores": torch.tensor([0.1, 0.2, 0.1, 0.2]),
            "evidence_scores": torch.tensor([0.9, 0.8, 0.1, 0.2]),
        }
        intervention = torch.tensor([True, True, False, False])
        stats = intervention_statistics(
            rows, intervention, 0.5, 0.5)
        passed, required = passes_gate(stats, GATE, require_f1_gain=True)
        self.assertTrue(stats["corrected_without_breaks"])
        self.assertTrue(passed)
        self.assertEqual(required, 2)

    def test_validation_selection_prefers_net_corrections(self):
        rows = {
            "labels": torch.tensor([1, 1, 0, 0, 0, 0]),
            "a2_scores": torch.tensor([0.45, 0.46, 0.10, 0.20, 0.30, 0.40]),
            "evidence_scores": torch.tensor(
                [0.90, 0.80, 0.10, 0.20, 0.30, 0.40]
            ),
            "support_count": torch.tensor([4.0] * 6),
        }
        grid = {
            "directions": ["both", "evidence_positive"],
            "a2_max_margin_quantiles": [1.0],
            "evidence_min_margin_quantiles": [0.0],
            "support_min_quantiles": [0.0],
        }
        selected, evaluated, eligible = select_policy(
            rows, 0.5, 0.5, grid, GATE)
        self.assertEqual(evaluated, 2)
        self.assertEqual(eligible, 2)
        self.assertTrue(selected["eligible"])
        self.assertEqual(
            selected["statistics"]["corrected_predictions"], 2)


if __name__ == "__main__":
    unittest.main()
