import importlib.util
import unittest
from pathlib import Path

import torch


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "run"
    / "tier_phase1_evidence_qualification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tier_phase1_evidence_qualification", RUNNER_PATH
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TierPhaseOneMetricTest(unittest.TestCase):
    def test_threshold_does_not_split_equal_scores(self):
        labels = torch.tensor([1, 0, 1, 0])
        scores = torch.tensor([0.9, 0.8, 0.8, 0.1])
        threshold, f1 = runner.best_f1_threshold(labels, scores)
        predictions = scores >= threshold
        self.assertAlmostEqual(f1, runner.binary_f1(labels, predictions))
        self.assertAlmostEqual(threshold, 0.8)

    def test_average_precision_is_one_for_perfect_ranking(self):
        labels = torch.tensor([1, 0, 1, 0])
        scores = torch.tensor([0.9, 0.2, 0.8, 0.1])
        self.assertAlmostEqual(runner.average_precision(labels, scores), 1.0)

    def test_qualification_requires_mechanism_and_complementarity(self):
        selected = {
            "evidence_diagnostics": {
                "coverage_rate": 0.95,
                "class": {
                    "0": {"coverage_rate": 0.95},
                    "1": {"coverage_rate": 1.0},
                },
            },
            "test": {
                "normal": {"f1": 0.50},
                "shuffled": {"f1": 0.47},
                "off": {"f1": 0.46},
            }
        }
        diagnostic = {
            "sampled_instances": {
                "a2_errors": 60,
                "correction_rate_on_a2_errors": 0.20,
                "corrected_to_broken_ratio": 2.0,
                "corrected_without_breaks": False,
                "changed_predictions": 55,
            }
        }
        gate = {
            "min_normal_minus_shuffled_test_f1": 0.01,
            "min_normal_minus_off_test_f1": 0.005,
            "min_correction_rate_on_a2_errors": 0.10,
            "min_corrected_to_broken_ratio": 1.5,
            "min_changed_predictions": 50,
            "min_changed_fraction_of_a2_errors": 0.10,
            "min_overall_coverage": 0.80,
            "min_each_class_coverage": 0.80,
        }
        decision, checks = runner.qualification_decision(
            selected, diagnostic, gate
        )
        self.assertEqual(decision, "pass")
        self.assertEqual(checks["changed_predictions"], 55)

        diagnostic["sampled_instances"]["corrected_to_broken_ratio"] = 1.4
        decision, _ = runner.qualification_decision(
            selected, diagnostic, gate
        )
        self.assertEqual(decision, "fail")


if __name__ == "__main__":
    unittest.main()
