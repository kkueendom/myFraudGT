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
    def test_aligns_a2_values_by_global_edge_id(self):
        target_ids = torch.tensor([30, 10, 20])
        source_ids = torch.tensor([20, 30, 10])
        scores = torch.tensor([2.0, 3.0, 1.0])
        labels = torch.tensor([0, 1, 1])
        aligned_scores, aligned_labels = runner.align_values_by_edge_id(
            target_ids, source_ids, scores, labels
        )
        self.assertTrue(torch.equal(
            aligned_scores, torch.tensor([3.0, 1.0, 2.0])
        ))
        self.assertTrue(torch.equal(
            aligned_labels, torch.tensor([1, 1, 0])
        ))

    def test_rejects_mismatched_paired_edge_sets(self):
        with self.assertRaises(AssertionError):
            runner.align_values_by_edge_id(
                torch.tensor([1, 2]),
                torch.tensor([1, 3]),
                torch.tensor([0.1, 0.2]),
            )

    def test_selects_a2_scored_subset_in_a2_order(self):
        requested_ids = torch.tensor([20, 30])
        source_ids = torch.tensor([30, 10, 20])
        scores = torch.tensor([3.0, 1.0, 2.0])
        labels = torch.tensor([1, 0, 1])
        selected_scores, selected_labels = (
            runner.select_values_by_edge_id(
                requested_ids, source_ids, scores, labels
            )
        )
        self.assertTrue(torch.equal(
            selected_scores, torch.tensor([2.0, 3.0])
        ))
        self.assertTrue(torch.equal(
            selected_labels, torch.tensor([1, 1])
        ))

    def test_rejects_absent_subset_edge(self):
        with self.assertRaises(AssertionError):
            runner.select_values_by_edge_id(
                torch.tensor([20, 40]),
                torch.tensor([10, 20, 30]),
                torch.tensor([1.0, 2.0, 3.0]),
            )

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
