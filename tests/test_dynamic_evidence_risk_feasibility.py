import unittest

import torch

from run.dynamic_evidence_risk_feasibility import (
    dataset_gate,
    merge_payloads,
    policy_statistics,
    wilson_upper,
)


def payload(fold, edge_ids, labels):
    size = len(labels)
    return {
        "fold": fold,
        "edge_ids": torch.tensor(edge_ids),
        "labels": torch.tensor(labels),
        "a2_scores": torch.tensor([0.1, 0.9][:size]),
        "evidence_scores": torch.tensor([0.9, 0.1][:size]),
        "shuffled_scores": torch.full((size,), 0.5),
        "off_scores": torch.full((size,), 0.5),
        "support_count": torch.ones(size),
        "a2_threshold": 0.5,
        "evidence_threshold": 0.5,
    }


class DynamicEvidenceRiskFeasibilityTest(unittest.TestCase):
    def test_wilson_upper_is_conservative_and_decreases_with_n(self):
        self.assertEqual(wilson_upper(0, 0), 1.0)
        self.assertGreater(wilson_upper(1, 10), 0.1)
        self.assertLess(wilson_upper(0, 100), wilson_upper(0, 10))

    def test_merge_payloads_preserves_fold_thresholds(self):
        rows = merge_payloads([
            payload(0, [1, 2], [1, 0]),
            payload(1, [3, 4], [1, 0]),
        ])
        self.assertEqual(rows["fold_ids"].tolist(), [0, 0, 1, 1])
        self.assertEqual(rows["a2_thresholds"].tolist(), [0.5] * 4)

    def test_policy_statistics_counts_corrected_and_broken(self):
        rows = merge_payloads([
            payload(0, [1, 2], [1, 0]),
        ])
        stats = policy_statistics(
            rows,
            "normal",
            torch.tensor([True, True]),
            fold=0,
        )
        self.assertEqual(stats["changed"], 2)
        self.assertEqual(stats["corrected"], 2)
        self.assertEqual(stats["broken"], 0)
        self.assertIsNone(stats["corrected_to_broken_ratio"])
        self.assertTrue(stats["corrected_without_breaks"])
        self.assertGreater(stats["paired_f1_delta"], 0.0)

    def test_dataset_gate_requires_cross_fold_coverage(self):
        good = {
            "normal": {
                "changed": 60,
                "corrected": 40,
                "broken": 20,
                "corrected_minus_broken": 20,
                "positive_net_folds": 2,
                "paired_f1_delta_sum": 0.03,
                "paired_f1_delta_mean": 0.01,
            },
            "shuffled": {
                "changed": 10,
                "corrected": 3,
                "broken": 7,
                "corrected_minus_broken": -4,
                "positive_net_folds": 0,
                "paired_f1_delta_sum": -0.01,
                "paired_f1_delta_mean": -0.003,
            },
        }
        gate = dataset_gate(good)
        self.assertTrue(gate["passed"])
        bad = {
            **good,
            "normal": {**good["normal"], "changed": 49},
        }
        self.assertFalse(dataset_gate(bad)["min_changed"])


if __name__ == "__main__":
    unittest.main()
