import unittest
from pathlib import Path

import torch

from run.cet_dynamic_reliability_audit import (
    scalar_distribution,
    summarize_events,
    support_conditioned_audit,
)


class CETDynamicReliabilityAuditTest(unittest.TestCase):
    def test_audit_source_has_no_fixed_evaluation_rng(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "run"
            / "cet_dynamic_reliability_audit.py"
        ).read_text()
        forbidden = (
            "torch." + "Generator(",
            "get_" + "rng_state(",
            "set_" + "rng_state(",
            "fixed_target_panel" + "=True",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_scalar_distribution_records_sample_std(self):
        row = scalar_distribution([1.0, 2.0, 3.0])
        self.assertEqual(row["count"], 3)
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["std"], 1.0)
        self.assertEqual(row["min"], 1.0)
        self.assertEqual(row["max"], 3.0)

    def test_support_conditioned_counts_corrections_and_breaks(self):
        test = {
            "labels": torch.tensor([1, 0, 1, 0]),
            "support": torch.tensor([
                [0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0],
                [12, 0, 0, 0, 0, 0],
                [48, 0, 0, 0, 0, 0],
            ]),
            "normal": torch.tensor([0.9, 0.9, 0.9, 0.1]),
            "shuffled": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "off": torch.tensor([0.4, 0.4, 0.4, 0.4]),
            "base": torch.tensor([0.1, 0.1, 0.9, 0.1]),
        }
        rows = {
            row["bin"]: row
            for row in support_conditioned_audit(test, 0.5, 0.5)
        }
        self.assertEqual(rows["zero"]["corrected"], 1)
        self.assertEqual(rows["low_1_7"]["broken"], 1)
        self.assertEqual(rows["medium_8_23"]["changed"], 0)
        self.assertEqual(rows["max_48"]["changed"], 0)

    def test_summarize_events_keeps_paired_mechanism_fields(self):
        event = {
            "test": {
                "normal": {"f1": 0.4},
                "shuffled": {"f1": 0.2},
                "off": {"f1": 0.3},
            },
            "base_test": {"f1": 0.35},
            "delta_vs_initial_a2": 0.1,
            "same_batch_delta_vs_frozen_a2": 0.05,
            "normal_minus_shuffled_f1": 0.2,
            "normal_minus_off_f1": 0.1,
            "intervention": {
                "changed_predictions": 20,
                "corrected_predictions": 12,
                "broken_predictions": 8,
                "corrected_minus_broken": 4,
            },
            "test_unique_edge_rate": 0.8,
        }
        summary = summarize_events([event, event])
        self.assertEqual(summary["normal_test_f1"]["mean"], 0.4)
        self.assertEqual(
            summary["normal_minus_shuffled_f1"]["mean"], 0.2)
        self.assertEqual(
            summary["corrected_minus_broken"]["mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
