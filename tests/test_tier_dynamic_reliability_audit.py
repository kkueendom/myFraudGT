import json
import unittest
from pathlib import Path

from run.tier_dynamic_reliability_audit import (
    scalar_distribution,
    summarize_events,
)


class TierDynamicReliabilityAuditTest(unittest.TestCase):
    def test_source_has_no_fixed_evaluation_rng(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "run"
            / "tier_dynamic_reliability_audit.py"
        ).read_text()
        forbidden = (
            "torch." + "Generator(",
            "get_" + "rng_state(",
            "set_" + "rng_state(",
            "fixed_target_panel" + "=True",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_spec_has_two_streams_per_model_dataset_pair(self):
        spec = json.loads((
            Path(__file__).resolve().parents[1]
            / "run"
            / "tier_dynamic_reliability_spec.json"
        ).read_text())
        grouped = {}
        for task in spec["tasks"]:
            key = (task["dataset"], task["selection"])
            grouped.setdefault(key, []).append(task)
            self.assertEqual(task["repeats"], 8)
        self.assertEqual(len(grouped), 4)
        for tasks in grouped.values():
            self.assertEqual(len(tasks), 2)
            self.assertEqual(
                len({task["audit_seed"] for task in tasks}), 2)

    def test_launcher_starts_seven_distinct_tasks_then_claims_eighth(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "run"
            / "tier_dynamic_reliability_7gpu.sh"
        ).read_text()
        initial_tasks = []
        for line in source.splitlines():
            if line.startswith("run_worker "):
                _, gpu, task, _ = line.split()
                self.assertEqual(int(gpu), len(initial_tasks))
                initial_tasks.append(int(task))
        self.assertEqual(initial_tasks, [4, 5, 6, 7, 0, 1, 2])
        self.assertIn('mkdir "$OUTPUT/.task3_claim"', source)
        self.assertIn('run_task "$gpu" 3', source)

    def test_scalar_distribution_uses_sample_std(self):
        row = scalar_distribution([1, 2, 3])
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["std"], 1.0)

    def test_summary_separates_sensitivity_and_utility(self):
        event = {
            "test": {
                "normal": {"f1": 0.4},
                "shuffled": {"f1": 0.2},
                "off": {"f1": 0.1},
            },
            "base_test": {"f1": 0.35},
            "same_batch_delta_vs_frozen_a2": 0.05,
            "normal_minus_shuffled_f1": 0.2,
            "normal_minus_off_f1": 0.3,
            "mean_abs_normal_minus_shuffled_score": 0.1,
            "mean_abs_normal_minus_off_score": 0.2,
            "interventions": {
                "normal": {
                    "changed_predictions": 20,
                    "corrected_predictions": 12,
                    "broken_predictions": 8,
                },
            },
            "coverage_rate": 0.9,
            "test_unique_edge_rate": 1.0,
            "evidence_threshold": 0.4,
            "base_threshold": 0.5,
        }
        summary = summarize_events([event, event])
        self.assertEqual(
            summary["normal_minus_shuffled_f1"]["mean"], 0.2)
        self.assertEqual(
            summary["same_batch_delta_vs_frozen_a2"]["mean"], 0.05)
        self.assertEqual(
            summary["normal_corrected_minus_broken"]["mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
