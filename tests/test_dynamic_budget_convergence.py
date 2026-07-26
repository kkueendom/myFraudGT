import json
import unittest
from pathlib import Path

import torch

from run.dynamic_budget_convergence import (
    _check_finite,
    build_budget_rows,
    verify_nested_prefixes,
)
from run.nested_dynamic_stability_audit import expand_tasks
from run.summarize_dynamic_budget_convergence import percentile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prefix(labels, scores, edge_ids):
    return {
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores),
        "edge_ids": torch.tensor(edge_ids),
        "steps": len(labels),
        "unique_edges": len(set(edge_ids)),
        "unique_edge_rate": len(set(edge_ids)) / len(edge_ids),
        "edge_id_sha256": "fixture",
    }


class DynamicBudgetConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(
            (
                REPO_ROOT
                / "run"
                / "dynamic_budget_convergence_spec.json"
            ).read_text()
        )

    def test_registered_factorial_design(self):
        tasks = expand_tasks(self.spec)
        self.assertEqual(len(tasks), 36)
        self.assertEqual(self.spec["event_count"], 288)
        self.assertEqual(
            self.spec["budgets"], [4, 8, 16, 32, 64, 128, 256])
        self.assertEqual(self.spec["reference_budget"], 256)
        self.assertEqual(
            {
                (
                    task["dataset"],
                    task["model_seed"],
                    task["audit_seed"],
                )
                for task in tasks
            },
            {
                (
                    dataset["dataset"],
                    model_seed,
                    audit_seed,
                )
                for dataset in self.spec["datasets"]
                for model_seed in dataset["model_seeds"]
                for audit_seed in self.spec["audit_seeds"]
            },
        )
        self.assertTrue(all(task["repeats"] == 8 for task in tasks))

    def test_budget_rows_use_paired_full_reference(self):
        val = {
            1: _prefix([0, 1], [0.1, 0.8], [1, 2]),
            2: _prefix(
                [0, 1, 0, 1],
                [0.1, 0.8, 0.7, 0.9],
                [1, 2, 3, 4],
            ),
        }
        test = {
            1: _prefix([0, 1], [0.2, 0.7], [5, 6]),
            2: _prefix(
                [0, 1, 0, 1],
                [0.2, 0.7, 0.8, 0.9],
                [5, 6, 7, 8],
            ),
        }
        rows = build_budget_rows(val, test, historical_f1=0.5)
        self.assertEqual(
            rows["2"]["paired_absolute_f1_error_vs_reference"],
            0.0,
        )
        self.assertAlmostEqual(
            rows["1"]["paired_f1_error_vs_reference"],
            rows["1"]["test"]["f1"] - rows["2"]["test"]["f1"],
        )
        self.assertEqual(
            rows["1"]["delta_sign_agrees_with_reference"],
            (
                rows["1"]["delta_sign_vs_initial_a2"]
                == rows["2"]["delta_sign_vs_initial_a2"]
            ),
        )

    def test_non_finite_values_are_rejected(self):
        _check_finite({"valid": [0.0, 1.0]})
        with self.assertRaises(ValueError):
            _check_finite({"invalid": float("nan")})

    def test_nested_prefix_verification(self):
        rows = {
            1: {"edge_ids": torch.tensor([1, 2])},
            2: {"edge_ids": torch.tensor([1, 2, 3, 4])},
        }
        self.assertTrue(verify_nested_prefixes(rows))
        rows[1]["edge_ids"] = torch.tensor([1, 3])
        with self.assertRaises(AssertionError):
            verify_nested_prefixes(rows)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.5), 5.0)
        self.assertEqual(percentile([0, 10, 20], 0.9), 18.0)


if __name__ == "__main__":
    unittest.main()
