import json
import unittest
from pathlib import Path

import torch
from yacs.config import CfgNode

from run.a2_dynamic_stability_audit import (
    configure_a2_fraudgt,
    metric_row,
    prune_unknown_config,
    scalar_distribution,
    summarize,
)


class A2DynamicStabilityAuditTest(unittest.TestCase):
    def test_source_has_no_fixed_evaluation_rng(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "run"
            / "a2_dynamic_stability_audit.py"
        ).read_text()
        forbidden = (
            "torch." + "Generator(",
            "get_" + "rng_state(",
            "set_" + "rng_state(",
            "fixed_target_panel" + "=True",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_spec_has_two_distinct_streams_per_dataset(self):
        spec = json.loads((
            Path(__file__).resolve().parents[1]
            / "run"
            / "a2_dynamic_stability_spec.json"
        ).read_text())
        datasets = {}
        for task in spec["tasks"]:
            datasets.setdefault(task["dataset"], []).append(task)
            self.assertEqual(task["repeats"], 8)
        self.assertEqual(len(datasets), 6)
        for tasks in datasets.values():
            self.assertEqual(len(tasks), 2)
            self.assertEqual(
                len({task["audit_seed"] for task in tasks}), 2)

    def test_metric_row_records_confusion_counts(self):
        row = metric_row(
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([0.9, 0.2, 0.8, 0.1]),
            0.5,
        )
        self.assertEqual(row["tp"], 1)
        self.assertEqual(row["fp"], 1)
        self.assertEqual(row["fn"], 1)
        self.assertEqual(row["tn"], 1)
        self.assertEqual(row["f1"], 0.5)

    def test_unknown_saved_runtime_keys_are_pruned(self):
        schema = CfgNode({
            "train": {
                "batch_size": 1024,
            },
            "seed": 0,
        })
        clean, dropped = prune_unknown_config({
            "train": {
                "batch_size": 2048,
                "runtime_only": True,
            },
            "seed": 42,
            "run_dir": "/tmp/output",
        }, schema)
        self.assertEqual(clean["train"]["batch_size"], 2048)
        self.assertEqual(clean["seed"], 42)
        self.assertEqual(
            dropped, ["train.runtime_only", "run_dir"])

    def test_archived_task_entity_is_normalized_to_tuple(self):
        config = {
            "dataset": {
                "task_entity": ["node", "to", "node"],
            },
        }
        _, normalized = configure_a2_fraudgt(
            config, seed=42, device="cpu")
        from fraudGT.graphgym.config import cfg
        self.assertEqual(
            cfg.dataset.task_entity, ("node", "to", "node"))
        self.assertEqual(
            normalized, ["dataset.task_entity:list_to_tuple"])

    def test_summarize_preserves_historical_delta(self):
        event = {
            "val": {"f1": 0.6, "threshold": 0.4},
            "test": {
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "prevalence": 0.1,
                "positives": 10,
            },
            "test_unique_edge_rate": 0.8,
        }
        summary = summarize([event, event], historical_f1=0.45)
        self.assertEqual(summary["test_f1"]["mean"], 0.5)
        self.assertAlmostEqual(
            summary["delta_vs_initial_a2"]["mean"], 0.05)
        self.assertEqual(summary["test_unique_edge_rate"]["mean"], 0.8)

    def test_scalar_distribution_uses_sample_standard_deviation(self):
        row = scalar_distribution([1.0, 2.0, 3.0])
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["std"], 1.0)


if __name__ == "__main__":
    unittest.main()
