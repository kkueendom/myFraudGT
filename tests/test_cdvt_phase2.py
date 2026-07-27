import json
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the training environment has PyYAML
    yaml = None

from run.cdvt_phase2_summary import (
    DATASETS,
    PHASE1_DATASETS,
    build_summary,
    markdown,
)
from run.cdvt_protocol import INITIAL_A2


class CDVTPhase2Test(unittest.TestCase):
    def write_result(self, root, dataset, delta):
        task = root / f"{dataset}_dual_view_seed42"
        task.mkdir(parents=True)
        baseline = INITIAL_A2[dataset]
        payload = {
            "phase": "CDVT_phase2",
            "sampling_protocol": "dynamic_random",
            "dataset": dataset,
            "variant": "dual_view",
            "architecture_variant": "dual_view",
            "lambda_cons": 0.0,
            "seed": 42,
            "git_commit": "abc12345",
            "config": f"configs/CDVT/phase2/AML-{dataset}.yaml",
            "checkpoint": str(task / "best_val.ckpt"),
            "val_selected_epoch": 80,
            "val_selected_test_f1": (
                baseline["val_selected_test_f1"] + delta),
            "raw_best_epoch": 84,
            "raw_best_test_f1": baseline["raw_best_test_f1"] + delta,
            "parameter_count": 100,
            "peak_gpu_memory_bytes": 200,
            "elapsed_seconds": 10.0,
            "config_snapshot": {
                "dataset": {"tier_evidence": True},
                "train": {
                    "sampler": "link_neighbor",
                    "iter_per_epoch": 256,
                    "batch_size": 2048,
                    "eval_period": 4,
                },
                "val": {
                    "iter_per_epoch": 256,
                    "fixed_target_panel": False,
                },
            },
        }
        (task / "manifest.json").write_text(json.dumps(payload))

    def test_summary_uses_separate_metric_columns_and_phase2_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            phase1 = Path(directory) / "phase1"
            phase2 = Path(directory) / "phase2"
            phase1.mkdir()
            phase2.mkdir()
            deltas = (0.01, 0.01, 0.01, 0.01, -0.002, -0.002)
            for dataset, delta in zip(DATASETS, deltas):
                root = phase1 if dataset in PHASE1_DATASETS else phase2
                self.write_result(root, dataset, delta)
            result = build_summary(phase1, phase2)
            self.assertEqual(result["val_selected_summary"]["wins"], 4)
            self.assertEqual(result["val_selected_summary"]["losses"], 2)
            self.assertEqual(
                result["val_selected_summary"]["possible_sampling_variation"],
                2,
            )
            self.assertAlmostEqual(
                result["val_selected_summary"]["mean_delta"], 0.006)
            self.assertTrue(result["phase2_gate"]["advance_to_phase3"])
            self.assertIn("A2 raw-best", markdown(result))

    def test_incomplete_results_require_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            phase1 = Path(directory) / "phase1"
            phase2 = Path(directory) / "phase2"
            phase1.mkdir()
            phase2.mkdir()
            with self.assertRaises(FileNotFoundError):
                build_summary(phase1, phase2)
            result = build_summary(phase1, phase2, allow_incomplete=True)
            self.assertEqual(len(result["missing_manifests"]), 6)
            self.assertFalse(result["phase2_gate"]["advance_to_phase3"])

    @unittest.skipIf(yaml is None, "PyYAML is unavailable")
    def test_phase2_configs_freeze_protocol_and_architecture(self):
        config_root = Path("configs/CDVT/phase2")
        datasets = ("Small-HI", "Medium-LI", "Medium-HI", "Large-HI")
        for dataset in datasets:
            config = yaml.safe_load(
                (config_root / f"AML-{dataset}.yaml").read_text())
            self.assertEqual(config["dataset"]["name"], dataset)
            self.assertTrue(config["dataset"]["tier_evidence"])
            self.assertEqual(config["train"]["batch_size"], 2048)
            self.assertEqual(config["train"]["iter_per_epoch"], 256)
            self.assertEqual(config["train"]["eval_period"], 4)
            self.assertEqual(config["val"]["iter_per_epoch"], 256)
            self.assertFalse(config["val"]["fixed_target_panel"])
            self.assertEqual(config["cdvt"]["variant"], "dual_view")
            self.assertEqual(config["cdvt"]["lambda_cons"], 0.0)
            self.assertEqual(config["optim"]["max_epoch"], 500)

    def test_queue_runs_only_four_new_datasets_on_idle_gpus(self):
        source = Path("run/cdvt_phase2_queue.sh").read_text()
        self.assertIn(
            "datasets=(Medium-LI Small-HI Medium-HI Large-HI)", source)
        self.assertIn("gpu_is_idle", source)
        self.assertIn("CDVT_POLL_SECONDS:-300", source)
        self.assertIn("--phase CDVT_phase2", source)
        self.assertIn("--lambda-cons 0.0", source)
        self.assertNotIn("Small-LI.yaml", source)
        self.assertNotIn("Large-LI.yaml", source)


if __name__ == "__main__":
    unittest.main()
