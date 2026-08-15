import json
import statistics
import tempfile
import unittest
from pathlib import Path

from run.cdvt_protocol import MULTI_FRAUDGT_PAPER
from run.cdvt_small_li_multiseed_summary import build_summary


def write_manifest(root, seed, selected, *, epochs=500, early_stop=False):
    task = root / f"Small-LI_multi_cdvt_seed{seed}"
    task.mkdir(parents=True)
    published = MULTI_FRAUDGT_PAPER["Small-LI"]
    payload = {
        "phase": "CDVT_multi_published_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": "Small-LI",
        "variant": "multi_cdvt",
        "architecture_variant": "dual_view",
        "seed": seed,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
        "early_stopping_enabled": early_stop,
        "max_epochs": 500,
        "epochs_completed": epochs,
        "primary_baseline": "published_multi_fraudgt",
        "edge_ff_chunk_size": 0,
        "edge_ff_checkpoint": False,
        "lambda_cons": 0.0,
        "history_k": 4,
        "use_relation_types": True,
        "val_selected_test_f1": selected,
        "val_selected_epoch": 300,
        "raw_best_test_f1": selected + 0.02,
        "raw_best_epoch": 400,
        "published_multi_fraudgt_val_selected_test_f1": published,
        "delta_val_selected_vs_published_multi_fraudgt": selected - published,
        "elapsed_seconds": 30.0,
        "training_seconds": 20.0,
        "inference_seconds": 8.0,
        "parameter_count": 1000,
        "peak_gpu_memory_bytes": 2000,
        "git_commit": f"commit{seed}",
        "config": str(task / "config.yaml"),
        "checkpoint": str(task / "best_val.ckpt"),
        "loader_audit": [
            {
                "split": split,
                "shuffle": True,
                "iter_per_epoch": 256,
                "generator": None,
            }
            for split in ("train", "val", "test")
        ],
        "config_snapshot": {
            "seed": seed,
            "dataset": {
                "name": "Small-LI",
                "reverse_mp": True,
                "add_ports": True,
                "tier_evidence": True,
            },
            "model": {"type": "CDVTModel"},
            "cdvt": {
                "variant": "dual_view",
                "history_k": 4,
                "lambda_cons": 0.0,
                "use_relation_types": True,
            },
            "optim": {"max_epoch": 500},
            "train": {
                "sampler": "link_neighbor",
                "add_ego_id": True,
                "iter_per_epoch": 256,
                "batch_size": 2048,
            },
            "val": {
                "sampler": "link_neighbor",
                "iter_per_epoch": 256,
                "fixed_target_panel": False,
            },
            "gt": {
                "edge_ff_chunk_size": 0,
                "edge_ff_checkpoint": False,
            },
        },
    }
    (task / "manifest.json").write_text(json.dumps(payload))


class CDVTSmallLIMultiseedTest(unittest.TestCase):
    def test_builds_descriptive_three_seed_summary(self):
        with tempfile.TemporaryDirectory() as seed42_dir, \
                tempfile.TemporaryDirectory() as multiseed_dir:
            seed42_root = Path(seed42_dir)
            multiseed_root = Path(multiseed_dir)
            values = (0.45, 0.48, 0.51)
            write_manifest(seed42_root, 42, values[0])
            write_manifest(multiseed_root, 43, values[1])
            write_manifest(multiseed_root, 44, values[2])
            summary = build_summary(seed42_root, multiseed_root)
        self.assertEqual(summary["seeds"], [42, 43, 44])
        self.assertAlmostEqual(
            summary["mean_val_selected_test_f1"],
            statistics.fmean(values),
        )
        self.assertAlmostEqual(
            summary["sample_std_val_selected_test_f1"],
            statistics.stdev(values),
        )
        self.assertTrue(summary["seed42_is_lowest"])
        self.assertTrue(summary["observed_range_gt_0_005"])

    def test_rejects_incomplete_or_early_stopped_seed(self):
        with tempfile.TemporaryDirectory() as seed42_dir, \
                tempfile.TemporaryDirectory() as multiseed_dir:
            seed42_root = Path(seed42_dir)
            multiseed_root = Path(multiseed_dir)
            write_manifest(seed42_root, 42, 0.45)
            write_manifest(multiseed_root, 43, 0.48, epochs=120)
            write_manifest(
                multiseed_root, 44, 0.51, early_stop=True)
            with self.assertRaises(ValueError):
                build_summary(seed42_root, multiseed_root)

    def test_launcher_freezes_everything_except_seed(self):
        source = Path(
            "run/cdvt_small_li_multiseed_queue.sh").read_text()
        self.assertIn("seeds=(43 44)", source)
        self.assertIn("CDVT_GPUS:-1 4", source)
        self.assertIn("--variant multi_cdvt", source)
        self.assertIn("--max-epochs 500", source)
        self.assertIn("--disable-early-stop", source)
        self.assertIn("model or frozen protocol differs", source)
        self.assertNotIn("multi_account_only", source)


if __name__ == "__main__":
    unittest.main()
