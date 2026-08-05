import json
import tempfile
import unittest
from pathlib import Path

from run.cdvt_multi_published_summary import build_summary
from run.cdvt_protocol import MULTI_FRAUDGT_PAPER


DATASETS = tuple(MULTI_FRAUDGT_PAPER)


def write_manifest(root, dataset, delta, *, epochs=500, early_stop=False):
    task = root / f"{dataset}_multi_cdvt_seed42"
    task.mkdir(parents=True)
    chunk_size = {
        "Large-LI": 65536,
        "Large-HI": 32768,
    }.get(dataset, 0)
    checkpointed = chunk_size > 0
    selected = MULTI_FRAUDGT_PAPER[dataset] + delta
    payload = {
        "phase": "CDVT_multi_published_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": "multi_cdvt",
        "architecture_variant": "dual_view",
        "lambda_cons": 0.0,
        "history_k": 4,
        "use_relation_types": True,
        "edge_ff_chunk_size": chunk_size,
        "edge_ff_checkpoint": checkpointed,
        "early_stopping_enabled": early_stop,
        "max_epochs": 500,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
        "seed": 42,
        "git_commit": "formal123",
        "config": str(task / "config.yaml"),
        "checkpoint": str(task / "best_val.ckpt"),
        "val_selected_epoch": 399,
        "val_selected_test_f1": selected,
        "raw_best_epoch": 449,
        "raw_best_test_f1": selected + 0.01,
        "published_multi_fraudgt_val_selected_test_f1": (
            MULTI_FRAUDGT_PAPER[dataset]),
        "delta_val_selected_vs_published_multi_fraudgt": delta,
        "primary_baseline": "published_multi_fraudgt",
        "epochs_completed": epochs,
        "elapsed_seconds": 30.0,
        "training_seconds": 20.0,
        "inference_seconds": 8.0,
        "parameter_count": 1000,
        "peak_gpu_memory_bytes": 2000,
        "loader_audit": [
            {
                "split": split,
                "shuffle": True,
                "iter_per_epoch": 256,
                "generator": None,
            }
            for split in ("train", "val", "test")
        ],
        "multi_dataset_audit": [
            {
                "split": split,
                "forward_edges": 10,
                "reverse_edges": 10,
            }
            for split in ("train", "val", "test")
        ],
        "config_snapshot": {
            "dataset": {
                "name": dataset,
                "reverse_mp": True,
                "add_ports": True,
                "tier_evidence": True,
            },
            "train": {
                "sampler": "link_neighbor",
                "add_ego_id": True,
                "iter_per_epoch": 256,
                "batch_size": 2048,
            },
            "val": {
                "iter_per_epoch": 256,
                "fixed_target_panel": False,
            },
            "model": {"type": "CDVTModel"},
            "cdvt": {
                "variant": "dual_view",
                "history_k": 4,
                "lambda_cons": 0.0,
                "use_relation_types": True,
            },
            "optim": {"max_epoch": 500},
            "gt": {
                "edge_ff_chunk_size": chunk_size,
                "edge_ff_checkpoint": checkpointed,
            },
        },
    }
    (task / "manifest.json").write_text(json.dumps(payload))


class CDVTMultiPublishedTest(unittest.TestCase):
    def build(self, deltas):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        for dataset, delta in zip(DATASETS, deltas):
            write_manifest(root, dataset, delta)
        return directory, root

    def test_clear_pass_requires_four_wins_and_mean_above_sampling_band(self):
        directory, root = self.build((0.02, 0.02, 0.02, 0.02, -0.005, -0.005))
        self.addCleanup(directory.cleanup)
        summary = build_summary(root)
        self.assertEqual(summary["positive_count"], 4)
        self.assertGreater(summary["mean_delta_val_selected"], 0.005)
        self.assertEqual(summary["gate_verdict"], "clear_pass")

    def test_positive_but_small_mean_is_boundary(self):
        directory, root = self.build((0.004, 0.004, 0.004, 0.004, 0.004, -0.004))
        self.addCleanup(directory.cleanup)
        summary = build_summary(root)
        self.assertEqual(summary["positive_count"], 5)
        self.assertEqual(summary["gate_verdict"], "boundary")
        self.assertEqual(summary["sampling_band_count"], 6)

    def test_fewer_than_three_wins_fails_even_with_positive_mean(self):
        directory, root = self.build((0.10, 0.10, -0.01, -0.01, -0.01, -0.01))
        self.addCleanup(directory.cleanup)
        summary = build_summary(root)
        self.assertGreater(summary["mean_delta_val_selected"], 0)
        self.assertEqual(summary["positive_count"], 2)
        self.assertEqual(summary["gate_verdict"], "failed")

    def test_rejects_early_stopped_or_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in DATASETS:
                write_manifest(root, dataset, 0.01)
            bad = root / "Small-LI_multi_cdvt_seed42" / "manifest.json"
            payload = json.loads(bad.read_text())
            payload["early_stopping_enabled"] = True
            payload["epochs_completed"] = 120
            bad.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                build_summary(root)

    def test_launcher_is_six_dataset_seed42_multi_cdvt_only(self):
        source = Path("run/cdvt_multi_published_500e_queue.sh").read_text()
        self.assertEqual(source.count("|42'"), 6)
        for dataset in DATASETS:
            self.assertIn(dataset, source)
        self.assertIn("--experiment-label multi_cdvt", source)
        self.assertNotIn("multi_account_only", source)
        self.assertIn("--max-epochs 500", source)
        self.assertIn("--disable-early-stop", source)
        self.assertIn("CDVT_GPUS:-0 1 2 3 4 5 6", source)
        self.assertIn("--edge-ff-chunk-size", source)
        self.assertIn("--edge-ff-checkpoint", source)

    def test_large_hi_recovery_is_from_epoch_zero_and_memory_only(self):
        source = Path(
            "run/cdvt_multi_published_large_hi_recovery.sh").read_text()
        self.assertIn("torch.OutOfMemoryError", source)
        self.assertIn("CDVT_EDGE_FF_CHUNK_SIZE:-32768", source)
        self.assertIn("--disable-early-stop", source)
        self.assertIn('"restart_epoch": 0', source)
        self.assertIn('"mathematical_definition_changed": False', source)

    def test_summary_can_read_large_hi_from_a_recovery_root(self):
        with tempfile.TemporaryDirectory() as main_directory, \
                tempfile.TemporaryDirectory() as recovery_directory:
            main_root = Path(main_directory)
            recovery_root = Path(recovery_directory)
            for dataset in DATASETS:
                target = recovery_root if dataset == "Large-HI" else main_root
                write_manifest(target, dataset, 0.01)
            summary = build_summary(
                main_root, large_hi_root=recovery_root)
            self.assertTrue(summary["complete"])
            self.assertEqual(
                Path(summary["large_hi_root"]), recovery_root.resolve())

    def test_runner_records_published_baseline_and_runtime(self):
        source = Path("run/cdvt_phase1_screen.py").read_text()
        self.assertIn("--disable-early-stop", source)
        self.assertIn("not args.disable_early_stop", source)
        self.assertIn(
            '"delta_val_selected_vs_published_multi_fraudgt"', source)
        self.assertIn('"training_seconds"', source)
        self.assertIn('"inference_seconds"', source)

    def test_gpu_smoke_has_a_non_repeating_single_batch_mode(self):
        source = Path("run/cdvt_phase0_smoke.py").read_text()
        self.assertIn("--single-batch-only", source)
        self.assertIn('"phase0_single_batch_gpu_smoke"', source)
        self.assertIn('"peak_gpu_memory_bytes"', source)
        self.assertIn("del batch, logits, labels, diagnostics, loss", source)

    def test_chunked_execution_avoids_a_full_output_cat(self):
        source = Path("fraudGT/layer/gt_layer.py").read_text()
        start = source.index("def memory_efficient_chunked_forward")
        end = source.index("\ndef memory_efficient_masked_forward", start)
        self.assertNotIn("torch.cat(outputs", source[start:end])
        self.assertIn("output[offset:next_offset]", source[start:end])
        self.assertIn("add_residual=True", source)
        self.assertIn(
            "return chunk + transformed if add_residual else transformed",
            source,
        )


if __name__ == "__main__":
    unittest.main()
