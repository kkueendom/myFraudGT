import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch_geometric.data import HeteroData

from fraudGT.cdvt.event_graph import CausalEventGraphIndex
from fraudGT.cdvt.fusion import DualViewFusionClassifier
from run.cdvt_materialize_config import materialize
from run.cdvt_multi_summary import build_summary
from run.cdvt_phase1_screen import audit_multi_dataset


TASK = ("node", "to", "node")
REVERSE_TASK = ("node", "rev_to", "node")


def multi_base_config():
    return {
        "seed": 42,
        "dataset": {
            "name": "Small-LI",
            "add_ports": True,
            "reverse_mp": False,
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
    }


def write_multi_manifest(root, dataset, variant, f1):
    task = root / f"{dataset}_{variant}_seed42"
    task.mkdir(parents=True)
    architecture = "account_only" if variant == "multi_account_only" else "dual_view"
    config = multi_base_config()
    config["dataset"]["name"] = dataset
    config["dataset"]["reverse_mp"] = True
    config["model"]["type"] = "GTModel" if architecture == "account_only" else "CDVTModel"
    config["cdvt"]["variant"] = architecture
    payload = {
        "phase": "CDVT_multi_screen",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": architecture,
        "seed": 42,
        "account_backbone": "Multi-FraudGT",
        "reverse_mp": True,
        "add_ports": True,
        "add_ego_id": True,
        "lambda_cons": 0.0,
        "history_k": 4,
        "use_relation_types": True,
        "git_commit": "abc1234",
        "config": str(task / "config.yaml"),
        "checkpoint": str(task / "best_val.ckpt"),
        "val_selected_epoch": 80,
        "val_selected_test_f1": f1,
        "raw_best_epoch": 84,
        "raw_best_test_f1": f1 + 0.01,
        "parameter_count": 100,
        "peak_gpu_memory_bytes": 1000,
        "elapsed_seconds": 12.0,
        "loader_audit": [
            {"split": split, "shuffle": True, "generator": None}
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
        "config_snapshot": config,
    }
    (task / "manifest.json").write_text(json.dumps(payload))


class CDVTMultiScreenTest(unittest.TestCase):
    def test_materializer_forces_rmp_and_preserves_ports_and_ego(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yaml"
            output = root / "multi.yaml"
            base.write_text(yaml.safe_dump(multi_base_config(), sort_keys=False))
            payload = materialize(base, output, 42, "multi_cdvt")
            self.assertTrue(payload["dataset"]["reverse_mp"])
            self.assertTrue(payload["dataset"]["add_ports"])
            self.assertTrue(payload["train"]["add_ego_id"])
            self.assertEqual(payload["model"]["type"], "CDVTModel")
            self.assertEqual(payload["cdvt"]["variant"], "dual_view")

    def test_materializer_rejects_missing_multi_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yaml"
            payload = multi_base_config()
            payload["dataset"]["add_ports"] = False
            base.write_text(yaml.safe_dump(payload, sort_keys=False))
            with self.assertRaises(ValueError):
                materialize(base, root / "multi.yaml", 42, "multi_cdvt")

    def test_reverse_relation_audit_requires_exact_forward_reversal(self):
        data = HeteroData()
        data["node"].x = torch.zeros(3, 2)
        forward = torch.tensor([[0, 1, 1], [1, 2, 0]])
        data[TASK].edge_index = forward
        data[REVERSE_TASK].edge_index = forward.flip(0)
        rows = audit_multi_dataset({split: data for split in ("train", "val", "test")})
        self.assertEqual(len(rows), 3)
        data[REVERSE_TASK].edge_index = forward[:, [1, 0, 2]].flip(0)
        with self.assertRaises(RuntimeError):
            audit_multi_dataset({split: data for split in ("train", "val", "test")})

    def test_target_alignment_and_future_leakage_remain_unchanged_with_multi_control(self):
        raw = torch.tensor([
            [1.0, 10.0, 0.0, 0.0],
            [2.0, 20.0, 0.0, 0.0],
            [3.0, 30.0, 1.0, 1.0],
            [4.0, 40.0, 1.0, 1.0],
        ])
        index = CausalEventGraphIndex(
            torch.tensor([[0, 1, 0, 2], [1, 2, 2, 0]]),
            torch.tensor([1, 2, 3, 4]),
            raw,
        )
        graph = index.query(torch.tensor([2]), k=4, hops=2)
        context = graph.node_edge_ids[graph.node_edge_ids != 2]
        self.assertTrue((context < 2).all())
        self.assertNotIn(2, context.tolist())
        account_ids = torch.tensor([30, 10, 99])
        event_ids = torch.tensor([99, 30, 10])
        positions = {int(value): i for i, value in enumerate(event_ids.tolist())}
        self.assertEqual(
            [positions[int(value)] for value in account_ids.tolist()],
            [1, 2, 0],
        )

    def test_multi_fusion_has_expected_shape_and_gradients(self):
        model = DualViewFusionClassifier(
            account_dim=48,
            num_currencies=2,
            num_payment_formats=2,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        graph = CausalEventGraphIndex(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([1, 2]),
            torch.tensor([[1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 1.0, 1.0]]),
        ).query(torch.tensor([1]), k=1, hops=1)
        account = torch.randn(1, 48, requires_grad=True)
        logits, _ = model(account, graph)
        self.assertEqual(tuple(logits.shape), (1,))
        logits.sum().backward()
        self.assertIsNotNone(account.grad)
        self.assertIsNotNone(model.cross_attention.in_proj_weight.grad)

    def test_multi_summary_gate_uses_paired_val_selected_f1_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset, account, cdvt in (
                ("Small-LI", 0.40, 0.42),
                ("Medium-LI", 0.40, 0.39),
                ("Large-LI", 0.40, 0.41),
            ):
                write_multi_manifest(root, dataset, "multi_account_only", account)
                write_multi_manifest(root, dataset, "multi_cdvt", cdvt)
            summary = build_summary(root)
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["advance_to_six_dataset_screen"])
            self.assertAlmostEqual(summary["mean_delta_val_selected"], 0.0066666667)
            self.assertEqual(summary["positive_count"], 2)

    def test_multi_summary_fails_when_mean_is_within_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("Small-LI", "Medium-LI", "Large-LI"):
                write_multi_manifest(root, dataset, "multi_account_only", 0.40)
                write_multi_manifest(root, dataset, "multi_cdvt", 0.404)
            summary = build_summary(root)
            self.assertFalse(summary["advance_to_six_dataset_screen"])
            self.assertFalse(summary["gate_criteria"]["mean_delta_gt_0_005"])

    def test_multi_cli_works_from_an_arbitrary_working_directory(self):
        script = Path(__file__).resolve().parents[1] / "run" / "cdvt_multi_summary.py"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_post_followup_is_the_only_allocator_and_has_nine_tasks(self):
        source = Path("run/cdvt_post_followup_queue.sh").read_text()
        self.assertIn("queue_complete.json", source)
        self.assertLess(
            source.index("queue_complete.json"),
            source.index("mkdir -p \"$additive_root"),
        )
        task_rows = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("'additive|")
            or line.strip().startswith("'multi|")
        ]
        self.assertEqual(len(task_rows), 9)
        self.assertIn('"training_tasks":9', source)
        self.assertIn("single_gpu_allocator", source)
        self.assertIn("multi_account_only", source)
        self.assertIn("multi_cdvt", source)


if __name__ == "__main__":
    unittest.main()
