import json
import tempfile
import unittest
from pathlib import Path

from run.cdvt_phase1_audit import audit


class CDVTPhase1AuditTest(unittest.TestCase):
    def write_result(self, root, dataset, variant, score, baseline, mechanism):
        task = root / f"{dataset}_{variant}_seed42"
        task.mkdir()
        event = {
            "epoch": 80,
            "val_f1": score,
            "test": {
                "normal": {"f1": score},
                "shuffled": {"f1": score - mechanism},
                "off": {"f1": score - mechanism},
            },
        }
        supporting = dict(event)
        supporting["epoch"] = 84
        supporting["val_f1"] = score - 0.005
        (task / "trajectory.jsonl").write_text(
            json.dumps(event) + "\n" + json.dumps(supporting) + "\n")
        payload = {
            "sampling_protocol": "dynamic_random",
            "dataset": dataset,
            "variant": variant,
            "seed": 42,
            "git_commit": "abc12345",
            "checkpoint": str(task / "best_val.ckpt"),
            "val_selected_epoch": 80,
            "val_selected_test_f1": score,
            "raw_best_test_f1": score + 0.001,
            "parameter_count": 100,
            "elapsed_seconds": 10.0,
            "best_event": event,
        }
        (task / "manifest.json").write_text(json.dumps(payload))

    def test_advances_only_when_all_preregistered_gates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baselines = {"Small-LI": 0.46247, "Large-LI": 0.30108}
            for dataset, baseline in baselines.items():
                for variant in (
                    "account_only", "event_only", "dual_view", "full_cdvt"
                ):
                    score = baseline + (0.006 if variant == "account_only" else 0.02)
                    self.write_result(
                        root, dataset, variant, score, baseline,
                        0.02 if variant == "full_cdvt" else 0.0)
            result = audit(root, "abc12345")
            self.assertTrue(result["advancement"]["advance_to_phase2"])

    def test_rejects_missing_matrix_member(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                audit(Path(directory))


if __name__ == "__main__":
    unittest.main()
