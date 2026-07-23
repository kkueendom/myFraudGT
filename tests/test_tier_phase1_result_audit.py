import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "tier_phase1_result_audit.py"


def manifest(dataset="Small-LI", selection="recent"):
    baseline = {
        "Small-LI": (0.46247, 0.50667),
        "Large-LI": (0.30108, 0.44720),
    }[dataset]
    selected = 0.50
    raw = 0.52
    return {
        "dataset": dataset,
        "evidence_selection": selection,
        "seed": 42,
        "git_commit": "deadbeef",
        "sampling_protocol": "dynamic_random",
        "val_selected_test_f1": selected,
        "raw_best_test_f1": raw,
        "initial_a2_val_selected_test_f1": baseline[0],
        "initial_a2_raw_best_test_f1": baseline[1],
        "delta_val_selected_f1": selected - baseline[0],
        "delta_raw_best_f1": raw - baseline[1],
        "qualification_decision": "pass",
        "selected_event": {
            "test": {
                "normal": {"f1": selected},
                "shuffled": {"f1": 0.45},
                "off": {"f1": 0.40},
            }
        },
        "same_batch_a2_diagnostic": {
            "sampled_instances": {
                "correction_rate_on_a2_errors": 0.2,
                "corrected_to_broken_ratio": 2.0,
                "changed_predictions": 60,
            }
        },
    }


class TierPhaseOneAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results"
        self.run = self.root / "Small-LI_all_recent_seed42_deadbeef"
        self.run.mkdir(parents=True)
        self.path = self.run / "experiment_manifest.json"
        self.output_md = Path(self.temp.name) / "audit.md"
        self.output_json = Path(self.temp.name) / "audit.json"

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, expected=1):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.root),
                "--expected-tasks",
                str(expected),
                "--output-md",
                str(self.output_md),
                "--output-json",
                str(self.output_json),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_writes_same_metric_summary(self):
        self.path.write_text(json.dumps(manifest()))
        self.invoke()
        content = self.output_md.read_text()
        self.assertIn("Val-selected Test F1", content)
        self.assertIn("Fixed-panel A2: excluded", content)
        payload = json.loads(self.output_json.read_text())
        self.assertEqual(payload["sampling_protocol"], "dynamic_random")
        self.assertAlmostEqual(
            payload["rows"][0]["selected_delta"], 0.50 - 0.46247
        )

    def test_rejects_cross_metric_delta(self):
        item = manifest()
        item["delta_raw_best_f1"] = (
            item["raw_best_test_f1"]
            - item["initial_a2_val_selected_test_f1"]
        )
        self.path.write_text(json.dumps(item))
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
