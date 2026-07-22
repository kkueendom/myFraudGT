import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DynamicRandomResultAuditTest(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "results"
        self.spec = Path(self.temp_dir.name) / "spec.json"
        self.manifest = Path(self.temp_dir.name) / "manifest.jsonl"
        self.spec.write_text(json.dumps({
            "sampling_protocol": "dynamic_random",
            "method_tag": "CPTR",
            "variant": "CPTR",
            "config_template": "configs/cptr/AML-{dataset}.yaml",
            "tasks": [["Small-LI", 42]],
        }))

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_run(self, last_epoch):
        seed_dir = (
            self.root /
            "AML-Small-LI-CPTRDynamic500-Seed42-deadbeef-gpu1" / "42")
        for split in ("train", "val", "test"):
            (seed_dir / split).mkdir(parents=True, exist_ok=True)
        (seed_dir / "train" / "stats.json").write_text(
            json.dumps({"epoch": last_epoch}) + "\n")
        (seed_dir / "val" / "stats.json").write_text(
            json.dumps({"epoch": last_epoch, "f1": 0.5}) + "\n")
        (seed_dir / "test" / "stats.json").write_text(
            json.dumps({"epoch": last_epoch, "f1": 0.48}) + "\n")

    def audit(self):
        return subprocess.run([
            sys.executable,
            str(self.repo / "run" / "dynamic_random_result_audit.py"),
            "--spec", str(self.spec),
            "--root", str(self.root),
            "--commit", "deadbeef",
            "--epoch-limit", "499",
            "--write-manifest", str(self.manifest),
        ], cwd=self.repo, text=True, capture_output=True, check=True)

    def test_partial_run_is_visible_but_not_formally_completed(self):
        self.write_run(123)
        result = self.audit()
        self.assertIn("Small-LI\tCPTR\t42\t123", result.stdout)
        self.assertNotIn("summary metric=", result.stdout)
        self.assertEqual(self.manifest.read_text(), "")

    def test_full_run_enters_summary_and_manifest(self):
        self.write_run(499)
        result = self.audit()
        self.assertIn("completed=1/1", result.stdout)
        record = json.loads(self.manifest.read_text())
        self.assertEqual(record["config"], "configs/cptr/AML-Small-LI.yaml")
        self.assertEqual(record["sampling_protocol"], "dynamic_random")


if __name__ == "__main__":
    unittest.main()
