import json
import unittest
from pathlib import Path


class CostarEvidenceDiagnosticsQueueTest(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]

    def test_spec_uses_dynamic_protocol_and_two_required_datasets(self):
        spec = json.loads((
            self.repo / "run" / "costar_evidence_diagnostics_spec.json"
        ).read_text())
        self.assertEqual(spec["sampling_protocol"], "dynamic_random")
        self.assertEqual(spec["required_last_epoch"], 499)
        self.assertEqual(
            [task["dataset"] for task in spec["tasks"]],
            ["Small-LI", "Large-LI"])

    def test_queue_reserves_gpu0_and_never_launches_training(self):
        source = (
            self.repo / "run" / "costar_evidence_diagnostics_queue.py"
        ).read_text()
        self.assertIn('"COSTAR_DIAG_GPUS", "4,6"', source)
        self.assertIn('if 0 in GPU_ALLOWLIST', source)
        self.assertIn('costar_evidence_diagnostics.py', source)
        self.assertNotIn('fraudGT.main', source)
        self.assertNotIn('optim.max_epoch', source)


if __name__ == "__main__":
    unittest.main()
