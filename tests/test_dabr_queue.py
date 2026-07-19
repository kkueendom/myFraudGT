import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


QUEUE_PATH = (
    Path(__file__).resolve().parents[1] / "run" /
    "dabr_pair_strength_queue.py")


def load_queue(strength):
    spec = importlib.util.spec_from_file_location(
        f"dabr_queue_{strength}", QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {
            "DABR_ROUTE_STRENGTH": str(strength),
            "DABR_GPU_ALLOWLIST": "3,4",
    }, clear=False):
        spec.loader.exec_module(module)
    return module


class DabrQueueTest(unittest.TestCase):
    def test_strength_isolated_namespaces(self):
        s25 = load_queue(0.25)
        s50 = load_queue(0.50)
        self.assertEqual(s25.STRENGTH_TAG, "s25")
        self.assertEqual(s50.STRENGTH_TAG, "s50")
        self.assertIn("DABRs25", s25.run_stem("Small-LI", 42, "deadbeef"))
        self.assertIn("DABRs50", s50.run_stem("Small-LI", 42, "deadbeef"))
        self.assertNotEqual(
            s25.run_stem("Small-LI", 42, "deadbeef"),
            s50.run_stem("Small-LI", 42, "deadbeef"))

    def test_gpu_zero_is_never_available(self):
        queue = load_queue(0.25)
        queue.GPU_ALLOWLIST = (0, 3)
        with patch.object(queue, "gpu_compute_pids", return_value=[]), \
                patch.object(queue, "gpu_free_and_util",
                             return_value=(22000, 0)):
            self.assertEqual(queue.idle_gpus([]), [3])


if __name__ == "__main__":
    unittest.main()
