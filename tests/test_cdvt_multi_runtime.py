import json
import math
import tempfile
import unittest
from pathlib import Path

from run.cdvt_multi_runtime_summary import build_summary


def write_benchmark(root, variant, latency, parameters, memory):
    architecture = (
        "account_only" if variant == "multi_account_only" else "dual_view")
    output = root / f"Small-LI_{variant}_seed42"
    output.mkdir(parents=True)
    payload = {
        "phase": "CDVT_multi_runtime_quick",
        "benchmark_mode": "normal_only_end_to_end_inference",
        "sampling_protocol": "dynamic_random",
        "dataset": "Small-LI",
        "variant": variant,
        "architecture_variant": architecture,
        "account_backbone": "Multi-FraudGT",
        "seed": 42,
        "checkpoint_loaded": False,
        "device_name": "NVIDIA GeForce RTX 2080 Ti",
        "repeats": 3,
        "batches_per_repeat": 32,
        "repeat_rows": [{"repeat": index + 1} for index in range(3)],
        "mean_seconds_per_batch": latency,
        "std_seconds_per_batch": 0.01,
        "mean_targets_per_second": 2048 / latency,
        "std_targets_per_second": 1.0,
        "parameter_count": parameters,
        "peak_gpu_memory_bytes": memory,
        "loader_audit": [
            {"split": split, "shuffle": True, "generator": None}
            for split in ("train", "val", "test")
        ],
    }
    (output / "benchmark.json").write_text(json.dumps(payload))


class MultiRuntimeSummaryTest(unittest.TestCase):
    def test_summary_computes_matched_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_benchmark(
                root, "multi_account_only", 0.5, 200_000, 2_000_000)
            write_benchmark(
                root, "multi_cdvt", 1.25, 300_000, 3_000_000)
            result = build_summary(root, ["Small-LI"])
            self.assertTrue(result["complete"])
            self.assertEqual(result["sampling_protocol"], "dynamic_random")
            self.assertEqual(len(result["rows"]), 2)
            comparison = result["comparisons"][0]
            self.assertTrue(math.isclose(
                comparison["parameter_ratio_cdvt_vs_multi_fraudgt"], 1.5))
            self.assertTrue(math.isclose(
                comparison["memory_ratio_cdvt_vs_multi_fraudgt"], 1.5))
            self.assertTrue(math.isclose(
                comparison["latency_ratio_cdvt_vs_multi_fraudgt"], 2.5))
            self.assertTrue(math.isclose(
                comparison["latency_overhead_percent"], 150.0))
            self.assertTrue(math.isclose(
                comparison["throughput_ratio_cdvt_vs_multi_fraudgt"], 0.4))

    def test_summary_rejects_incomplete_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_benchmark(
                root, "multi_account_only", 0.5, 200_000, 2_000_000)
            with self.assertRaises(FileNotFoundError):
                build_summary(root, ["Small-LI"])


if __name__ == "__main__":
    unittest.main()
