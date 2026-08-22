import json
import math
import tempfile
import unittest
from pathlib import Path

from run.cdvt_multi_runtime_summary import build_summary


def write_benchmark(
        root, variant, latency, parameters, memory,
        evidence_tier="quick", repeats=3):
    architecture = (
        "account_only" if variant == "multi_account_only" else "dual_view")
    output = root / f"Small-LI_{variant}_seed42"
    output.mkdir(parents=True)
    payload = {
        "phase": (
            "CDVT_multi_runtime_formal_256"
            if evidence_tier == "formal_256"
            else "CDVT_multi_runtime_quick"),
        "evidence_tier": evidence_tier,
        "benchmark_mode": "normal_only_end_to_end_inference",
        "sampling_protocol": "dynamic_random",
        "dataset": "Small-LI",
        "variant": variant,
        "architecture_variant": architecture,
        "account_backbone": "Multi-FraudGT",
        "seed": 42,
        "checkpoint_loaded": False,
        "device_name": "NVIDIA GeForce RTX 2080 Ti",
        "repeats": repeats,
        "batches_per_repeat": 32,
        "total_measured_batches": repeats * 32,
        "loader_restart_policy": (
            "continue_dynamic_random_after_loader_exhaustion"
            if evidence_tier == "formal_256" else "none"),
        "total_loader_restarts": 0,
        "repeat_rows": [
            {"repeat": index + 1, "loader_restarts": 0}
            for index in range(repeats)],
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
    def test_quick_launcher_parameterizes_the_li_dataset(self):
        source = Path("run/cdvt_multi_runtime_quick.sh").read_text()
        self.assertIn('dataset="${CDVT_DATASET:-Small-LI}"', source)
        self.assertIn(
            "Small-LI|Small-HI|Medium-LI|Medium-HI|Large-LI|Large-HI",
            source,
        )
        self.assertIn("multi_published_500e_large_hi_recovery_decabdb", source)
        self.assertIn('AML-${dataset}-${variant}-seed42.yaml', source)
        self.assertIn('--datasets "$dataset"', source)

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

    def test_quick_summary_accepts_pre_tier_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, latency in (
                    ("multi_account_only", 0.5),
                    ("multi_cdvt", 1.25)):
                write_benchmark(
                    root, variant, latency, 200_000, 2_000_000)
                path = root / f"Small-LI_{variant}_seed42/benchmark.json"
                payload = json.loads(path.read_text())
                payload.pop("evidence_tier")
                path.write_text(json.dumps(payload))
            result = build_summary(root, ["Small-LI"])
            self.assertTrue(result["complete"])

    def test_formal_summary_requires_256_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_benchmark(
                root, "multi_account_only", 0.5, 200_000, 2_000_000,
                evidence_tier="formal_256", repeats=8)
            write_benchmark(
                root, "multi_cdvt", 1.25, 300_000, 3_000_000,
                evidence_tier="formal_256", repeats=8)
            result = build_summary(
                root, ["Small-LI"], evidence_tier="formal_256")
            self.assertEqual(result["evidence_tier"], "formal_256")
            self.assertTrue(result["complete"])

            payload_path = (
                root / "Small-LI_multi_cdvt_seed42" / "benchmark.json")
            payload = json.loads(payload_path.read_text())
            payload["total_measured_batches"] = 96
            payload_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                build_summary(
                    root, ["Small-LI"], evidence_tier="formal_256")

    def test_formal_launcher_is_six_dataset_serial_256_batch(self):
        source = Path(
            "run/cdvt_multi_runtime_formal_256.sh").read_text()
        self.assertIn("Small-LI Small-HI Medium-LI Medium-HI", source)
        self.assertIn("Large-LI Large-HI", source)
        self.assertIn("repeats=8", source)
        self.assertIn("batches_per_repeat=32", source)
        self.assertIn("--evidence-tier formal_256", source)
        self.assertIn('wait_pattern="${CDVT_WAIT_PROCESS_PATTERN', source)


if __name__ == "__main__":
    unittest.main()
