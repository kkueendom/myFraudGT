import subprocess
import sys
import unittest
from pathlib import Path

from run.summarize_multi_model_dynamic_reliability import (
    a2_single_run_reversal,
    summarize_family,
)


class SummarizeMultiModelDynamicReliabilityTest(unittest.TestCase):
    def test_family_summary_does_not_count_inactive_as_mismatch(self):
        units = [
            {
                "dataset": "Small-LI",
                "events": 8,
                "mechanism_classification": "sensitive_but_harmful",
                "single_event_conclusion_reversal": False,
            },
            {
                "dataset": "Large-LI",
                "events": 8,
                "mechanism_classification": "inactive_evidence",
                "single_event_conclusion_reversal": False,
            },
        ]
        row = summarize_family("example", units)
        self.assertEqual(row["mismatch_unit_count"], 1)
        self.assertFalse(row["cross_scale_mismatch"])
        self.assertEqual(row["inactive_unit_count"], 1)

    def test_a2_reversal_requires_both_sides_of_warning_band(self):
        a2 = {
            "datasets": {
                "Small-LI": {
                    "diagnostic_delta_vs_historical_initial_a2": {
                        "min": -0.02,
                        "max": 0.03,
                    },
                },
                "Large-LI": {
                    "diagnostic_delta_vs_historical_initial_a2": {
                        "min": 0.01,
                        "max": 0.04,
                    },
                },
            },
        }
        self.assertEqual(a2_single_run_reversal(a2), ["Small-LI"])

    def test_script_entrypoint_can_import_repository_modules(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "run"
                    / "summarize_multi_model_dynamic_reliability.py"
                ),
                "--help",
            ],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

