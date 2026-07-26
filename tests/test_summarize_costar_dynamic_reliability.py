import subprocess
import sys
import unittest
from pathlib import Path


class SummarizeCostarDynamicReliabilityTest(unittest.TestCase):
    def test_script_entrypoint_can_import_repository_modules(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "run"
                    / "summarize_costar_dynamic_reliability.py"
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

