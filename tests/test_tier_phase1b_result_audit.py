import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "tier_phase1b_result_audit.py"
DATASETS = ("Small-LI", "Large-LI")
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")


def manifest(dataset, family, selection, qualified=False):
    baseline = {
        "Small-LI": (0.46247, 0.50667, 42),
        "Large-LI": (0.30108, 0.44720, 44),
    }[dataset]
    selected = 0.50
    raw = 0.52
    corrected = 20 if qualified else 10
    broken = 10 if qualified else 20
    return {
        "dataset": dataset,
        "model": "TIER-EvidenceOnly-FamilyDecomposition",
        "variant": f"{family}_{selection}",
        "evidence_family": family,
        "evidence_selection": selection,
        "seed": baseline[2],
        "git_commit": "deadbeef",
        "sampling_protocol": "dynamic_random",
        "val_selected_test_f1": selected,
        "raw_best_test_f1": raw,
        "initial_a2_val_selected_test_f1": baseline[0],
        "initial_a2_raw_best_test_f1": baseline[1],
        "delta_val_selected_f1": selected - baseline[0],
        "delta_raw_best_f1": raw - baseline[1],
        "qualification_decision": "pass" if qualified else "fail",
        "selected_event": {
            "test": {
                "normal": {"f1": selected},
                "shuffled": {"f1": 0.45},
                "off": {"f1": 0.40},
            }
        },
        "same_batch_a2_diagnostic": {
            "sampled_instances": {
                "a2_errors": 100,
                "correction_rate_on_a2_errors": 0.20,
                "corrected_predictions": corrected,
                "broken_predictions": broken,
                "corrected_to_broken_ratio": corrected / broken,
                "changed_predictions": corrected + broken + 30,
            },
            "unique_edges": {},
        },
        "qualification_checks": {
            "overall_coverage": 0.95,
            "class_coverage": {"0": 0.95, "1": 0.90},
        },
    }


class TierPhaseOneBResultAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results"
        self.output_md = Path(self.temp.name) / "audit.md"
        self.output_json = Path(self.temp.name) / "audit.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_matrix(self, qualified_keys=()):
        qualified_keys = set(qualified_keys)
        paths = {}
        for dataset, family, selection in product(
            DATASETS, FAMILIES, SELECTIONS
        ):
            run = self.root / f"{dataset}_{family}_{selection}"
            run.mkdir(parents=True)
            path = run / "experiment_manifest.json"
            key = (dataset, family, selection)
            path.write_text(
                json.dumps(
                    manifest(
                        dataset,
                        family,
                        selection,
                        qualified=key in qualified_keys,
                    )
                )
            )
            paths[key] = path
        return paths

    def invoke(self):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.root),
                "--expected-tasks",
                "12",
                "--expected-commit",
                "deadbeef",
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

    def test_audits_complete_informative_but_unsafe_matrix(self):
        self.write_matrix()
        self.invoke()
        payload = json.loads(self.output_json.read_text())
        self.assertEqual(len(payload["rows"]), 12)
        for family in FAMILIES:
            summary = payload["family_summaries"][family]
            self.assertEqual(summary["counterfactually_active_tasks"], 4)
            self.assertEqual(summary["qualified_tasks"], 0)
            self.assertEqual(summary["decision"], "informative_but_unsafe")
        self.assertIn("Phase 2 is not authorized", self.output_md.read_text())

    def test_requires_qualification_on_both_scales(self):
        self.write_matrix(
            {
                ("Small-LI", "flow_role", "recent"),
                ("Large-LI", "flow_role", "role_motif"),
            }
        )
        self.invoke()
        payload = json.loads(self.output_json.read_text())
        summary = payload["family_summaries"]["flow_role"]
        self.assertEqual(summary["small_li_passes"], 1)
        self.assertEqual(summary["large_li_passes"], 1)
        self.assertEqual(summary["decision"], "cross_scale_pass")

    def test_rejects_inconsistent_stored_gate(self):
        paths = self.write_matrix()
        path = paths[("Small-LI", "structure", "recent")]
        item = json.loads(path.read_text())
        item["qualification_decision"] = "pass"
        path.write_text(json.dumps(item))
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
