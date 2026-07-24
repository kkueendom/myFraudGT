import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "tier_phase1c_result_audit.py"
DATASETS = ("Small-LI", "Large-LI")
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")


def stats(passed):
    corrected = 60 if passed else 20
    broken = 20
    return {
        "a2_errors": 200,
        "changed_predictions": corrected + broken,
        "corrected_predictions": corrected,
        "broken_predictions": broken,
        "corrected_minus_broken": corrected - broken,
        "corrected_to_broken_ratio": corrected / broken,
        "corrected_without_breaks": False,
        "correction_rate_on_a2_errors": corrected / 200,
        "a2_same_batch_f1": 0.40,
        "routed_same_batch_f1": 0.45 if passed else 0.39,
        "delta_paired_f1": 0.05 if passed else -0.01,
    }


def manifest(dataset, family, selection, passed=False):
    baseline = {
        "Small-LI": (0.46247, 0.50667, 42),
        "Large-LI": (0.30108, 0.44720, 44),
    }[dataset]
    validation = stats(passed)
    test = stats(passed)
    return {
        "dataset": dataset,
        "model": "TIER-HighPrecisionIntervention-Diagnostic",
        "variant": f"{family}_{selection}",
        "evidence_family": family,
        "evidence_selection": selection,
        "seed": baseline[2],
        "git_commit": "deadbeef",
        "parent_evidence_commit": "feedface",
        "sampling_protocol": "dynamic_random",
        "loader_audit": [
            {
                "split": split,
                "shuffle": True,
                "loader_generator": None,
                "sampler_generator": None,
            }
            for split in ("train", "val", "test")
        ],
        "initial_a2_val_selected_test_f1": baseline[0],
        "initial_a2_raw_best_test_f1": baseline[1],
        "raw_best_test_f1": None,
        "validation_statistics": validation,
        "test_statistics": test,
        "validation_policy_passed": passed,
        "qualification_decision": "pass" if passed else "fail",
        "validation_required_changed_predictions": 50,
        "test_required_changed_predictions": 50,
        "validation_eligible_policies": 1 if passed else 0,
        "selected_policy": {"direction": "both"},
        "delta_vs_initial_a2_val_selected_f1": (
            test["routed_same_batch_f1"] - baseline[0]
        ),
        "result_scope": (
            "paired_same_batch_mechanism_diagnostic_not_headline_result"
        ),
    }


class TierPhaseOneCResultAuditTest(unittest.TestCase):
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
            path = run / "phase1c_manifest.json"
            key = (dataset, family, selection)
            path.write_text(json.dumps(manifest(
                dataset,
                family,
                selection,
                passed=key in qualified_keys,
            )))
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
                "--parent-commit",
                "feedface",
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

    def test_audits_complete_failed_matrix(self):
        self.write_matrix()
        self.invoke()
        payload = json.loads(self.output_json.read_text())
        self.assertEqual(len(payload["rows"]), 12)
        self.assertTrue(all(
            item["decision"] == "not_separable_at_registered_coverage"
            for item in payload["family_summaries"].values()
        ))
        self.assertIn(
            "not authorized", self.output_md.read_text())

    def test_requires_pass_on_both_scales(self):
        self.write_matrix({
            ("Small-LI", "flow_role", "recent"),
            ("Large-LI", "flow_role", "role_motif"),
        })
        self.invoke()
        payload = json.loads(self.output_json.read_text())
        self.assertEqual(
            payload["family_summaries"]["flow_role"]["decision"],
            "cross_scale_pass",
        )

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
