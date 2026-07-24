import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "tier_phase2a_p0_result_audit.py"
DATASETS = ("Small-LI", "Large-LI")
FAMILIES = ("structure", "temporal", "flow_role")
SELECTIONS = ("recent", "role_motif")


def stats(passed, calibration=False):
    corrected = 60 if passed else 20
    broken = 20
    changed = corrected + broken
    return {
        "a2_errors": 200,
        "changed_predictions": changed,
        "corrected_predictions": corrected,
        "broken_predictions": broken,
        "corrected_minus_broken": corrected - broken,
        "corrected_to_broken_ratio": corrected / broken,
        "corrected_without_breaks": False,
        "correction_rate_on_a2_errors": corrected / 200,
        "a2_same_batch_f1": 0.40,
        "routed_same_batch_f1": 0.45 if passed else 0.39,
        "delta_paired_f1": 0.05 if passed else -0.01,
        "add_interventions": changed // 2,
        "remove_interventions": changed - changed // 2,
    }


def manifest(dataset, family, selection, passed=False):
    baseline = {
        "Small-LI": (0.46247, 0.50667, 42),
        "Large-LI": (0.30108, 0.44720, 44),
    }[dataset]
    return {
        "dataset": dataset,
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
        "add_probe_fit": {"trained": True},
        "remove_probe_fit": {"trained": True},
        "calibration_statistics": stats(passed, calibration=True),
        "validation_statistics": stats(passed),
        "test_statistics": stats(passed),
        "calibration_policy_passed": passed,
        "validation_policy_passed": passed,
        "test_policy_passed": passed,
        "qualification_decision": "pass" if passed else "fail",
        "calibration_required_changed_predictions": 10,
        "validation_required_changed_predictions": 50,
        "test_required_changed_predictions": 50,
        "calibration_eligible_policies": 1 if passed else 0,
        "result_scope": (
            "optimistic_in_sample_teacher_feasibility_probe_"
            "not_headline_result"
        ),
    }


class TierPhaseTwoAP0ResultAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results"
        self.md = Path(self.temp.name) / "audit.md"
        self.js = Path(self.temp.name) / "audit.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_matrix(self, passing=()):
        passing = set(passing)
        paths = {}
        for dataset, family, selection in product(
            DATASETS, FAMILIES, SELECTIONS
        ):
            directory = self.root / f"{dataset}_{family}_{selection}"
            directory.mkdir(parents=True)
            path = directory / "phase2a_p0_manifest.json"
            key = (dataset, family, selection)
            path.write_text(json.dumps(manifest(
                dataset, family, selection, key in passing)))
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
                str(self.md),
                "--output-json",
                str(self.js),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_audits_failed_matrix(self):
        self.write_matrix()
        self.invoke()
        payload = json.loads(self.js.read_text())
        self.assertEqual(len(payload["rows"]), 12)
        self.assertIn("not authorized", self.md.read_text())

    def test_cross_scale_pass_requires_both_datasets(self):
        self.write_matrix({
            ("Small-LI", "flow_role", "recent"),
            ("Large-LI", "flow_role", "role_motif"),
        })
        self.invoke()
        payload = json.loads(self.js.read_text())
        self.assertEqual(
            payload["family_summaries"]["flow_role"]["decision"],
            "cross_scale_pass",
        )

    def test_rejects_inconsistent_gate(self):
        paths = self.write_matrix()
        path = paths[("Small-LI", "structure", "recent")]
        item = json.loads(path.read_text())
        item["qualification_decision"] = "pass"
        path.write_text(json.dumps(item))
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
