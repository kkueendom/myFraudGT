import json
import tempfile
import unittest
from pathlib import Path

from run.gtf1c_phase0_v2 import V2_REGIMES
from run.summarize_trefic_phase0 import (
    build_aggregate,
)
from run.trefic_phase0 import METHODS


def method(rate, false_rate=0.0):
    trials = 3
    return {
        "trials": trials,
        "qualified": round(trials * rate),
        "qualification_rate": rate,
        "violations": round(trials * false_rate),
        "false_qualification_rate": false_rate,
        "conditional_violation_rate": false_rate,
        "practical_failures": round(trials * false_rate),
        "evaluation_delta_mean": 0.02,
        "evaluation_delta_median": 0.02,
        "evaluation_changes_mean": 50.0,
        "oracle_fraction_mean": 0.5,
        "oracle_fraction_median": 0.5,
    }


class TREFICSummaryTest(unittest.TestCase):
    def test_registered_good_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            paths = []
            for offset, regime in enumerate(V2_REGIMES):
                directory = root / regime
                directory.mkdir()
                positive = regime.endswith("positive")
                datasets = {}
                records = []
                for dataset in ("Small-LI", "Large-LI"):
                    datasets[dataset] = {
                        "trials": 3,
                        "oracle_opportunity_rate": 1.0,
                        "methods": {
                            name: method(
                                1.0 if positive else (
                                    1.0 if name in (
                                        "row_net", "gtprc_row_harm")
                                    else 0.0
                                ),
                                0.2 if (
                                    not positive
                                    and name in (
                                        "row_net", "gtprc_row_harm")
                                ) else 0.0,
                            )
                            for name in METHODS
                        },
                        "controls": {
                            "shuffled": method(0.0),
                            "harmful": method(0.0),
                        },
                    }
                    for fold in range(3):
                        chosen = {
                            name: (
                                0 if positive or name in (
                                    "row_net", "gtprc_row_harm")
                                else -1
                            )
                            for name in METHODS
                        }
                        records.append({
                            "dataset": dataset,
                            "replicate": 0,
                            "evaluation_fold": fold,
                            "chosen_candidate": chosen,
                            "oracle_changed": 50,
                            "normal_evaluations": {
                                name: {
                                    "changed": 50,
                                    "paired_f1_delta": (
                                        0.02 if positive else -0.02),
                                }
                                for name in METHODS
                            },
                        })
                records_path = directory / "trefic_trials.jsonl"
                records_path.write_text("".join(
                    json.dumps(row) + "\n" for row in records))
                manifest = {
                    "regime": regime,
                    "method_version": 1,
                    "git_commit": "abc123",
                    "seed": 75001 + offset,
                    "replicates": 1,
                    "sampling_protocol": "dynamic_random",
                    "validation_loader_iterations": 0,
                    "test_loader_iterations": 0,
                    "locked_candidate_count": 3,
                    "reference_candidate_delta": 0.05 / 3,
                    "trefic_endpoint_delta": 0.05 / 6,
                    "rho_lower": 0.0,
                    "delta": 0.05,
                    "datasets": datasets,
                    "trial_records": str(records_path),
                }
                path = directory / "trefic_phase0_manifest.json"
                path.write_text(json.dumps(manifest))
                manifests.append(manifest)
                paths.append(path)
            aggregate = build_aggregate(manifests, paths, 1)
            self.assertTrue(aggregate["identity_audit"]["passed"])
            self.assertTrue(aggregate["gate"]["passed"])
            self.assertEqual(
                aggregate["decision"],
                "PROCEED_TO_FRESH_FORMAL_PREREGISTRATION",
            )


if __name__ == "__main__":
    unittest.main()
