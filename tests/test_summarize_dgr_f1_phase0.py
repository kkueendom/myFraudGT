import unittest

from run.dgr_f1_phase0 import SCENARIO_TARGETS
from run.summarize_dgr_f1_phase0 import build_aggregate


def method(target):
    return {
        "decision_counts": {
            "REPLICABLE_IMPROVEMENT": (
                1 if target == "REPLICABLE_IMPROVEMENT" else 0),
            "REPLICABLE_HARM": (
                1 if target == "REPLICABLE_HARM" else 0),
            "INSUFFICIENT_INFORMATION": (
                1 if target == "INSUFFICIENT_INFORMATION" else 0),
        },
        "improvement_rate": (
            1.0 if target == "REPLICABLE_IMPROVEMENT" else 0.0),
        "harm_rate": (
            1.0 if target == "REPLICABLE_HARM" else 0.0),
        "insufficient_rate": (
            1.0 if target == "INSUFFICIENT_INFORMATION" else 0.0),
        "median_stopping_checkpoint": (
            None if target == "INSUFFICIENT_INFORMATION" else 32.0),
    }


class DGRF1SummaryTest(unittest.TestCase):
    def test_registered_good_fixture_passes(self):
        manifests = []
        for offset, (scenario, target) in enumerate(
                SCENARIO_TARGETS.items()):
            templates = {}
            for template in ("Small-LI", "Large-LI"):
                methods = {
                    name: method(target)
                    for name in (
                        "one_stream", "row_iid", "mean_only",
                        "replication_only", "dgr_f1")
                }
                if target == "INSUFFICIENT_INFORMATION":
                    methods["one_stream"] = method(
                        "REPLICABLE_IMPROVEMENT")
                templates[template] = {
                    "registered_target": target,
                    "population_target": {"decision": target},
                    "identity_failures": 0,
                    "reference_identity_failures": 0,
                    "simultaneous_mean_coverage_rate": 0.95,
                    "methods": methods,
                }
            manifests.append({
                "experiment": "DGR_F1_Phase0_controlled_development",
                "method_version": 1,
                "scenario": scenario,
                "registered_target": target,
                "seed": 76001 + offset,
                "git_commit": "abc123",
                "replicates": 1,
                "streams_per_replicate": 64,
                "checkpoints": [8, 16, 32, 64],
                "epsilon": 0.005,
                "pi0": 0.75,
                "delta": 0.05,
                "endpoint_delta": 0.05 / 12,
                "validation_loader_iterations": 0,
                "test_loader_iterations": 0,
                "templates": templates,
            })
        aggregate = build_aggregate(manifests, 1)
        self.assertTrue(aggregate["gate"]["passed"])
        self.assertEqual(
            aggregate["decision"],
            "PROCEED_TO_PROSPECTIVE_FROZEN_CHECKPOINT_PLAN",
        )


if __name__ == "__main__":
    unittest.main()
