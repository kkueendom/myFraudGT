import unittest

from run.gtf1c_phase0_real_graph import REGIMES
from run.summarize_gtf1c_phase0 import (
    build_aggregate,
    validate,
)


def method(rate, violation=0.0, coverage=0.5):
    return {
        "trials": 192,
        "qualified": int(192 * rate),
        "qualification_rate": rate,
        "violations": int(192 * violation),
        "false_qualification_rate": violation,
        "conditional_violation_rate": 0.0,
        "practical_failures": 0,
        "evaluation_delta_mean": 0.02,
        "evaluation_delta_median": 0.02,
        "evaluation_changes_mean": 50.0,
        "oracle_fraction_mean": coverage,
        "oracle_fraction_median": coverage,
    }


def manifest(regime):
    positive = regime.endswith("positive")
    datasets = {}
    for dataset in ("Small-LI", "Large-LI"):
        row_violation = 0.2 if not positive else 0.0
        datasets[dataset] = {
            "trials": 192,
            "oracle_opportunities": 192,
            "oracle_opportunity_rate": 1.0,
            "methods": {
                "row_net": method(0.5, row_violation),
                "gtprc_row_harm": method(0.5, row_violation),
                "iid_paired_f1": method(0.5),
                "time_block_paired_f1": method(0.45),
                "gtf1c_graph_time": method(
                    0.45 if positive else 0.0),
            },
            "controls": {
                "shuffled": method(0.0),
                "harmful": method(0.0),
            },
        }
    return {
        "regime": regime,
        "git_commit": "abc123",
        "replicates": 64,
        "sampling_protocol": "dynamic_random",
        "validation_loader_iterations": 0,
        "test_loader_iterations": 0,
        "datasets": datasets,
    }


class SummarizeGTF1CTest(unittest.TestCase):
    def test_requires_all_regimes(self):
        with self.assertRaises(ValueError):
            validate([manifest("iid_positive")], 64)

    def test_registered_good_fixture_passes(self):
        aggregate = build_aggregate(
            [manifest(regime) for regime in REGIMES], 64)
        self.assertTrue(aggregate["gate"]["passed"])
        self.assertEqual(
            aggregate["decision"], "PROCEED_TO_FORMAL_PREREGISTRATION")


if __name__ == "__main__":
    unittest.main()
