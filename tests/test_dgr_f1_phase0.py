import unittest

import torch

from run.dgr_f1_phase0 import (
    SCENARIO_TARGETS,
    population_target,
    simulate_streams,
)


class DGRF1Phase0Test(unittest.TestCase):
    def test_population_target_three_states(self):
        self.assertEqual(
            population_target(
                [0.02] * 100, 0.005, 0.75)["decision"],
            "REPLICABLE_IMPROVEMENT",
        )
        self.assertEqual(
            population_target(
                [-0.02] * 100, 0.005, 0.75)["decision"],
            "REPLICABLE_HARM",
        )
        self.assertEqual(
            population_target(
                [0.02] * 60 + [-0.01] * 40, 0.005, 0.75)[
                    "decision"
                ],
            "INSUFFICIENT_INFORMATION",
        )

    def test_all_scenarios_are_registered(self):
        self.assertEqual(len(SCENARIO_TARGETS), 7)

    def test_simulated_counts_are_feasible_and_identity_holds(self):
        template = {"tp": 5, "fp": 20, "fn": 30, "tn": 1000}
        scenario = {
            "mode": "standard",
            "parameters": {
                "count_log_sd": 0.2,
                "dependence_strength": 0.3,
                "add_corrected_rate": 0.4,
                "add_broken_rate": 0.001,
                "remove_corrected_rate": 0.2,
                "remove_broken_rate": 0.1,
            },
        }
        generator = torch.Generator().manual_seed(17)
        rows = simulate_streams(
            template, scenario, (1000,), generator, torch.device("cpu"))
        self.assertEqual(int(rows["identity_failure"].sum()), 0)
        self.assertTrue(torch.all(
            rows["add_corrected"] <= rows["fn"]))
        self.assertTrue(torch.all(
            rows["remove_broken"] <= rows["tp"]))


if __name__ == "__main__":
    unittest.main()
