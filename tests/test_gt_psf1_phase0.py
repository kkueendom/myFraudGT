import unittest

import torch

from fraudGT.evidence.gt_psf1 import (
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
)
from run.gt_psf1_phase0 import (
    _scope_status,
    build_topology,
    simulate_experiments,
)


PARAMETERS = {
    "base_separation": 0.85,
    "alternative_separation_change": 0.22,
    "target_noise_sd": 1.0,
    "shared_classifier_noise_fraction": 0.4,
    "label_logit_graph_shock": 0.3,
    "target_cost": 8.0,
    "neighbor_cost": 1.0,
    "scenario_dependence_and_sampler_sd": {
        "iid_deterministic": [0.0, 0.0],
        "endpoint_dyadic": [0.35, 0.25],
        "two_hop_spillover": [0.45, 0.25],
        "temporal_ar1": [0.45, 0.25],
        "low_sampler_variance": [0.35, 0.15],
        "high_sampler_variance": [0.35, 0.9],
        "dense_hub_out_of_scope": [0.6, 0.3],
    },
}


class GTPSF1Phase0Test(unittest.TestCase):
    def test_registered_topologies_have_expected_scope(self):
        template = {
            "targets": 96,
            "entities": 64,
            "fraud_prevalence": 0.1,
        }
        scenarios = (
            "iid_deterministic",
            "endpoint_dyadic",
            "two_hop_spillover",
            "temporal_ar1",
            "low_sampler_variance",
            "high_sampler_variance",
        )
        for scenario in scenarios:
            topology = build_topology(
                template, scenario, torch.device("cpu"))
            self.assertEqual(_scope_status(topology)[0], STATUS_OK)
        dense = build_topology(
            template,
            "dense_hub_out_of_scope",
            torch.device("cpu"),
        )
        self.assertEqual(
            _scope_status(dense)[0], STATUS_OUT_OF_SCOPE)

    def test_simulation_shapes_and_nonnegative_variance(self):
        template = {
            "targets": 48,
            "entities": 32,
            "fraud_prevalence": 0.1,
        }
        topology = build_topology(
            template, "endpoint_dyadic", torch.device("cpu"))
        generator = torch.Generator()
        generator.manual_seed(123)
        result = simulate_experiments(
            template,
            "endpoint_dyadic",
            1.0,
            experiments=8,
            repeats=4,
            parameters=PARAMETERS,
            topology=topology,
            generator=generator,
        )
        self.assertEqual(result["point"].shape, (8,))
        self.assertEqual(result["within_per_draw"].shape, (8,))
        for variance in result["variances"].values():
            self.assertTrue(bool((variance >= 0.0).all()))

    def test_null_is_symmetric_before_formal_run(self):
        template = {
            "targets": 96,
            "entities": 64,
            "fraud_prevalence": 0.1,
        }
        topology = build_topology(
            template, "two_hop_spillover", torch.device("cpu"))
        generator = torch.Generator()
        generator.manual_seed(321)
        result = simulate_experiments(
            template,
            "two_hop_spillover",
            0.0,
            experiments=256,
            repeats=4,
            parameters=PARAMETERS,
            topology=topology,
            generator=generator,
        )
        self.assertLess(abs(float(result["point"].mean())), 0.02)


if __name__ == "__main__":
    unittest.main()
