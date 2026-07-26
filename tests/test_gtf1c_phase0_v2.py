import copy
import math
import unittest

import torch

from run.gtf1c_phase0_v2 import (
    V2_REGIMES,
    choose_candidate,
    directional_candidates,
    generate_v2_scores,
)


def toy_view():
    labels = torch.tensor(
        [1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
    base = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bool)
    return {
        "labels": labels,
        "base": base,
        "source": torch.arange(8),
        "destination": torch.arange(8) + 8,
        "timestamps": torch.arange(8),
        "edge_ids": torch.arange(8),
        "time_groups": torch.arange(8) // 2,
        "graph_groups": torch.arange(8) // 2,
        "degree_signal": torch.zeros(8),
    }


class GTF1CPhase0V2Test(unittest.TestCase):
    def test_directional_selection_emits_three_locked_candidates(self):
        view = toy_view()
        generator = torch.Generator().manual_seed(12)
        scores = generate_v2_scores(
            view, V2_REGIMES["iid_positive"], "selection", generator)
        candidates = directional_candidates(
            view,
            scores,
            torch.linspace(0.5, 0.99, 16),
            V2_REGIMES["iid_positive"],
            min_changes=1,
        )
        self.assertEqual(
            [candidate["name"] for candidate in candidates],
            ["add", "remove", "joint"],
        )

    def test_negative_offset_preserves_score_order(self):
        view = toy_view()
        config = V2_REGIMES["cpse_remove_fragility"]
        base_config = copy.deepcopy(config)
        base_config["offsets"]["certification"] = 0.0
        g1 = torch.Generator().manual_seed(13)
        g2 = torch.Generator().manual_seed(13)
        base = generate_v2_scores(
            view, base_config, "certification", g1)
        shifted = generate_v2_scores(
            view, config, "certification", g2)
        difference = (
            shifted["remove"]["normal"]
            - base["remove"]["normal"]
        )
        self.assertTrue(torch.allclose(
            difference, torch.full_like(difference, 2.0)))

    def test_candidate_choice_uses_conservative_graph_time_bound(self):
        certifications = []
        for time_lcb, graph_lcb, changed in (
            (0.03, 0.01, 20),
            (0.02, 0.02, 30),
            (-0.01, 0.05, 40),
        ):
            certifications.append({
                "qualifications": {"gtf1c_graph_time": (
                    time_lcb > 0 and graph_lcb > 0)},
                "time_f1": {"lower_bound": time_lcb},
                "graph_f1": {"lower_bound": graph_lcb},
                "statistics": {"changed": changed},
            })
        self.assertEqual(
            choose_candidate("gtf1c_graph_time", certifications), 1)

    def test_disabled_direction_stays_infinite(self):
        view = toy_view()
        config = V2_REGIMES["cpse_remove_fragility"]
        generator = torch.Generator().manual_seed(14)
        scores = generate_v2_scores(
            view, config, "selection", generator)
        candidates = directional_candidates(
            view,
            scores,
            torch.linspace(0.5, 0.99, 16),
            config,
            min_changes=1,
        )
        self.assertTrue(math.isinf(candidates[0]["add_threshold"]))


if __name__ == "__main__":
    unittest.main()
