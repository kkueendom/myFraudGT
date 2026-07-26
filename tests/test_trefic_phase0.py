import unittest

import torch

from run.gtf1c_phase0_v2 import V2_REGIMES, directional_candidates
from run.trefic_phase0 import (
    certify_trefic_candidates,
    choose_trefic_candidate,
)


def toy_view():
    labels = torch.tensor(
        [1] * 12 + [0] * 88, dtype=torch.bool)
    base = torch.tensor(
        [1] * 2 + [0] * 10 + [1] * 38 + [0] * 50,
        dtype=torch.bool,
    )
    return {
        "labels": labels,
        "base": base,
        "source": torch.arange(100),
        "destination": torch.arange(100) + 100,
        "timestamps": torch.arange(100),
        "edge_ids": torch.arange(100),
        "time_groups": torch.arange(100) // 5,
        "graph_groups": torch.arange(100) // 4,
        "degree_signal": torch.zeros(100),
    }


class TREFICPhase0Test(unittest.TestCase):
    def test_candidate_choice_uses_worst_endpoint_bound(self):
        rows = [
            {
                "qualified": True,
                "worst_lower_bound": 0.01,
                "statistics": {"changed": 30},
            },
            {
                "qualified": True,
                "worst_lower_bound": 0.02,
                "statistics": {"changed": 20},
            },
            {
                "qualified": False,
                "worst_lower_bound": 0.05,
                "statistics": {"changed": 50},
            },
        ]
        self.assertEqual(choose_trefic_candidate(rows), 1)

    def test_candidate_multiplicity_is_delta_over_six(self):
        view = toy_view()
        scores = {
            "add": {
                condition: torch.where(
                    view["labels"], torch.tensor(5.0), torch.tensor(-5.0))
                for condition in ("normal", "shuffled", "harmful")
            },
            "remove": {
                condition: torch.where(
                    ~view["labels"], torch.tensor(5.0), torch.tensor(-5.0))
                for condition in ("normal", "shuffled", "harmful")
            },
        }
        candidates = directional_candidates(
            view,
            scores,
            torch.linspace(0.5, 0.99, 16),
            V2_REGIMES["iid_positive"],
            min_changes=1,
        )
        results = certify_trefic_candidates(
            view,
            scores,
            "normal",
            candidates,
            rho_upper=0.1,
            min_changes=1,
            delta=0.06,
            practical_delta=-1.0,
        )
        self.assertEqual(len(results), 3)
        for result in results:
            for endpoint in result["endpoint_results"].values():
                self.assertEqual(
                    endpoint["time"]["group_count"], 20)


if __name__ == "__main__":
    unittest.main()
