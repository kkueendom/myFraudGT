import unittest
from pathlib import Path

import torch

from run.cdvt_phase1_screen import bernoulli_js, common_positions


class CDVTPhase1ScreenTest(unittest.TestCase):
    def test_common_positions_align_independently_ordered_views(self):
        first = torch.tensor([7, 2, 9, 4])
        second = torch.tensor([4, 7, 3, 2])
        first_positions, second_positions = common_positions(first, second)
        self.assertTrue(torch.equal(
            first[first_positions], second[second_positions]))
        self.assertEqual(set(first[first_positions].tolist()), {2, 4, 7})

    def test_js_is_symmetric_zero_for_equal_and_positive_otherwise(self):
        first = torch.tensor([-2.0, 0.0, 2.0])
        same = bernoulli_js(first, first)
        second = torch.tensor([2.0, 0.0, -2.0])
        forward = bernoulli_js(first, second)
        reverse = bernoulli_js(second, first)
        self.assertAlmostEqual(float(same), 0.0, places=7)
        self.assertGreater(float(forward), 0.0)
        self.assertAlmostEqual(float(forward), float(reverse), places=7)

    def test_runner_does_not_create_or_restore_rng_state(self):
        source = Path("run/cdvt_phase1_screen.py").read_text()
        forbidden = (
            "torch." + "Generator(",
            "get_" + "rng_state(",
            "set_" + "rng_state(",
            "fixed_target_panel" + "=True",
        )
        self.assertFalse(any(token in source for token in forbidden))
        self.assertIn("shuffle=True", source)
        self.assertIn('"sampling_protocol": "dynamic_random"', source)


if __name__ == "__main__":
    unittest.main()
