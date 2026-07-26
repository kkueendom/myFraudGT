import itertools
import math
import unittest

import torch

from fraudGT.evidence.gtf1c import f1_from_counts
from fraudGT.evidence.trefic import (
    additive_utility_lcb,
    base_f1_sensitivity_ratio,
    directional_intervention_counts,
    directional_utility_rows,
    exact_f1_sign_utility,
    ratio_envelope_certification,
    wilson_upper,
)


def sign(value, tolerance=1e-12):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


class TREFICTest(unittest.TestCase):
    def test_exact_identity_exhaustive_small_confusions(self):
        for tp, fp, fn, tn in itertools.product(range(4), repeat=4):
            total = tp + fp + fn
            if total == 0:
                continue
            base_f1 = f1_from_counts(tp, fp, fn)
            rho = tp / total
            for ac in range(fn + 1):
                for ab in range(tn + 1):
                    for rc in range(fp + 1):
                        for rb in range(tp + 1):
                            routed_f1 = f1_from_counts(
                                tp + ac - rb,
                                fp + ab - rc,
                                fn - ac + rb,
                            )
                            utility = exact_f1_sign_utility(
                                ac, ab, rc, rb, rho)
                            self.assertEqual(
                                sign(routed_f1 - base_f1),
                                sign(utility),
                                (tp, fp, fn, tn, ac, ab, rc, rb),
                            )

    def test_directional_rows_sum_to_count_utility(self):
        labels = torch.tensor([1, 0, 1, 0, 1, 0]).bool()
        base = torch.tensor([0, 0, 1, 1, 1, 1]).bool()
        routed = torch.tensor([1, 1, 1, 0, 0, 1]).bool()
        rho = 0.2
        counts = directional_intervention_counts(labels, base, routed)
        rows = directional_utility_rows(labels, base, routed, rho)
        expected = exact_f1_sign_utility(
            counts["add_corrected"],
            counts["add_broken"],
            counts["remove_corrected"],
            counts["remove_broken"],
            rho,
        )
        self.assertAlmostEqual(float(rows.sum()), expected)

    def test_remove_break_fails_zero_lower_endpoint(self):
        labels = torch.tensor(
            [1] * 2 + [0] * 40 + [1] * 8 + [0] * 50).bool()
        base = torch.tensor(
            [1] * 2 + [1] * 40 + [0] * 8 + [0] * 50).bool()
        routed = base.clone()
        routed[0] = False
        routed[2:42] = False
        view = {
            "labels": labels,
            "base": base,
            "time_groups": torch.arange(100) // 5,
            "graph_groups": torch.arange(100) // 4,
        }
        result = ratio_envelope_certification(
            view,
            routed,
            rho_upper=0.1,
            min_changes=20,
            delta=0.05 / 6,
            practical_delta=-1.0,
        )
        self.assertLess(
            result["endpoint_results"]["lower"]["sum"], 0.0)
        self.assertFalse(result["qualified"])

    def test_additive_grouping_changes_standard_error(self):
        utility = torch.tensor([1.0, 1.0, -1.0, -1.0] * 8)
        iid = additive_utility_lcb(utility, delta=0.05)
        grouped = additive_utility_lcb(
            utility, torch.arange(32) // 4, delta=0.05)
        self.assertNotEqual(
            iid["standard_error"], grouped["standard_error"])

    def test_ratio_and_wilson_upper_are_ordered(self):
        labels = torch.tensor([1, 0, 1, 0, 1]).bool()
        base = torch.tensor([1, 1, 0, 0, 0]).bool()
        ratio = base_f1_sensitivity_ratio(labels, base)
        self.assertAlmostEqual(ratio, 1 / 4)
        self.assertGreater(wilson_upper(1, 4), ratio)
        self.assertEqual(wilson_upper(0, 0), 1.0)
        with self.assertRaises(ValueError):
            wilson_upper(5, 4)

    def test_single_group_cannot_certify(self):
        result = additive_utility_lcb(
            torch.ones(5), torch.zeros(5, dtype=torch.long))
        self.assertTrue(math.isinf(result["standard_error"]))
        self.assertEqual(result["lower_bound"], -math.inf)


if __name__ == "__main__":
    unittest.main()
