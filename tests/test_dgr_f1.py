import unittest

import numpy as np

from fraudGT.evidence.dgr_f1 import (
    DECISION_HARM,
    DECISION_IMPROVEMENT,
    DECISION_INSUFFICIENT,
    clopper_pearson_interval,
    dgr_f1_trajectory,
    f1_intervention_delta,
    iid_paired_f1_interval,
    mean_only_trajectory,
    one_stream_decision,
    replication_only_trajectory,
)


class DGRF1Test(unittest.TestCase):
    def test_f1_mechanism_sign_matches_delta(self):
        result = f1_intervention_delta(
            tp=5,
            fp=163,
            fn=67,
            add_corrected=0,
            add_broken=0,
            remove_corrected=59,
            remove_broken=3,
        )
        self.assertLess(result["delta"], 0.0)
        self.assertLess(result["utility"], 0.0)

    def test_joint_decision_requires_mean_and_replication(self):
        stable = np.full(64, 0.03)
        result = dgr_f1_trajectory(stable)
        self.assertEqual(result["decision"], DECISION_IMPROVEMENT)
        mixed = np.asarray([0.03] * 40 + [-0.01] * 24)
        self.assertEqual(
            mean_only_trajectory(mixed)["decision"],
            DECISION_IMPROVEMENT,
        )
        self.assertEqual(
            dgr_f1_trajectory(mixed)["decision"],
            DECISION_INSUFFICIENT,
        )

    def test_harm_and_one_stream_decisions(self):
        harmful = np.full(64, -0.03)
        self.assertEqual(
            dgr_f1_trajectory(harmful)["decision"],
            DECISION_HARM,
        )
        self.assertEqual(
            replication_only_trajectory(harmful)["decision"],
            DECISION_HARM,
        )
        self.assertEqual(
            one_stream_decision(0.01), DECISION_IMPROVEMENT)
        self.assertEqual(
            one_stream_decision(-0.01), DECISION_HARM)
        self.assertEqual(
            one_stream_decision(0.001), DECISION_INSUFFICIENT)

    def test_clopper_pearson_orders_rate(self):
        interval = clopper_pearson_interval(55, 64, 0.05)
        self.assertLess(interval["lower"], interval["rate"])
        self.assertGreater(interval["upper"], interval["rate"])
        self.assertEqual(
            clopper_pearson_interval(0, 64, 0.05)["lower"], 0.0)
        self.assertEqual(
            clopper_pearson_interval(64, 64, 0.05)["upper"], 1.0)

    def test_row_iid_interval_matches_point_delta(self):
        result = iid_paired_f1_interval(
            tp=10,
            fp=30,
            fn=20,
            tn=940,
            add_corrected=10,
            add_broken=2,
            remove_corrected=5,
            remove_broken=1,
        )
        exact = f1_intervention_delta(
            tp=10,
            fp=30,
            fn=20,
            add_corrected=10,
            add_broken=2,
            remove_corrected=5,
            remove_broken=1,
        )
        self.assertAlmostEqual(result["point"], exact["delta"])
        self.assertLess(result["lower"], result["upper"])

    def test_checkpoint_validation(self):
        with self.assertRaises(ValueError):
            dgr_f1_trajectory(np.ones(64), checkpoints=(16, 8))
        with self.assertRaises(ValueError):
            dgr_f1_trajectory(np.ones(8), checkpoints=(8, 16))


if __name__ == "__main__":
    unittest.main()
