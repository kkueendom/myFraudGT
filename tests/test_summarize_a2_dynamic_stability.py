import unittest

from run.summarize_a2_dynamic_stability import (
    bootstrap_mean_interval,
    distribution,
    quantile,
)


class SummarizeA2DynamicStabilityTest(unittest.TestCase):
    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(quantile([0, 10], 0.5), 5.0)
        self.assertEqual(quantile([3], 0.025), 3.0)

    def test_distribution_records_sample_standard_deviation(self):
        row = distribution([1, 2, 3])
        self.assertEqual(row["count"], 3)
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["std"], 1.0)
        self.assertEqual(row["median"], 2.0)

    def test_bootstrap_interval_is_deterministic(self):
        first = bootstrap_mean_interval(
            [1, 2, 3, 4], seed=42, draws=1000)
        second = bootstrap_mean_interval(
            [1, 2, 3, 4], seed=42, draws=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.5)
        self.assertGreaterEqual(first[1], 2.5)


if __name__ == "__main__":
    unittest.main()
