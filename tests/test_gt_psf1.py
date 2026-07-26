import math
import unittest

import torch

from fraudGT.evidence.gt_psf1 import (
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    allocation_plan,
    comparator_intervals,
    dependency_adjacency,
    graph_diffusion_kernel,
    gt_psf1_interval,
    paired_contributions,
    paired_f1_target_summary,
    paired_f1_value_gradient,
)


class GTPSF1Test(unittest.TestCase):
    def _balanced_fixture(self):
        target_ids = torch.arange(40).repeat_interleave(2)
        labels_by_target = (torch.arange(40) % 7 == 0)
        labels = labels_by_target.repeat_interleave(2)
        pred_a = (torch.arange(80) % 5 == 0)
        pred_b = pred_a.clone()
        pred_b[3::17] = ~pred_b[3::17]
        source = torch.arange(40) * 2
        destination = source + 1
        timestamps = torch.arange(40, dtype=torch.float64) * 10.0
        adjacency = dependency_adjacency(source, destination)
        kernel = graph_diffusion_kernel(adjacency)
        return {
            "labels": labels,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "target_ids": target_ids,
            "source": source,
            "destination": destination,
            "timestamps": timestamps,
            "adjacency": adjacency,
            "kernel": kernel,
        }

    def test_plugin_f1_matches_direct_counts(self):
        fixture = self._balanced_fixture()
        summary = paired_f1_target_summary(
            fixture["labels"],
            fixture["pred_a"],
            fixture["pred_b"],
            fixture["target_ids"],
        )
        labels = fixture["labels"].reshape(40, 2)[:, 0]
        pred_a = fixture["pred_a"].reshape(40, 2).double().mean(1)
        pred_b = fixture["pred_b"].reshape(40, 2).double().mean(1)

        def soft_f1(predictions):
            tp = float((predictions * labels.double()).mean())
            fp = float((predictions * ~labels).mean())
            fn = float(((1.0 - predictions) * labels.double()).mean())
            return 2.0 * tp / (2.0 * tp + fp + fn)

        self.assertAlmostEqual(
            summary["point"], soft_f1(pred_b) - soft_f1(pred_a))

    def test_gradient_matches_finite_difference(self):
        moments = torch.tensor(
            [0.08, 0.12, 0.02, 0.09, 0.10, 0.01],
            dtype=torch.float64,
        )
        _, gradient = paired_f1_value_gradient(moments)
        epsilon = 1e-6
        numerical = []
        for index in range(6):
            shift = torch.zeros(6, dtype=torch.float64)
            shift[index] = epsilon
            upper, _ = paired_f1_value_gradient(moments + shift)
            lower, _ = paired_f1_value_gradient(moments - shift)
            numerical.append(float((upper - lower) / (2.0 * epsilon)))
        self.assertTrue(torch.allclose(
            gradient,
            torch.tensor(numerical, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_target_pooling_does_not_overweight_repeats(self):
        labels = torch.tensor([1, 1, 1, 0])
        pred_a = torch.tensor([0, 0, 0, 0])
        pred_b = torch.tensor([1, 1, 1, 1])
        target_ids = torch.tensor([10, 10, 10, 20])
        summary = paired_f1_target_summary(
            labels, pred_a, pred_b, target_ids)
        expected = paired_contributions(
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
        ).mean(dim=0)
        self.assertTrue(torch.allclose(summary["moments"], expected))
        self.assertEqual(summary["target_ids"].tolist(), [10, 20])

    def test_deterministic_neighborhood_has_zero_within_variance(self):
        labels = torch.tensor([1, 1, 0, 0])
        pred_a = torch.tensor([0, 0, 1, 1])
        pred_b = torch.tensor([1, 1, 0, 0])
        summary = paired_f1_target_summary(
            labels, pred_a, pred_b, torch.tensor([0, 0, 1, 1]))
        self.assertTrue(torch.equal(
            summary["within_variance"], torch.zeros(2)))

    def test_allocation_boundaries_and_monotonicity(self):
        grid = (2, 4, 8, 16)
        self.assertEqual(
            allocation_plan(1.0, 0.0, replicate_grid=grid)[
                "selected_replicates"],
            2,
        )
        self.assertEqual(
            allocation_plan(0.0, 1.0, replicate_grid=grid)[
                "selected_replicates"],
            16,
        )
        low = allocation_plan(
            1.0, 1.0, replicate_grid=grid)["selected_replicates"]
        high = allocation_plan(
            1.0, 100.0, replicate_grid=grid)["selected_replicates"]
        self.assertGreaterEqual(high, low)

    def test_independent_fixture_is_in_scope(self):
        fixture = self._balanced_fixture()
        result = gt_psf1_interval(
            fixture["labels"],
            fixture["pred_a"],
            fixture["pred_b"],
            fixture["target_ids"],
            fixture["source"],
            fixture["destination"],
            fixture["timestamps"],
            fixture["adjacency"],
            fixture["kernel"],
        )
        self.assertEqual(result["status"], STATUS_OK)
        self.assertGreaterEqual(result["total_variance"], 0.0)
        self.assertIsNotNone(result["allocation"])
        comparators = comparator_intervals(
            fixture["labels"],
            fixture["pred_a"],
            fixture["pred_b"],
            fixture["target_ids"],
            fixture["source"],
            fixture["destination"],
            fixture["kernel"],
        )
        self.assertAlmostEqual(
            comparators["target_iid"]["standard_error"],
            comparators["graph_hac"]["standard_error"],
        )

    def test_dense_hub_is_out_of_scope(self):
        target_count = 40
        repeats = 2
        target_ids = torch.arange(target_count).repeat_interleave(repeats)
        labels = (
            torch.arange(target_count).repeat_interleave(repeats) % 9 == 0)
        pred_a = (
            torch.arange(target_count).repeat_interleave(repeats) % 7 == 0)
        pred_b = pred_a.clone()
        pred_b[::13] = ~pred_b[::13]
        source = torch.zeros(target_count, dtype=torch.long)
        destination = torch.arange(target_count) + 1
        timestamps = torch.arange(target_count, dtype=torch.float64)
        adjacency = dependency_adjacency(source, destination)
        kernel = graph_diffusion_kernel(adjacency)
        result = gt_psf1_interval(
            labels,
            pred_a,
            pred_b,
            target_ids,
            source,
            destination,
            timestamps,
            adjacency,
            kernel,
        )
        self.assertEqual(result["status"], STATUS_OUT_OF_SCOPE)
        self.assertIn("max_degree", result["diagnostic_reasons"])

    def test_classifier_swap_changes_sign(self):
        fixture = self._balanced_fixture()
        forward = paired_f1_target_summary(
            fixture["labels"],
            fixture["pred_a"],
            fixture["pred_b"],
            fixture["target_ids"],
        )
        reverse = paired_f1_target_summary(
            fixture["labels"],
            fixture["pred_b"],
            fixture["pred_a"],
            fixture["target_ids"],
        )
        self.assertAlmostEqual(forward["point"], -reverse["point"])

    def test_single_repeat_blocks_allocation(self):
        fixture = self._balanced_fixture()
        selected = torch.arange(0, 80, 2)
        result = gt_psf1_interval(
            fixture["labels"][selected],
            fixture["pred_a"][selected],
            fixture["pred_b"][selected],
            fixture["target_ids"][selected],
            fixture["source"],
            fixture["destination"],
            fixture["timestamps"],
            fixture["adjacency"],
            fixture["kernel"],
        )
        self.assertEqual(result["status"], STATUS_OUT_OF_SCOPE)
        self.assertIn("sampler_replicates", result["diagnostic_reasons"])
        self.assertTrue(math.isnan(result["within_per_draw"]))


if __name__ == "__main__":
    unittest.main()
