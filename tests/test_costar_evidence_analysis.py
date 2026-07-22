import unittest

import numpy as np
import torch

from fraudGT.analysis.costar_evidence import (
    contribution_audit,
    error_subset,
    evaluate_variants,
    gradient_group_summary,
)


class CostarEvidenceAnalysisTest(unittest.TestCase):
    def split(self):
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        base = np.array([-2.0, 0.5, -0.3, 2.0, -1.0, 0.8])
        prototype = np.array([0.1, -1.0, 1.0, 0.2, 0.1, 0.1])
        costar = np.array([0.0, -0.2, 0.2, 0.0, 0.0, 0.0])
        anchor = base + prototype
        final = anchor + costar
        count = labels.size
        return {
            "edge_id": np.arange(count),
            "labels": labels,
            "base_margin": base[:, None],
            "anchor_margin": anchor[:, None],
            "final_margin": final[:, None],
            "prototype_margin_delta": prototype[:, None],
            "costar_margin_delta": costar[:, None],
            "total_evidence_margin_delta": (prototype + costar)[:, None],
            "prototype_alpha": np.full((count, 1), 0.5),
            "prototype_ready": np.ones((count, 1)),
            "prototype_reliability": np.ones((count, 1)),
            "prototype_support_margin": np.linspace(-1, 1, count)[:, None],
            "router_current": np.zeros((count, 1)),
            "router_ema": np.zeros((count, 1)),
            "router_consensus": np.zeros((count, 1)),
            "router_consistency": np.full((count, 1), 0.9),
            "residual_weight": np.full((count, 1), 0.1),
            "fallback_mask": np.array(
                [[False], [False], [False], [True], [True], [True]]),
        }

    def test_paired_variants_include_normal_shuffled_and_off(self):
        val = self.split()
        test = self.split()
        permutation = np.array([5, 4, 3, 2, 1, 0])
        metrics, val_variants, test_variants = evaluate_variants(
            val, test, permutation, permutation)
        self.assertEqual(set(metrics), {
            "z_base", "a2_anchor", "costar_full", "evidence_only_total",
            "evidence_only_prototype", "evidence_only_costar",
            "evidence_off", "evidence_shuffled",
        })
        np.testing.assert_array_equal(
            val_variants["evidence_off"], val["base_margin"].reshape(-1))
        self.assertFalse(np.array_equal(
            test_variants["evidence_shuffled"],
            test_variants["costar_full"]))
        self.assertGreaterEqual(metrics["costar_full"]["test_f1"], 0.0)

    def test_error_subset_counts_are_exhaustive(self):
        split = self.split()
        result = error_subset(
            split["labels"], split["base_margin"],
            split["total_evidence_margin_delta"], 0.5, 0.5)
        total = sum(result[key] for key in (
            "a2_wrong_evidence_right", "a2_right_evidence_wrong",
            "both_wrong", "both_right"))
        self.assertEqual(total, len(split["labels"]))
        self.assertEqual(
            result["net_corrected_minus_broken"],
            result["a2_wrong_evidence_right"] -
            result["a2_right_evidence_wrong"])

    def test_contribution_reports_open_rate_and_changed_decisions(self):
        result = contribution_audit(self.split(), full_threshold=0.5)
        self.assertFalse(result["hard_gate_present"])
        self.assertAlmostEqual(result["diagnostic_open_rate"], 0.5)
        self.assertGreater(result["applied_correction_rate"], 0.0)
        self.assertIn("costar_over_base", result["relative_contribution"])
        self.assertIn(
            "changed_rate", result["costar_vs_a2_decision_changes"])

    def test_gradient_groups_separate_new_branch(self):
        parameters = []
        for name in (
                "head.costar_router.weight", "head.eg_proto_head.weight",
                "head.layer_post_mp.weight", "encoder.weight"):
            parameter = torch.nn.Parameter(torch.ones(2))
            parameter.grad = torch.tensor([1.0, 0.0])
            parameters.append((name, parameter))
        result = gradient_group_summary(parameters)
        for group in (
                "costar_router", "prototype_branch", "z_base_decoder",
                "encoder_and_other"):
            self.assertEqual(result[group]["numel"], 2)
            self.assertGreater(result[group]["l2_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
