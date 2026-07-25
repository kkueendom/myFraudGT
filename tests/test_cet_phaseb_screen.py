import unittest

import torch

from run.cet_phaseb_screen import (
    OOFTeacher,
    alignment_positions,
    counterfactual_ranking_loss,
    task_label,
    task_spec,
)


class CETPhaseBScreenTest(unittest.TestCase):
    def test_alignment_positions_restore_requested_order(self):
        source = torch.tensor([30, 10, 40, 20])
        requested = torch.tensor([20, 40, 10])
        positions = alignment_positions(requested, source)
        self.assertTrue(torch.equal(
            source[positions], requested))

    def test_alignment_rejects_missing_and_duplicate_edges(self):
        with self.assertRaises(AssertionError):
            alignment_positions(
                torch.tensor([10, 50]),
                torch.tensor([10, 20]),
            )
        with self.assertRaises(AssertionError):
            alignment_positions(
                torch.tensor([10]),
                torch.tensor([10, 10]),
            )

    def test_oof_lookup_marks_only_available_base_errors(self):
        teacher = OOFTeacher(
            edge_ids=torch.tensor([10, 20, 40]),
            labels=torch.tensor([0, 1, 1]),
            scores=torch.tensor([0.1, 0.2, 0.9]),
            thresholds=torch.tensor([0.5, 0.5, 0.5]),
        )
        result = teacher.lookup(
            torch.tensor([40, 30, 20]),
            torch.tensor([1, 0, 1]),
        )
        self.assertTrue(torch.equal(
            result["available"],
            torch.tensor([True, False, True]),
        ))
        self.assertTrue(torch.equal(
            result["errors"],
            torch.tensor([False, False, True]),
        ))

    def test_oof_lookup_rejects_label_mismatch(self):
        teacher = OOFTeacher(
            edge_ids=torch.tensor([10]),
            labels=torch.tensor([1]),
            scores=torch.tensor([0.9]),
            thresholds=torch.tensor([0.5]),
        )
        with self.assertRaises(AssertionError):
            teacher.lookup(torch.tensor([10]), torch.tensor([0]))

    def test_counterfactual_loss_rewards_better_normal_evidence(self):
        labels = torch.tensor([1.0, 0.0])
        good_normal = torch.tensor([4.0, -4.0])
        bad_normal = torch.tensor([-4.0, 4.0])
        counterfactual = torch.zeros(2)
        good_loss = counterfactual_ranking_loss(
            good_normal,
            counterfactual,
            counterfactual,
            labels,
            margin=0.05,
        )
        bad_loss = counterfactual_ranking_loss(
            bad_normal,
            counterfactual,
            counterfactual,
            labels,
            margin=0.05,
        )
        self.assertLess(float(good_loss), float(bad_loss))

    def test_task_spec_applies_registered_objective_override(self):
        spec = {
            "lambda_counterfactual": 0.2,
            "lambda_complementarity": 0.5,
        }
        resolved = task_spec(spec, {
            "spec_overrides": {"lambda_complementarity": 0.0},
        })
        self.assertEqual(resolved["lambda_counterfactual"], 0.2)
        self.assertEqual(resolved["lambda_complementarity"], 0.0)
        self.assertEqual(spec["lambda_complementarity"], 0.5)

    def test_task_spec_rejects_unregistered_override(self):
        with self.assertRaises(ValueError):
            task_spec({}, {"spec_overrides": {"max_epochs": 1}})

    def test_task_label_defaults_to_variant_and_rejects_paths(self):
        self.assertEqual(task_label({"variant": "fusion"}), "fusion")
        self.assertEqual(task_label({
            "variant": "fusion",
            "experiment_label": "no_oof",
        }), "no_oof")
        with self.assertRaises(ValueError):
            task_label({
                "variant": "fusion",
                "experiment_label": "../no_oof",
            })


if __name__ == "__main__":
    unittest.main()
