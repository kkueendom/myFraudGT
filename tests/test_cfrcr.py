import unittest

import torch

from fraudGT.head.hetero_edge import (
    _add_margin_correction,
    _binary_margin,
    _risk_controlled_committee,
)
from fraudGT.train.custom_train import (
    _cfrcr_aux_loss,
    _cfrcr_pair_regret,
)


class _DummyCFRCR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.corrections = torch.nn.Parameter(torch.zeros(6, 3))
        self._cfrcr_base_margin = torch.tensor([
            0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
        self._cfrcr_corrections = self.corrections
        self._cfrcr_labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self._cfrcr_env_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        self._cfrcr_cur_epoch = 0
        self._cfrcr_log_step = 0
        self.cfrcr_num_experts = 3
        self.cfrcr_warmup_epochs = 0
        self.cfrcr_ramp_epochs = 1
        self.cfrcr_pair_limit = 8
        self.cfrcr_rank_temperature = 0.25
        self.cfrcr_rank_margin = 0.50
        self.cfrcr_gain_target = 0.10
        self.cfrcr_worst_group_weight = 0.50
        self.cfrcr_l2_weight = 0.0
        self.cfrcr_aux_weight = 1.0
        self.cfrcr_log_interval = 1000


class CFRCRTest(unittest.TestCase):
    def test_margin_correction_changes_only_binary_margin(self):
        logits = torch.tensor([[1.0, 3.0], [-2.0, 4.0]])
        correction = torch.tensor([0.4, -0.2])
        output = _add_margin_correction(logits, correction)

        self.assertTrue(torch.allclose(
            output.mean(dim=-1), logits.mean(dim=-1)))
        self.assertTrue(torch.allclose(
            _binary_margin(output), _binary_margin(logits) + correction))

    def test_risk_controlled_committee_falls_back_on_disagreement(self):
        corrections = torch.tensor([
            [0.30, 0.30, 0.30],
            [0.30, -0.30, 0.00],
            [0.02, 0.02, 0.02],
        ])
        committed, mean, disagreement = _risk_controlled_committee(
            corrections, kappa=0.5, deadzone=0.01)

        self.assertTrue(torch.allclose(
            mean, torch.tensor([0.30, 0.00, 0.02])))
        self.assertTrue(torch.allclose(
            committed, torch.tensor([0.29, 0.00, 0.01])))
        self.assertGreater(disagreement[1], disagreement[0])

    def test_pair_regret_has_a_finite_gain_stop(self):
        base = torch.tensor([0.0, 0.1])
        labels = torch.tensor([0, 1])
        improved = torch.tensor([-1.0, 1.1], requires_grad=True)
        worse = torch.tensor([0.2, -0.1], requires_grad=True)

        improved_regret, improved_valid = _cfrcr_pair_regret(
            base, improved, labels, 8, 0.25, 0.50, 0.01)
        worse_regret, worse_valid = _cfrcr_pair_regret(
            base, worse, labels, 8, 0.25, 0.50, 0.01)

        self.assertTrue(improved_valid and worse_valid)
        self.assertEqual(improved_regret.item(), 0.0)
        self.assertGreater(worse_regret, 0.0)

    def test_each_expert_excludes_held_out_environment_labels(self):
        module = _DummyCFRCR()
        loss = _cfrcr_aux_loss(module)
        loss.backward()

        gradients = module.corrections.grad
        for expert_idx in range(3):
            held_out = (module._cfrcr_env_ids == expert_idx)
            included = ~held_out
            self.assertEqual(
                torch.count_nonzero(gradients[held_out, expert_idx]), 0)
            self.assertGreater(
                torch.count_nonzero(gradients[included, expert_idx]), 0)


if __name__ == '__main__':
    unittest.main()
