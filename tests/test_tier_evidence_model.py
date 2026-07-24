import unittest

import torch

from fraudGT.evidence.tier import EvidenceBatch
from fraudGT.evidence.tier_model import (
    TransactionEvidenceEncoder,
    evidence_family_channels,
    evidence_family_support_mask,
)


class TransactionEvidenceEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.batch_size = 5
        self.max_tokens = 4
        tokens = torch.randn(self.batch_size, self.max_tokens, 13)
        tokens[..., 0] = torch.arange(
            self.max_tokens).view(1, -1).float()
        tokens[..., 2:4] = torch.randint(
            0, 3, (self.batch_size, self.max_tokens, 2)).float()
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
            [False, False, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ])
        self.evidence = EvidenceBatch(
            tokens=tokens,
            mask=mask,
            context_edge_ids=torch.zeros(
                self.batch_size, self.max_tokens, dtype=torch.long),
            support=torch.rand(self.batch_size, 6),
        )
        self.target = torch.tensor([
            [1.0, 0.2, 0.0, 1.0],
            [2.0, -0.5, 1.0, 2.0],
            [3.0, 0.1, 2.0, 0.0],
            [4.0, 1.0, 0.0, 1.0],
            [5.0, -1.0, 1.0, 2.0],
        ])

    def test_all_families_produce_finite_logits_and_gradients(self):
        for family in ("all", "structure", "temporal", "flow_role"):
            model = TransactionEvidenceEncoder(
                num_currencies=3,
                num_payment_formats=3,
                family=family,
                hidden_dim=32,
                num_heads=4,
                dropout=0.0,
            )
            logits, diagnostics = model(self.evidence, self.target)
            self.assertEqual(logits.shape, (self.batch_size,))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertFalse(diagnostics["has_evidence"][2])
            logits.sum().backward()
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in model.parameters()
            ))

    def test_off_mode_is_finite_and_removes_context(self):
        model = TransactionEvidenceEncoder(
            3, 3, hidden_dim=32, num_heads=4, dropout=0.0)
        logits, diagnostics = model(self.evidence.off(), self.target)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(diagnostics["has_evidence"].any())
        self.assertTrue(torch.equal(
            diagnostics["pooled_norm"], torch.zeros(self.batch_size)))

    def test_family_definition_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            evidence_family_channels("prototype")
        with self.assertRaises(ValueError):
            evidence_family_support_mask("prototype")

    def test_family_support_masks_match_channel_semantics(self):
        self.assertEqual(
            evidence_family_support_mask("all"),
            (1, 1, 1, 1, 1, 1),
        )
        self.assertEqual(
            evidence_family_support_mask("structure"),
            (1, 1, 0, 1, 1, 1),
        )
        self.assertEqual(
            evidence_family_support_mask("temporal"),
            (1, 0, 1, 0, 0, 0),
        )
        self.assertEqual(
            evidence_family_support_mask("flow_role"),
            (1, 1, 0, 0, 0, 0),
        )

    def test_disabled_support_channels_cannot_change_family_output(self):
        for family in ("structure", "temporal", "flow_role"):
            torch.manual_seed(21)
            model = TransactionEvidenceEncoder(
                num_currencies=3,
                num_payment_formats=3,
                family=family,
                hidden_dim=32,
                num_heads=4,
                dropout=0.0,
            )
            model.eval()
            changed_support = self.evidence.support.clone()
            enabled = torch.tensor(
                evidence_family_support_mask(family),
                dtype=torch.bool,
            )
            changed_support[:, ~enabled] += 1000
            changed = EvidenceBatch(
                tokens=self.evidence.tokens,
                mask=self.evidence.mask,
                context_edge_ids=self.evidence.context_edge_ids,
                support=changed_support,
            )
            original_logits, _ = model(self.evidence, self.target)
            changed_logits, _ = model(changed, self.target)
            self.assertTrue(torch.equal(original_logits, changed_logits))


if __name__ == "__main__":
    unittest.main()
