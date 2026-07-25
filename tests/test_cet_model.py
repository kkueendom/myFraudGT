import unittest

import torch

from fraudGT.evidence.cet_model import (
    CETFusionClassifier,
    CausalTemporalSubgraphEncoder,
)
from fraudGT.evidence.tier import EvidenceBatch


class CETModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch_size = 4
        self.max_tokens = 5
        tokens = torch.randn(self.batch_size, self.max_tokens, 13)
        tokens[..., 0] = torch.arange(self.max_tokens).float()
        tokens[..., 2:4] = torch.randint(
            0, 3, (self.batch_size, self.max_tokens, 2)).float()
        tokens[..., 5:10] = 0
        tokens[0, :2, 5] = 1
        tokens[1, :3, 7] = 1
        tokens[2, 0, 6] = 1
        tokens[2, 1, 8] = 1
        mask = torch.tensor([
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
        ])
        self.evidence = EvidenceBatch(
            tokens=tokens,
            mask=mask,
            context_edge_ids=torch.where(
                mask,
                torch.arange(self.max_tokens).view(1, -1),
                torch.full((self.batch_size, self.max_tokens), -1),
            ),
            support=torch.rand(self.batch_size, 6),
        )
        self.target = torch.tensor([
            [10.0, 0.2, 0.0, 1.0],
            [20.0, -0.4, 1.0, 2.0],
            [30.0, 0.7, 2.0, 0.0],
            [40.0, 0.1, 0.0, 1.0],
        ])

    def test_encoder_separates_endpoint_streams_and_handles_empty_history(self):
        encoder = CausalTemporalSubgraphEncoder(
            3, 3, hidden_dim=32, num_heads=4, dropout=0.0)
        representation, tokens, mask, diagnostics = encoder(
            self.evidence, self.target)
        self.assertEqual(representation.shape, (self.batch_size, 32))
        self.assertEqual(tokens.shape, (self.batch_size, self.max_tokens, 32))
        self.assertTrue(torch.equal(mask, self.evidence.mask))
        self.assertFalse(diagnostics["has_history"][-1])
        self.assertEqual(
            float(diagnostics["history_norm"][-1]), 0.0)
        self.assertTrue(torch.isfinite(representation).all())

    def test_fusion_is_representation_level_and_has_gradients(self):
        model = CETFusionClassifier(
            base_feature_dim=48,
            num_currencies=3,
            num_payment_formats=3,
            variant="fusion",
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        base = torch.randn(self.batch_size, 48)
        logits, diagnostics = model(base, self.evidence, self.target)
        self.assertEqual(logits.shape, (self.batch_size,))
        self.assertGreater(
            float(diagnostics["fusion_gain_norm"][:3].sum()), 0.0)
        logits.sum().backward()
        self.assertIsNotNone(model.cross_projection.weight.grad)
        self.assertIsNotNone(
            model.evidence_encoder.token_numeric.weight.grad)

    def test_encoder_only_does_not_depend_on_base_features(self):
        model = CETFusionClassifier(
            base_feature_dim=48,
            num_currencies=3,
            num_payment_formats=3,
            variant="encoder_only",
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        model.eval()
        first, _ = model(
            torch.randn(self.batch_size, 48), self.evidence, self.target)
        second, _ = model(
            torch.randn(self.batch_size, 48), self.evidence, self.target)
        self.assertTrue(torch.equal(first, second))

    def test_off_counterfactual_removes_history_contribution(self):
        model = CETFusionClassifier(
            base_feature_dim=48,
            num_currencies=3,
            num_payment_formats=3,
            variant="fusion",
            hidden_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        model.eval()
        _, diagnostics = model(
            torch.randn(self.batch_size, 48),
            self.evidence.off(),
            self.target,
        )
        self.assertFalse(diagnostics["has_history"].any())
        self.assertTrue(torch.equal(
            diagnostics["history_norm"],
            torch.zeros(self.batch_size),
        ))


if __name__ == "__main__":
    unittest.main()
