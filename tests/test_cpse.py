import unittest

import torch

from fraudGT.evidence.cpse import CausalPredictiveSurpriseEncoder
from fraudGT.evidence.tier import EvidenceBatch


def evidence_batch():
    tokens = torch.zeros((2, 3, 13))
    tokens[:, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    tokens[:, :, 1] = 0.5
    tokens[:, :, 2] = 1
    tokens[:, :, 3] = 1
    tokens[:, :, 4] = 0.2
    tokens[:, :, 5] = 1
    return EvidenceBatch(
        tokens=tokens,
        mask=torch.ones((2, 3), dtype=torch.bool),
        context_edge_ids=torch.arange(6).view(2, 3),
        support=torch.ones((2, 6)),
    )


class CPSETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = CausalPredictiveSurpriseEncoder(3, 3, hidden_dim=16)
        self.model.eval()

    def test_prediction_does_not_receive_target_attributes(self):
        evidence = evidence_batch()
        first = self.model(evidence)["context"]
        second = self.model(evidence)["context"]
        self.assertTrue(torch.allclose(first, second))

    def test_surprise_changes_with_target_but_context_does_not(self):
        evidence = evidence_batch()
        prediction = self.model(evidence)
        target_a = torch.tensor([[4.0, 0.0, 1.0, 1.0]] * 2)
        target_b = torch.tensor([[5.0, 3.0, 2.0, 2.0]] * 2)
        a = self.model.surprise_features(
            prediction, evidence, target_a)
        b = self.model.surprise_features(
            prediction, evidence, target_b)
        self.assertEqual(a.shape, (2, self.model.FEATURE_DIM))
        self.assertFalse(torch.allclose(a, b))

    def test_off_evidence_is_finite(self):
        evidence = evidence_batch().off()
        prediction = self.model(evidence)
        target = torch.tensor([[4.0, 0.0, 1.0, 1.0]] * 2)
        features = self.model.surprise_features(
            prediction, evidence, target)
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()

