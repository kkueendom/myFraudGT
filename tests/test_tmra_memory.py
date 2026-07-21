import unittest

import torch

from fraudGT.train.custom_train import (
    _tmra_append_memory,
    _tmra_memory_rank_loss,
)


class _DummyTMRA:
    def __init__(self):
        self.tmra_num_environments = 3
        self.tmra_memory_size = 4
        self.epra_pair_limit = 8
        self.epra_rank_temperature = 0.25
        self.epra_rank_margin = 0.50
        self._tmra_pos_memory = [None, None, None]
        self._tmra_neg_memory = [None, None, None]


class TMRAMemoryTest(unittest.TestCase):
    def test_fifo_memory_keeps_only_recent_detached_values(self):
        previous = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([4.0, 5.0], requires_grad=True)
        output = _tmra_append_memory(previous, values, capacity=4)

        self.assertTrue(torch.equal(
            output, torch.tensor([2.0, 3.0, 4.0, 5.0])))
        self.assertFalse(output.requires_grad)

    def test_positive_memory_trains_a_negative_only_batch(self):
        module = _DummyTMRA()
        module._tmra_pos_memory = [
            torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])]
        logits = torch.nn.Parameter(torch.zeros(4, 2))
        labels = torch.zeros(4, dtype=torch.long)
        env_ids = torch.tensor([0, 0, 1, 2])

        loss, active = _tmra_memory_rank_loss(
            module, logits, labels, env_ids)
        loss.backward()

        margin_grad = logits.grad[:, 1] - logits.grad[:, 0]
        self.assertEqual(active, 3)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue((margin_grad > 0.0).all())

    def test_negative_memory_trains_a_positive_only_batch(self):
        module = _DummyTMRA()
        module._tmra_neg_memory = [
            torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0])]
        logits = torch.nn.Parameter(torch.zeros(3, 2))
        labels = torch.ones(3, dtype=torch.long)
        env_ids = torch.tensor([0, 1, 2])

        loss, active = _tmra_memory_rank_loss(
            module, logits, labels, env_ids)
        loss.backward()

        margin_grad = logits.grad[:, 1] - logits.grad[:, 0]
        self.assertEqual(active, 3)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue((margin_grad < 0.0).all())

    def test_first_single_class_batch_populates_memory_without_gradient(self):
        module = _DummyTMRA()
        logits = torch.nn.Parameter(torch.zeros(3, 2))
        labels = torch.zeros(3, dtype=torch.long)
        env_ids = torch.tensor([0, 1, 2])

        loss, active = _tmra_memory_rank_loss(
            module, logits, labels, env_ids)
        loss.backward()

        self.assertEqual(active, 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertTrue(all(
            item is not None for item in module._tmra_neg_memory))


if __name__ == '__main__':
    unittest.main()
