import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from fraudGT.sampler.custom_sampler import (
    LoaderWrapper,
    _fixed_target_panel,
)


class FixedEvalPanelTest(unittest.TestCase):
    def test_panel_is_fixed_sized_and_label_independent(self):
        mask = torch.zeros(1000, dtype=torch.bool)
        mask[100:900] = True
        first, first_seed = _fixed_target_panel(
            mask, 120, 'AML-Small-LI:val', 1729)
        second, second_seed = _fixed_target_panel(
            mask, 120, 'AML-Small-LI:val', 1729)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(int(first.sum()), 120)
        self.assertFalse((first & ~mask).any())

    def test_panel_selection_preserves_global_rng(self):
        mask = torch.ones(512, dtype=torch.bool)
        torch.manual_seed(91)
        expected = torch.rand(8)
        torch.manual_seed(91)
        _fixed_target_panel(mask, 64, 'AML-Large-LI:test', 1729)
        actual = torch.rand(8)
        self.assertTrue(torch.equal(actual, expected))

    def test_eval_iterator_replays_same_generator_trajectory(self):
        data = TensorDataset(torch.arange(64))
        generator = torch.Generator().manual_seed(43)
        loader = DataLoader(
            data, batch_size=8, shuffle=True, generator=generator)
        wrapper = LoaderWrapper(
            loader, n_step=len(loader), split='val',
            reset_generator=generator)
        first = torch.cat([batch[0] for batch in wrapper])
        second = torch.cat([batch[0] for batch in wrapper])
        self.assertTrue(torch.equal(first, second))


if __name__ == '__main__':
    unittest.main()
