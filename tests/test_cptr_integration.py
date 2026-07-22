import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

import fraudGT  # noqa: F401 - register FraudGT configuration extensions
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loss import compute_loss
from fraudGT.head.hetero_edge import HeteroGNNEdgeHead
from fraudGT.train.custom_train import (
    _clip_gradients, _costar_training_terms)


class _DummyDataset(dict):
    def __getitem__(self, key):
        if key == 0:
            key = 'train'
        return super().__getitem__(key)


class CPTRIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[1]
        set_cfg(cfg)
        load_cfg(cfg, SimpleNamespace(
            cfg_file=str(repo / 'configs/cptr/AML-Small-LI.yaml'),
            opts=[]))
        cfg.device = 'cpu'

    @staticmethod
    def _dataset(edge_count):
        task = cfg.dataset.task_entity
        dataset = _DummyDataset()
        for split in ('train', 'val', 'test'):
            data = HeteroData()
            data[task].split_mask = torch.ones(edge_count, dtype=torch.bool)
            dataset[split] = data
        return dataset

    def test_decoder_constructs_with_zero_adapter_and_checkpoint_state(self):
        head = HeteroGNNEdgeHead(4, 2, self._dataset(12))
        self.assertTrue(head.use_cptr)
        self.assertTrue(head.use_dmprd)
        self.assertTrue(head.eg_proto_only)
        self.assertFalse(any(
            isinstance(module, nn.Dropout)
            for module in head.cptr_adapter.modules()))
        self.assertEqual(
            head.cptr_adapter[-1].weight.abs().sum().item(), 0.0)
        state = head.state_dict()
        self.assertIn('cptr_threshold_ema', state)
        self.assertIn('cptr_uplift_variance', state)
        self.assertIn('cptr_uplift_count', state)
        self.assertIn('cptr_env_boundaries', state)

    def test_full_training_forward_preserves_a2_wce_gradient_path(self):
        torch.manual_seed(11)
        task = cfg.dataset.task_entity
        edge_count = 12
        head = HeteroGNNEdgeHead(4, 2, self._dataset(edge_count))
        head._eg_cur_epoch = 20
        head.train()

        batch = HeteroData()
        batch['node'].x = torch.randn(6, 4)
        batch[task].edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5],
            [1, 2, 3, 4, 5, 0, 2, 4, 0, 3, 5, 1],
        ])
        batch[task].edge_attr = torch.randn(edge_count, 4)
        batch[task].y = torch.arange(edge_count) % 2
        batch[task].e_id = torch.arange(edge_count)
        batch[task].input_id = torch.arange(edge_count)
        batch.split = 'train'

        prediction, labels = head(batch)
        self.assertTrue(torch.equal(
            prediction, head._cptr_anchor_logits.detach()))
        anchor_logits, adapter_loss = _costar_training_terms(head)
        anchor_loss, _ = compute_loss(anchor_logits, labels)
        total = anchor_loss + adapter_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()

        adapter_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.cptr_adapter.parameters()
            if parameter.grad is not None)
        anchor_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in head.named_parameters()
            if not name.startswith('cptr_adapter.') and
            parameter.grad is not None)
        self.assertGreater(adapter_gradient, 0.0)
        self.assertGreater(anchor_gradient, 0.0)
        _clip_gradients(head, 1.0, separate_costar=True)


if __name__ == '__main__':
    unittest.main()
