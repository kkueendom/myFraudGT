import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

import fraudGT  # noqa: F401 - registers FraudGT configuration extensions
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loss import compute_loss
from fraudGT.head.hetero_edge import HeteroGNNEdgeHead
from fraudGT.train.custom_train import (
    _clip_gradients, _costar_training_terms)


class DummyDataset(dict):
    def __getitem__(self, key):
        if key == 0:
            key = 'train'
        return super().__getitem__(key)


class CostarIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[1]
        set_cfg(cfg)
        load_cfg(cfg, SimpleNamespace(
            cfg_file=str(repo / 'configs/costar/AML-Small-LI.yaml'),
            opts=[]))
        cfg.device = 'cpu'

    def test_full_decoder_constructs_with_checkpointed_ema(self):
        task = cfg.dataset.task_entity
        dataset = DummyDataset()
        for split in ('train', 'val', 'test'):
            data = HeteroData()
            data[task].split_mask = torch.tensor([True, False, True])
            dataset[split] = data

        head = HeteroGNNEdgeHead(64, 2, dataset)
        self.assertTrue(head.use_costar)
        self.assertTrue(head.use_dmprd)
        self.assertTrue(head.eg_proto_only)
        self.assertFalse(any(
            isinstance(module, nn.Dropout)
            for module in head.costar_router.modules()))
        self.assertTrue(torch.equal(
            head.costar_router[-1].weight,
            torch.zeros_like(head.costar_router[-1].weight)))
        self.assertTrue(torch.equal(
            head.costar_router[-1].bias,
            torch.zeros_like(head.costar_router[-1].bias)))
        state = head.state_dict()
        self.assertIn('costar_router_ema.0.weight', state)
        self.assertIn('costar_center_numerator', state)
        self.assertIn('costar_ema_updates', state)

    def test_training_terms_preserve_exact_a2_task_gradient(self):
        class ToyCostar(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor([
                    [0.2, -0.1], [-0.3, 0.4], [0.1, -0.2], [-0.4, 0.3],
                ]))
                self.costar_router = nn.Linear(2, 2, bias=False)
                self._costar_anchor_logits = None
                self._costar_adapter_loss = None

            def forward(self, features):
                correction = self.costar_router(features.detach())
                self._costar_anchor_logits = self.anchor
                self._costar_adapter_loss = correction.square().mean()
                return self.anchor.detach() + correction

        labels = torch.tensor([0, 1, 0, 1])
        features = torch.tensor([
            [1.0, -1.0], [-0.5, 0.2], [0.3, 0.7], [-0.2, -0.8],
        ])
        model = ToyCostar()
        final_logits = model(features)
        _, _ = compute_loss(final_logits, labels)
        anchor_logits, adapter_loss = _costar_training_terms(model)
        anchor_loss, _ = compute_loss(anchor_logits, labels)
        (anchor_loss + adapter_loss).backward()
        isolated_anchor_grad = model.anchor.grad.detach().clone()
        self.assertGreater(
            model.costar_router.weight.grad.abs().sum().item(), 0.0)

        control_anchor = nn.Parameter(model.anchor.detach().clone())
        control_loss, _ = compute_loss(control_anchor, labels)
        control_loss.backward()
        self.assertTrue(torch.equal(
            isolated_anchor_grad, control_anchor.grad))

        # Separate clipping must keep the anchor update equal to clipping the
        # same A2 gradient without any adapter parameters present.
        _clip_gradients(model, 0.05, separate_costar=True)
        torch.nn.utils.clip_grad_norm_([control_anchor], 0.05)
        self.assertTrue(torch.allclose(
            model.anchor.grad, control_anchor.grad,
            atol=1e-8, rtol=1e-7))

    def test_full_training_forward_keeps_prototype_update_after_prediction(self):
        task = cfg.dataset.task_entity
        edge_count = 8
        dataset = DummyDataset()
        for split in ('train', 'val', 'test'):
            data = HeteroData()
            data[task].split_mask = torch.ones(edge_count, dtype=torch.bool)
            dataset[split] = data
        head = HeteroGNNEdgeHead(4, 2, dataset)
        head._eg_cur_epoch = 20
        head.train()

        batch = HeteroData()
        batch['node'].x = torch.randn(5, 4)
        batch[task].edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 2, 4, 1],
            [1, 2, 3, 4, 2, 4, 0, 3],
        ])
        batch[task].edge_attr = torch.randn(edge_count, 4)
        batch[task].y = torch.arange(edge_count) % 2
        batch[task].e_id = torch.arange(edge_count)
        batch[task].input_id = torch.arange(edge_count)
        batch.split = 'train'

        self.assertEqual(head.support_class_proto_ready.sum().item(), 0.0)
        first_pred, labels = head(batch)
        self.assertTrue(torch.equal(
            first_pred, head._costar_anchor_logits.detach()))
        self.assertGreater(head.support_class_proto_ready.sum().item(), 0.0)

        second_pred, labels = head(batch)
        self.assertTrue(torch.equal(
            second_pred, head._costar_anchor_logits.detach()))
        self.assertTrue(torch.isfinite(head._costar_adapter_loss))
        self.assertIsNotNone(head._costar_diag)
        self.assertIn('platform_width', head._costar_diag)
        head._costar_adapter_loss.backward()
        router_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.costar_router.parameters()
            if parameter.grad is not None)
        self.assertGreater(router_grad, 0.0)

    def test_three_cpu_optimizer_steps_keep_all_gradients_finite(self):
        task = cfg.dataset.task_entity
        edge_count = 8
        dataset = DummyDataset()
        for split in ('train', 'val', 'test'):
            data = HeteroData()
            data[task].split_mask = torch.ones(edge_count, dtype=torch.bool)
            dataset[split] = data
        head = HeteroGNNEdgeHead(4, 2, dataset)
        head._eg_cur_epoch = 20
        head.train()

        batch = HeteroData()
        batch['node'].x = torch.randn(5, 4)
        batch[task].edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 2, 4, 1],
            [1, 2, 3, 4, 2, 4, 0, 3],
        ])
        batch[task].edge_attr = torch.randn(edge_count, 4)
        batch[task].y = torch.arange(edge_count) % 2
        batch[task].e_id = torch.arange(edge_count)
        batch[task].input_id = torch.arange(edge_count)
        batch.split = 'train'

        optimizer = torch.optim.AdamW([
            parameter for parameter in head.parameters()
            if parameter.requires_grad
        ], lr=1e-3, weight_decay=1e-5)
        for _ in range(3):
            optimizer.zero_grad()
            _, labels = head(batch)
            anchor_loss, _ = compute_loss(
                head._costar_anchor_logits, labels)
            loss = anchor_loss + head._costar_adapter_loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [
                parameter.grad for parameter in head.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(
                torch.isfinite(gradient).all() for gradient in gradients))
            _clip_gradients(head, 1.0, separate_costar=True)
            optimizer.step()
            head.costar_update_ema()

        self.assertEqual(head.costar_ema_updates.item(), 3)
        # The first batch initializes prototype banks, so its effective A2
        # residual is zero and correctly does not update orthogonal moments.
        self.assertGreaterEqual(head.costar_center_updates.item(), 2)
        self.assertTrue(torch.isfinite(head.costar_center_numerator).all())
        self.assertGreater(
            head.costar_router[-1].weight.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
