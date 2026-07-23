import inspect
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from fraudGT.datasets.aml_dataset import AMLDataset
from fraudGT.encoder.hetero_raw_encoder import HeteroRawEdgeEncoder
from fraudGT.sampler.custom_sampler import AddEgoIdsForLinkNeighbor


TASK = ('node', 'to', 'node')


def prefix_data(edge_index, timestamps, count):
    data = HeteroData()
    data['node'].x = torch.ones((5, 1))
    data['node'].num_nodes = 5
    data[TASK].edge_index = edge_index[:, :count]
    data[TASK].edge_attr = torch.arange(
        count * 4, dtype=torch.float32).view(count, 4)
    data[TASK].timestamps = timestamps[:count]
    data[TASK].y = torch.arange(count) % 2
    data[TASK].split_mask = torch.ones(count, dtype=torch.bool)
    return data


class TierSidecarMigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / 'Small-LI' / 'processed').mkdir(parents=True)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])
        timestamps = torch.tensor([0, 1, 2, 3])
        self.dataset = AMLDataset.__new__(AMLDataset)
        self.dataset.root = str(self.root)
        self.dataset.name = 'Small-LI'
        self.dataset.data_dict = {
            'train': prefix_data(edge_index, timestamps, 2),
            'val': prefix_data(edge_index, timestamps, 3),
            'test': prefix_data(edge_index, timestamps, 4),
        }
        pd.DataFrame({
            'Amount Received': [1.0, 3.0, 100.0, 1000.0],
            'Received Currency': [0, 1, 1, 0],
            'Payment Format': [2, 2, 3, 3],
        }).to_csv(
            self.root / 'formatted_transactions_Small-LI.csv',
            index=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_migrates_legacy_cache_to_versioned_sidecar(self):
        self.dataset._ensure_tier_raw_edge_attr()

        sidecar = torch.load(self.dataset.tier_sidecar_path)
        self.assertEqual(sidecar['schema_version'], 2)
        self.assertEqual(sidecar['dataset'], 'Small-LI')
        self.assertEqual(sidecar['source'], 'formatted_csv_v2')
        self.assertEqual(
            sidecar['amount_transform'], 'train_population_zscore')
        self.assertEqual(sidecar['train_end'], 2)
        self.assertEqual(sidecar['num_edges'], 4)
        for split, count in (('train', 2), ('val', 3), ('test', 4)):
            raw = self.dataset.data_dict[split][TASK].raw_edge_attr
            self.assertEqual(raw.shape, (count, 4))
            self.assertTrue(torch.equal(
                raw, sidecar['raw_edge_attr'][:count]))

    def test_reuses_sidecar_without_reading_encoder_edge_attr(self):
        self.dataset._ensure_tier_raw_edge_attr()
        expected = self.dataset.data_dict['test'][TASK].raw_edge_attr.clone()
        os.unlink(self.root / 'formatted_transactions_Small-LI.csv')
        for split in self.dataset.data_dict.values():
            split[TASK].edge_attr.fill_(99999.0)

        self.dataset._ensure_tier_raw_edge_attr()

        self.assertTrue(torch.equal(
            self.dataset.data_dict['test'][TASK].raw_edge_attr, expected))

    def test_recovers_sidecar_when_formatted_csv_is_unavailable(self):
        raw = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 1.0],
            [2.0, 7.0, 2.0, 0.0],
            [3.0, 9.0, 0.0, 2.0],
            [4.0, 100.0, 1.0, 1.0],
            [5.0, 1000.0, 2.0, 2.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 4],
            [1, 2, 3, 4, 4, 1],
        ])

        def cached_prefix(count):
            data = prefix_data(
                edge_index, raw[:, 0].long(), count)
            values = raw[:count]
            data[TASK].edge_attr = (
                (values - values.mean(0, keepdim=True))
                / values.std(0, unbiased=True, keepdim=True)
            )
            return data

        self.dataset.data_dict = {
            'train': cached_prefix(4),
            'val': cached_prefix(5),
            'test': cached_prefix(6),
        }
        os.unlink(self.root / 'formatted_transactions_Small-LI.csv')
        self.dataset._ensure_tier_raw_edge_attr()

        sidecar = torch.load(self.dataset.tier_sidecar_path)
        self.assertEqual(sidecar['source'], 'legacy_cache_affine_v2')
        expected_amount = (
            (raw[:, 1] - raw[:4, 1].mean())
            / raw[:4, 1].std(unbiased=False)
        )
        self.assertTrue(torch.allclose(
            sidecar['raw_edge_attr'][:, 1], expected_amount, atol=1e-5))
        self.assertTrue(torch.equal(
            sidecar['raw_edge_attr'][:, 2], raw[:, 2]))
        self.assertTrue(torch.equal(
            sidecar['raw_edge_attr'][:, 3], raw[:, 3]))

    def test_tier_is_opt_in(self):
        default = inspect.signature(AMLDataset.__init__).parameters[
            'tier_evidence'].default
        self.assertFalse(default)


class TierLinkNeighborIntegrationTest(unittest.TestCase):
    def test_sampling_retains_raw_attributes_and_global_target_ids(self):
        data = HeteroData()
        data['node'].x = torch.ones((5, 1))
        data['node'].num_nodes = 5
        edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 4],
            [1, 2, 3, 4, 4, 1],
        ])
        data[TASK].edge_index = edge_index
        data[TASK].edge_attr = torch.arange(
            24, dtype=torch.float32).view(6, 4)
        data[TASK].raw_edge_attr = data[TASK].edge_attr + 1000.0
        target_edge_ids = torch.tensor([4, 5])
        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1],
            edge_label_index=(TASK, edge_index[:, target_edge_ids]),
            edge_label=torch.tensor([0, 1]),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            transform=AddEgoIdsForLinkNeighbor(
                target_edge_ids=target_edge_ids,
                task=TASK,
                add_ego_ids=False,
            ),
        )

        batch = next(iter(loader))
        store = batch[TASK]
        self.assertTrue(torch.equal(
            store.target_edge_id, target_edge_ids[store.input_id]))
        self.assertTrue(torch.equal(
            store.raw_edge_attr, data[TASK].raw_edge_attr[store.e_id]))

        raw_before = store.raw_edge_attr.clone()
        edge_before = store.edge_attr.clone()
        encoder = HeteroRawEdgeEncoder.__new__(HeteroRawEdgeEncoder)
        nn.Module.__init__(encoder)
        encoder.linear = nn.ModuleDict({
            'node__to__node': nn.Linear(4, 3, bias=False),
        })
        expected = encoder.linear['node__to__node'](edge_before)
        encoder(batch)
        self.assertTrue(torch.equal(store.raw_edge_attr, raw_before))
        self.assertTrue(torch.allclose(store.edge_attr, expected))


if __name__ == '__main__':
    unittest.main()
