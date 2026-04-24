import json, itertools
import sys, os
import os.path as osp
import pandas as pd
import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from .temporal_dataset import TemporalDataset

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def format_dataset(inPath, outPath=None, dataset_name=None, log_every=5_000_000):
    r'''
    Turn text attributed dataset into a dataset only contains numbers.
    '''
    if outPath is None:
        outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"
    tmp_out_path = outPath + ".tmp"
    tag = dataset_name or os.path.basename(inPath)
    print(f"Formatting AML dataset {tag} from {inPath} -> {outPath}")

    raw = dt.fread(inPath, columns = dt.str32)

    currency = dict()
    paymentFormat = dict()
    bankAcc = dict()
    account = dict()

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = "EdgeID,from_id,to_id,Timestamp,\
    Amount Sent,Sent Currency,Amount Received,Received Currency,\
    Payment Format,Is Laundering\n"

    firstTs = -1

    with open(tmp_out_path, 'w') as writer:
        writer.write(header)
        for i in range(raw.nrows):
            datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
            ts = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year
            hour = datetime_object.hour
            minute = datetime_object.minute

            if firstTs == -1:
                startTime = datetime(year, month, day)
                firstTs = startTime.timestamp() - 10

            ts = ts - firstTs

            cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
            cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

            fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

            fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = raw[i,"To Bank"] + raw[i,4]
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i,"Amount Received"])
            amountPaidOrig = float(raw[i,"Amount Paid"])

            isl = int(raw[i,"Is Laundering"])

            line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                        (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

            writer.write(line)
            if log_every and (i + 1) % log_every == 0:
                print(f"Formatted {i + 1} / {raw.nrows} rows for {tag}")

    formatted = dt.fread(tmp_out_path)
    formatted = formatted[:,:,sort(3)]

    formatted.to_csv(outPath)
    os.remove(tmp_out_path)
    print(f"Finished formatting AML dataset {tag}: {outPath}")

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data['node', 'to', 'node'].timestamps is None else data['node', 'to', 'node'].timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports


def aml_ports(edge_index, num_nodes):
    """Compute AML in/out port ids without Python adjacency lists.

    AML edges are timestamp-sorted, so the first occurrence of each (src, dst)
    pair is also its earliest interaction. We can therefore derive port ids from
    the order in which unique pairs first appear, avoiding the huge Python
    object overhead of per-node adjacency dictionaries.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    pair_keys = src * np.int64(num_nodes) + dst

    pair_codes, unique_pair_keys = pd.factorize(pair_keys, sort=False)
    unique_src = (unique_pair_keys // np.int64(num_nodes)).astype(np.int64, copy=False)
    unique_dst = (unique_pair_keys % np.int64(num_nodes)).astype(np.int64, copy=False)

    unique_in_ports = (
        pd.Series(unique_dst)
        .groupby(unique_dst, sort=False)
        .cumcount()
        .to_numpy(dtype=np.float32, copy=False)
    )
    unique_out_ports = (
        pd.Series(unique_src)
        .groupby(unique_src, sort=False)
        .cumcount()
        .to_numpy(dtype=np.float32, copy=False)
    )

    in_ports = torch.from_numpy(unique_in_ports[pair_codes]).reshape(-1, 1)
    out_ports = torch.from_numpy(unique_out_ports[pair_codes]).reshape(-1, 1)
    return in_ports, out_ports

class AMLDataset(TemporalDataset):
    dataset_sizes = ['Small', 'Medium', 'Large']
    dataset_rates = ['LI', 'HI']
    csv_names = {
        'Small-LI': 'LI-Small_Trans.csv',
        'Small-HI': 'HI-Small_Trans.csv',
        'Medium-LI': 'LI-Medium_Trans.csv',
        'Medium-HI': 'HI-Medium_Trans.csv',
        'Large-LI': 'LI-Large_Trans.csv',
        'Large-HI': 'HI-Large_Trans.csv',
    }

    def __init__(self, root: str, name: str, reverse_mp: bool = False,
                 add_ports: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name # Small-LI
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        assert self.name.split('-')[0] in self.dataset_sizes
        assert self.name.split('-')[1] in self.dataset_rates
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])
        # del self._data['node'].x
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']
            # del self.slices['node', 'rev_to', 'node']
        if add_ports:
            self.ports_dict = torch.load(self.processed_paths[1])
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])

    def add_ports_func(self, data, ports):
        reverse_ports = True
        if not self.reverse_mp:
            # adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            # in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            # out_ports = [ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)] if reverse_ports else []
            in_ports, out_ports = ports
            out_ports = [out_ports]
            data['node', 'to', 'node'].edge_attr = \
                torch.cat([data['node', 'to', 'node'].edge_attr, in_ports] + out_ports, dim=1)
            # return data

        else:
            '''Adds port numberings to the edge features'''
            # adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            # in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            # out_ports = ports(data['node', 'rev_to', 'node'].edge_index, adj_list_out)
            in_ports, out_ports = ports
            data['node', 'to', 'node'].edge_attr = torch.cat([data['node', 'to', 'node'].edge_attr, in_ports], dim=1)
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([data['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        # return [osp.join(self.names[self.name], f) for f in x]
        return []

    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'ports.pt']

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        # data = HeteroData()

        # #data['user'].num_nodes = n_users  # Users do not have any features.
        # data['account'].x = model_htne_pre.node_emb.weight.data.detach().cpu()
        # data["account"].node_id = torch.arange(data.num_nodes)

        # data['account', 'transfer_to', 'account'].edge_index = torch.from_numpy(graph.df_data[["src", 'tar']].T.to_numpy())
        # data['account', 'transfer_to', 'account'].edge_label = torch.from_numpy(graph.df_data['label'].to_numpy())
        
        # train_size = 0.6
        
        # edge_indices = edge_indices = np.arange(data.num_edges)
        # edge_labels = data['account', 'transfer_to', 'account'].edge_label
        # train_indices, temp_indices = train_test_split(edge_indices, train_size=train_size, stratify=edge_labels)
        # val_indices, test_indices = train_test_split(temp_indices, test_size=0.75, stratify=edge_labels[temp_indices])
        
        # # Creating masks based on the full set of indices
        # train_mask = np.isin(range(data.num_edges), train_indices)
        # val_mask = np.isin(range(data.num_edges), val_indices)
        # test_mask = np.isin(range(data.num_edges), test_indices)

        # data['account', 'transfer_to', 'account'].train_mask = torch.from_numpy(train_mask)
        # data['account', 'transfer_to', 'account'].val_mask = torch.from_numpy(val_mask)
        # data['account', 'transfer_to', 'account'].test_mask = torch.from_numpy(test_mask)

        transaction_file = osp.join(
            self.root, f"formatted_transactions_{self.name}.csv"
        )
        if not osp.exists(transaction_file):
            format_dataset(
                osp.join(self.root, self.csv_names[self.name]),
                outPath=transaction_file,
                dataset_name=self.name,
            )
        else:
            print(f"Using cached formatted AML CSV: {transaction_file}")
        read_cols = [
            'from_id',
            'to_id',
            'Timestamp',
            'Amount Received',
            'Received Currency',
            'Payment Format',
            'Is Laundering',
        ]
        dtype_map = {
            'from_id': np.int32,
            'to_id': np.int32,
            'Timestamp': np.int64,
            'Amount Received': np.float32,
            'Received Currency': np.int16,
            'Payment Format': np.int16,
            'Is Laundering': np.int8,
        }
        print(f'Loading formatted AML CSV with constrained dtypes: {transaction_file}')
        df_edges = pd.read_csv(
            transaction_file,
            usecols=read_cols,
            dtype=dtype_map,
        )

        print(f'Available Edge Features: {read_cols}')

        print('Extracting AML columns with minimal-copy tensor conversion...')
        n_samples = int(len(df_edges))
        from_np = df_edges.pop('from_id').to_numpy(copy=False)
        to_np = df_edges.pop('to_id').to_numpy(copy=False)
        timestamps_np = df_edges.pop('Timestamp').to_numpy(dtype=np.int64, copy=False)
        timestamps_np = timestamps_np - timestamps_np.min()
        amount_received_np = df_edges.pop('Amount Received').to_numpy(copy=False)
        received_currency_np = df_edges.pop('Received Currency').to_numpy(copy=False)
        payment_format_np = df_edges.pop('Payment Format').to_numpy(copy=False)
        y_np = df_edges.pop('Is Laundering').to_numpy(copy=False)
        del df_edges

        illicit_count = int(y_np.sum())
        max_n_id = int(max(from_np.max(), to_np.max()) + 1)
        timestamps = torch.from_numpy(timestamps_np)
        y = torch.as_tensor(y_np, dtype=torch.long)
        del y_np

        print('Building AML tensors without stacked intermediate numpy arrays...')
        edge_index = torch.empty((2, n_samples), dtype=torch.long)
        edge_index[0].copy_(torch.as_tensor(from_np, dtype=torch.long))
        edge_index[1].copy_(torch.as_tensor(to_np, dtype=torch.long))

        edge_attr = torch.empty((n_samples, 4), dtype=torch.float32)
        edge_attr[:, 0].copy_(torch.as_tensor(timestamps_np, dtype=torch.float32))
        edge_attr[:, 1].copy_(torch.as_tensor(amount_received_np, dtype=torch.float32))
        edge_attr[:, 2].copy_(torch.as_tensor(received_currency_np, dtype=torch.float32))
        edge_attr[:, 3].copy_(torch.as_tensor(payment_format_np, dtype=torch.float32))

        del from_np, to_np, amount_received_np, received_currency_np, payment_format_np

        print(f"Illicit ratio = {illicit_count} / {len(y)} = {illicit_count / len(y) * 100:.2f}%")
        print(f"Number of nodes (holdings doing transcations) = {int(max_n_id)}")
        print(f"Number of transactions = {n_samples}")

        edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        print(f'Edge features being used: {edge_features}')
        print('Node features being used: [Feature] ("Feature" is a placeholder feature of all 1s)')

        x = torch.ones((int(max_n_id), 1), dtype=torch.float32)
        day_seconds = 3600 * 24
        day_ids = (timestamps_np // day_seconds).astype(np.int64, copy=False)
        n_days = int(day_ids.max() + 1)
        print(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

        daily_totals = np.bincount(day_ids, minlength=n_days)
        daily_cumulative = daily_totals.cumsum()
        del day_ids

        #data splitting
        split_per = [0.6, 0.2, 0.2]
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i,j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v/split_totals_sum for v in split_totals]
                split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
                score = max(split_error) #- (split_totals_sum/total) + 1
                split_scores[(i,j)] = score
            else:
                continue

        i,j = min(split_scores, key=split_scores.get)
        #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
        print(f'Calculate split: {split}')

        train_end = int(daily_cumulative[i - 1]) if i > 0 else 0
        val_end = int(daily_cumulative[j - 1]) if j > 0 else 0
        test_end = int(n_samples)
        print(
            'Derived cumulative split boundaries: '
            f'train_end={train_end}, val_end={val_end}, test_end={test_end}'
        )

        split_bounds = {
            'train': (train_end, 0, train_end),
            'val': (val_end, train_end, val_end),
            'test': (test_end, val_end, test_end),
        }

        
        self.ports_dict = {}
        self.data_dict = {}
        for split in ['train', 'val', 'test']:
            e_count, label_start, label_end = split_bounds[split]
            print(f'Building {split} split with prefix edge count {e_count}')

            # AML edges are timestamp-sorted, so each split is a prefix of the full edge list.
            masked_edge_index = edge_index[:, :e_count]
            masked_edge_attr = z_norm(edge_attr[:e_count])
            masked_y = y[:e_count]
            masked_timestamps = timestamps[:e_count]

            data = HeteroData()
            data['node'].x = x # z_norm(x) will render all x be 0
            data['node'].num_nodes = int(x.shape[0])
            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            # We use "y" here so LinkNeighborLoader won't mess up the edge label
            data['node', 'to', 'node'].y = masked_y
            data['node', 'to', 'node'].timestamps = masked_timestamps
            # if args.ports:
            #     #swap the in- and outgoing port numberings for the reverse edges
            #     data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr

            # Define the labels in the training/validation/test sets
            split_mask = torch.zeros(masked_edge_index.shape[1], dtype=torch.bool)
            split_mask[label_start:label_end] = True
            data['node', 'to', 'node'].split_mask = split_mask

            in_ports, out_ports = aml_ports(masked_edge_index, int(x.shape[0]))
            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])

    def __repr__(self) -> str:
        return f'AML_Dataset(name={self.name})'
