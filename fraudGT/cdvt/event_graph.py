from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CausalEventGraphBatch:
    node_raw: torch.Tensor
    node_target_delta: torch.Tensor
    node_edge_ids: torch.Tensor
    node_graph: torch.Tensor
    target_nodes: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_relation: torch.Tensor
    graph_ptr: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return int(self.target_nodes.numel())

    def to(self, device) -> "CausalEventGraphBatch":
        return CausalEventGraphBatch(**{
            name: value.to(device)
            for name, value in self.__dict__.items()
        })

    def off(self) -> "CausalEventGraphBatch":
        keep = self.target_nodes
        node_raw = self.node_raw[keep]
        node_delta = self.node_target_delta[keep]
        node_ids = self.node_edge_ids[keep]
        node_graph = torch.arange(
            self.num_graphs, device=keep.device, dtype=torch.long)
        target_nodes = node_graph.clone()
        ptr = torch.arange(
            self.num_graphs + 1, device=keep.device, dtype=torch.long)
        return CausalEventGraphBatch(
            node_raw=node_raw,
            node_target_delta=node_delta,
            node_edge_ids=node_ids,
            node_graph=node_graph,
            target_nodes=target_nodes,
            edge_index=torch.empty(
                (2, 0), device=keep.device, dtype=torch.long),
            edge_attr=self.edge_attr.new_empty((0, self.edge_attr.size(1))),
            edge_relation=torch.empty(
                0, device=keep.device, dtype=torch.long),
            graph_ptr=ptr,
        )

    def shuffled(self, generator=None) -> "CausalEventGraphBatch":
        context = torch.ones(
            self.node_raw.size(0), device=self.node_raw.device,
            dtype=torch.bool)
        context[self.target_nodes] = False
        positions = context.nonzero(as_tuple=False).view(-1)
        node_raw = self.node_raw.clone()
        node_delta = self.node_target_delta.clone()
        if positions.numel() > 1:
            permutation = torch.randperm(
                positions.numel(), device=positions.device,
                generator=generator)
            node_raw[positions] = self.node_raw[positions[permutation]]
            node_delta[positions] = self.node_target_delta[
                positions[permutation]]
        edge_attr = self.edge_attr.clone()
        edge_relation = self.edge_relation.clone()
        if edge_attr.size(0) > 1:
            permutation = torch.randperm(
                edge_attr.size(0), device=edge_attr.device,
                generator=generator)
            edge_attr = edge_attr[permutation]
            edge_relation = edge_relation[permutation]
        return CausalEventGraphBatch(
            node_raw=node_raw,
            node_target_delta=node_delta,
            node_edge_ids=self.node_edge_ids,
            node_graph=self.node_graph,
            target_nodes=self.target_nodes,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            edge_relation=edge_relation,
            graph_ptr=self.graph_ptr,
        )


class CausalEventGraphIndex:
    """Build transaction-event DAGs using only target-admissible history."""

    EDGE_ATTR_DIM = 10
    NUM_RELATIONS = 4
    OUT_OUT = 0
    IN_IN = 1
    OUT_IN = 2
    IN_OUT = 3

    def __init__(self, edge_index, timestamps, raw_edge_attr):
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        num_edges = int(edge_index.size(1))
        if timestamps.shape != (num_edges,):
            raise ValueError("timestamps must contain one value per edge")
        if raw_edge_attr.dim() != 2 or raw_edge_attr.size(0) != num_edges:
            raise ValueError("raw_edge_attr must contain one row per edge")
        if raw_edge_attr.size(1) != 4:
            raise ValueError("raw_edge_attr must contain four transaction fields")
        if num_edges and (edge_index < 0).any():
            raise ValueError("account IDs must be non-negative")
        if timestamps.numel() > 1 and (timestamps[1:] < timestamps[:-1]).any():
            raise ValueError("transactions must be timestamp sorted")
        if not torch.isfinite(raw_edge_attr.float()).all():
            raise ValueError("raw_edge_attr must be finite")

        self.edge_index = edge_index.detach().cpu().long().contiguous()
        self.timestamps = timestamps.detach().cpu().long().contiguous()
        self.raw_edge_attr = raw_edge_attr.detach().cpu().float().contiguous()
        self.num_edges = num_edges
        self.num_accounts = (
            int(self.edge_index.max()) + 1 if num_edges else 0)
        self._build_incident_index()

    def _build_incident_index(self):
        edge_ids = torch.arange(self.num_edges, dtype=torch.long)
        accounts = torch.cat((self.edge_index[0], self.edge_index[1]))
        events = torch.cat((edge_ids, edge_ids))
        if accounts.numel():
            order_key = accounts * (self.num_edges + 1) + events
            order = torch.argsort(order_key)
            self.incident_events = events[order]
            counts = torch.bincount(
                accounts[order], minlength=self.num_accounts)
        else:
            self.incident_events = torch.empty(0, dtype=torch.long)
            counts = torch.empty(0, dtype=torch.long)
        self.offsets = torch.zeros(self.num_accounts + 1, dtype=torch.long)
        if counts.numel():
            self.offsets[1:] = counts.cumsum(0)

    def _admissible(self, event_ids, target_id):
        target_time = self.timestamps[target_id]
        times = self.timestamps[event_ids]
        return (times < target_time) | (
            (times == target_time) & (event_ids < target_id))

    def _latest_for_account(self, account, target_id, k):
        start = int(self.offsets[account])
        stop = int(self.offsets[account + 1])
        events = self.incident_events[start:stop]
        events = events[self._admissible(events, target_id)]
        if not events.numel():
            return events
        key = self.timestamps[events] * (self.num_edges + 1) + events
        return events[torch.argsort(key, descending=True)[:k]]

    def _collect_nodes(self, target_id, k, hops, max_events, time_window):
        target_accounts = self.edge_index[:, target_id].tolist()
        selected = {int(target_id)}
        frontier = set(int(account) for account in target_accounts)
        target_time = int(self.timestamps[target_id])
        for _ in range(hops):
            next_frontier = set()
            candidates = []
            for account in sorted(frontier):
                candidates.extend(
                    self._latest_for_account(account, target_id, k).tolist())
            candidates = sorted(
                set(candidates),
                key=lambda event: (
                    int(self.timestamps[event]), int(event)),
                reverse=True,
            )
            for event in candidates:
                if time_window is not None and (
                    target_time - int(self.timestamps[event]) > time_window
                ):
                    continue
                if event in selected:
                    continue
                selected.add(event)
                next_frontier.update(
                    int(account) for account in self.edge_index[:, event])
                if len(selected) >= max_events:
                    break
            if len(selected) >= max_events or not next_frontier:
                break
            frontier = next_frontier
        context = sorted(
            selected - {int(target_id)},
            key=lambda event: (int(self.timestamps[event]), int(event)),
        )
        return context + [int(target_id)]

    def _relation_for_account(self, first, second, account):
        first_src, first_dst = self.edge_index[:, first].tolist()
        second_src, second_dst = self.edge_index[:, second].tolist()
        first_out = first_src == account
        second_out = second_src == account
        if first_out and second_out:
            return self.OUT_OUT, False, True, True
        if not first_out and not second_out:
            return self.IN_IN, False, True, True
        if first_out:
            return self.OUT_IN, True, False, False
        return self.IN_OUT, True, False, False

    def _transition_features(self, first, second, relation_flags):
        relation, role_change, same_side, same_role = relation_flags
        first_raw = self.raw_edge_attr[first]
        second_raw = self.raw_edge_attr[second]
        delta_time = float(self.timestamps[second] - self.timestamps[first])
        first_amount = float(first_raw[1])
        second_amount = float(second_raw[1])
        amount_scale = abs(first_amount) + 1.0
        log_ratio = torch.log(torch.tensor(
            (abs(second_amount) + 1.0) / amount_scale)).item()
        return torch.tensor((
            torch.log1p(torch.tensor(delta_time)).item(),
            log_ratio,
            (second_amount - first_amount) / amount_scale,
            float(relation == self.OUT_OUT),
            float(relation == self.IN_IN),
            float(relation == self.IN_OUT),
            float(relation == self.OUT_IN),
            float(first_raw[2].round() != second_raw[2].round()),
            float(first_raw[3].round() != second_raw[3].round()),
            float(role_change and not same_side and not same_role),
        ), dtype=torch.float32)

    def _build_one(self, target_id, k, hops, max_events, time_window):
        nodes = self._collect_nodes(
            target_id, k, hops, max_events, time_window)
        local = {event: position for position, event in enumerate(nodes)}
        transitions = []
        for second_position, second in enumerate(nodes):
            second_accounts = set(
                int(account) for account in self.edge_index[:, second])
            for account in sorted(second_accounts):
                candidates = []
                for first in nodes[:second_position]:
                    if account not in self.edge_index[:, first].tolist():
                        continue
                    if self._admissible(
                        torch.tensor([first]), second
                    ).item():
                        candidates.append(first)
                candidates.sort(
                    key=lambda event: (
                        int(self.timestamps[event]), int(event)),
                    reverse=True,
                )
                for first in candidates[:k]:
                    flags = self._relation_for_account(
                        first, second, account)
                    transitions.append((
                        local[first], local[second], flags[0],
                        self._transition_features(first, second, flags),
                    ))
        if transitions:
            edge_index = torch.tensor(
                [(row[0], row[1]) for row in transitions],
                dtype=torch.long).t().contiguous()
            relation = torch.tensor(
                [row[2] for row in transitions], dtype=torch.long)
            edge_attr = torch.stack([row[3] for row in transitions])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            relation = torch.empty(0, dtype=torch.long)
            edge_attr = torch.empty((0, self.EDGE_ATTR_DIM))
        node_ids = torch.tensor(nodes, dtype=torch.long)
        target_time = self.timestamps[target_id]
        target_delta = target_time - self.timestamps[node_ids]
        return node_ids, target_delta.float(), edge_index, edge_attr, relation

    def query(
        self,
        target_edge_ids: torch.Tensor,
        k: int = 4,
        hops: int = 2,
        max_events: int = 48,
        time_window: Optional[int] = None,
    ) -> CausalEventGraphBatch:
        if target_edge_ids.dim() != 1 or target_edge_ids.dtype != torch.long:
            raise ValueError("target_edge_ids must be a one-dimensional long tensor")
        if target_edge_ids.numel() and (
            (target_edge_ids < 0).any()
            or (target_edge_ids >= self.num_edges).any()
        ):
            raise ValueError("target edge ID is out of range")
        if k < 1 or hops < 1 or max_events < 2:
            raise ValueError("k, hops, and max_events must be positive")
        if time_window is not None and time_window < 0:
            raise ValueError("time_window must be non-negative")

        output_device = target_edge_ids.device
        rows = [self._build_one(
            int(target), k, hops, max_events, time_window)
            for target in target_edge_ids.detach().cpu().tolist()]
        node_ids = []
        node_delta = []
        node_graph = []
        target_nodes = []
        edge_indices = []
        edge_attrs = []
        edge_relations = []
        ptr = [0]
        offset = 0
        for graph_id, row in enumerate(rows):
            ids, delta, edge_index, edge_attr, relation = row
            node_ids.append(ids)
            node_delta.append(delta)
            node_graph.append(torch.full(
                (ids.numel(),), graph_id, dtype=torch.long))
            target_nodes.append(offset + ids.numel() - 1)
            edge_indices.append(edge_index + offset)
            edge_attrs.append(edge_attr)
            edge_relations.append(relation)
            offset += int(ids.numel())
            ptr.append(offset)

        if rows:
            all_node_ids = torch.cat(node_ids)
            batch = CausalEventGraphBatch(
                node_raw=self.raw_edge_attr[all_node_ids],
                node_target_delta=torch.cat(node_delta),
                node_edge_ids=all_node_ids,
                node_graph=torch.cat(node_graph),
                target_nodes=torch.tensor(target_nodes, dtype=torch.long),
                edge_index=torch.cat(edge_indices, dim=1),
                edge_attr=torch.cat(edge_attrs),
                edge_relation=torch.cat(edge_relations),
                graph_ptr=torch.tensor(ptr, dtype=torch.long),
            )
        else:
            batch = CausalEventGraphBatch(
                node_raw=torch.empty((0, 4)),
                node_target_delta=torch.empty(0),
                node_edge_ids=torch.empty(0, dtype=torch.long),
                node_graph=torch.empty(0, dtype=torch.long),
                target_nodes=torch.empty(0, dtype=torch.long),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.EDGE_ATTR_DIM)),
                edge_relation=torch.empty(0, dtype=torch.long),
                graph_ptr=torch.zeros(1, dtype=torch.long),
            )
        return batch.to(output_device)
