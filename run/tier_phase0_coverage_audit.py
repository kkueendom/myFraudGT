#!/usr/bin/env python3
"""Audit TIER raw-evidence coverage under dynamic FraudGT sampling."""

import argparse
import json
import os
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import mask_to_index

from fraudGT.datasets.aml_dataset import AMLDataset
from fraudGT.evidence.tier import TemporalIncidentIndex
from fraudGT.sampler.custom_sampler import AddEgoIdsForLinkNeighbor


TASK = ('node', 'to', 'node')
ROLE_NAMES = ('dst_to_src', 'src_from_src', 'dst_to_dst',
              'src_from_dst', 'reverse')
MOTIF_NAMES = ('reciprocal', 'relay', 'cycle')


def git_output(repo, *args):
    result = subprocess.run(
        ['git', *args], cwd=str(repo), text=True, capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def verify_repository(repo, spec):
    status = git_output(repo, 'status', '--porcelain')
    if status:
        raise RuntimeError('refusing to audit from a dirty worktree')
    branch = git_output(repo, 'branch', '--show-current')
    expected = spec.get('expected_branch')
    if expected and branch and branch != expected:
        raise RuntimeError(f'expected branch {expected}, found {branch}')
    return git_output(repo, 'rev-parse', '--short=8', 'HEAD')


def load_config(repo, spec, task):
    template = str(spec['config_template'])
    if '{dataset}' not in template:
        raise ValueError('config_template must contain {dataset}')
    relative_path = template.format(dataset=task['dataset'])
    path = repo / relative_path
    config = yaml.safe_load(path.read_text())
    return relative_path, config


def audit_protocol(repo, spec, task, config):
    if spec.get('sampling_protocol') != 'dynamic_random':
        raise ValueError('sampling_protocol must be dynamic_random')
    if config['dataset'].get('tier_evidence') is not True:
        raise ValueError('dataset.tier_evidence must be true')
    if config['val'].get('fixed_target_panel') is not False:
        raise ValueError('val.fixed_target_panel must be false')
    if int(config['train']['batch_size']) != int(
            task['expected_batch_size']):
        raise ValueError('train.batch_size differs from the registered value')
    if int(config['val']['iter_per_epoch']) != int(
            task['expected_val_iter_per_epoch']):
        raise ValueError(
            'val.iter_per_epoch differs from the registered value')

    sampler_source = (
        repo / 'fraudGT' / 'sampler' / 'custom_sampler.py').read_text()
    loader_source = (
        repo / 'fraudGT' / 'graphgym' / 'loader.py').read_text()
    forbidden = (
        '_fixed_target_panel', 'reset_generator',
        'generator=reset_generator',
    )
    if any(token in sampler_source for token in forbidden):
        raise RuntimeError('fixed-panel or RNG-reset sampler logic detected')
    if 'shuffle=shuffle' not in sampler_source:
        raise RuntimeError('LinkNeighborLoader no longer forwards shuffle')
    if 'def create_loader(dataset = None, shuffle = True' not in loader_source:
        raise RuntimeError('create_loader no longer defaults to shuffle=True')


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == 'Darwin':
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def tensor_distribution(values):
    values = values.detach().float().view(-1)
    if not values.numel():
        return {
            'count': 0, 'mean': None, 'median': None, 'p90': None,
            'max': None,
        }
    return {
        'count': int(values.numel()),
        'mean': float(values.mean()),
        'median': float(torch.quantile(values, 0.5)),
        'p90': float(torch.quantile(values, 0.9)),
        'max': float(values.max()),
    }


def coverage_decision(row, gate):
    positives = row['class_counts'].get('1', 0)
    if positives < int(gate['min_positive_samples']):
        return 'inconclusive_too_few_positive_samples'
    if row['coverage_rate'] < float(gate['min_overall_coverage']):
        return 'fail_overall_coverage'
    positive_coverage = row['class_coverage_rate'].get('1')
    if (
        positive_coverage is None
        or positive_coverage < float(gate['min_positive_coverage'])
    ):
        return 'fail_positive_coverage'
    return 'pass'


def build_loader(data, target_edge_ids, config, split):
    store = data[TASK]
    batch_size = int(config['train']['batch_size'])
    return LinkNeighborLoader(
        data=data,
        num_neighbors=config['train']['neighbor_sizes'],
        edge_label_index=(TASK, store.edge_index[:, target_edge_ids]),
        edge_label=store.y[target_edge_ids],
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        transform=AddEgoIdsForLinkNeighbor(
            target_edge_ids=target_edge_ids,
            task=TASK,
            add_ego_ids=bool(config['train'].get('add_ego_id', False)),
        ),
    )


def validate_batch_mapping(data, batch, target_edge_ids):
    store = batch[TASK]
    expected_targets = target_edge_ids[store.input_id]
    if not torch.equal(store.target_edge_id, expected_targets):
        raise AssertionError('global target_edge_id mapping is incorrect')
    expected_raw = data[TASK].raw_edge_attr[store.e_id]
    if not torch.equal(store.raw_edge_attr, expected_raw):
        raise AssertionError('sampled raw_edge_attr is not aligned with e_id')


def validate_temporal_admissibility(index, target_ids, evidence):
    for row_index, target_id in enumerate(target_ids.tolist()):
        context = evidence.context_edge_ids[row_index][evidence.mask[row_index]]
        if not context.numel():
            continue
        if (context == target_id).any():
            raise AssertionError('target edge leaked into its own evidence')
        context_times = index.timestamps[context]
        target_time = index.timestamps[target_id]
        admissible = (context_times < target_time) | (
            (context_times == target_time) & (context < target_id))
        if not admissible.all():
            raise AssertionError('future or tie-ordered context edge detected')


def audit_split(index, data, split, config, spec):
    split_started = time.monotonic()
    target_edge_ids = mask_to_index(data[TASK].split_mask)
    loader = build_loader(data, target_edge_ids, config, split)
    max_batches = min(
        int(spec['max_batches']),
        int(config[
            'train' if split == 'train' else 'val']['iter_per_epoch']),
    )
    min_batches = int(spec['min_batches'])
    min_positive_targets = int(spec['min_positive_targets'])

    token_counts = []
    supports = []
    labels = []
    role_activations = torch.zeros(len(ROLE_NAMES), dtype=torch.float64)
    motif_activations = torch.zeros(len(MOTIF_NAMES), dtype=torch.float64)
    valid_token_count = 0
    sampled_target_count = 0
    positive_target_count = 0
    unique_target_ids = set()
    query_seconds = 0.0
    batches = 0

    for batch in loader:
        validate_batch_mapping(data, batch, target_edge_ids)
        batch_targets = batch[TASK].target_edge_id.detach().cpu().long()
        batch_labels = batch[TASK].edge_label.detach().cpu().long()
        query_started = time.monotonic()
        evidence = index.query(
            batch_targets,
            max_tokens=int(spec['max_tokens']),
            time_window=spec.get('time_window'),
        )
        query_seconds += time.monotonic() - query_started
        validate_temporal_admissibility(index, batch_targets, evidence)

        counts = evidence.mask.sum(dim=1).cpu()
        token_counts.append(counts)
        supports.append(evidence.support.cpu())
        labels.append(batch_labels)
        sampled_target_count += int(batch_targets.numel())
        positive_target_count += int((batch_labels == 1).sum())
        unique_target_ids.update(batch_targets.tolist())

        valid_tokens = evidence.tokens[evidence.mask].cpu()
        valid_token_count += int(valid_tokens.size(0))
        if valid_tokens.numel():
            role_start = index.raw_edge_attr.size(1) + 1
            motif_start = role_start + index.ROLE_DIM
            role_activations += valid_tokens[
                :, role_start:motif_start].double().sum(0)
            motif_activations += valid_tokens[
                :, motif_start:motif_start + index.MOTIF_DIM].double().sum(0)

        batches += 1
        if (
            batches >= min_batches
            and positive_target_count >= min_positive_targets
        ) or batches >= max_batches:
            break

    all_counts = torch.cat(token_counts) if token_counts else torch.empty(0)
    all_support = (
        torch.cat(supports) if supports
        else torch.empty((0, index.SUPPORT_DIM))
    )
    all_labels = torch.cat(labels) if labels else torch.empty(
        0, dtype=torch.long)
    covered = all_counts > 0
    class_counts = {}
    class_coverage = {}
    class_token_distribution = {}
    for class_id in (0, 1):
        class_mask = all_labels == class_id
        count = int(class_mask.sum())
        class_counts[str(class_id)] = count
        class_coverage[str(class_id)] = (
            float(covered[class_mask].float().mean()) if count else None)
        class_token_distribution[str(class_id)] = tensor_distribution(
            all_counts[class_mask])

    denominator = max(valid_token_count, 1)
    row = {
        'split': split,
        'batches': batches,
        'sampled_targets': sampled_target_count,
        'unique_targets': len(unique_target_ids),
        'duplicate_target_rate': (
            1.0 - len(unique_target_ids) / sampled_target_count
            if sampled_target_count else 0.0
        ),
        'class_counts': class_counts,
        'coverage_rate': (
            float(covered.float().mean()) if covered.numel() else 0.0),
        'class_coverage_rate': class_coverage,
        'token_count': tensor_distribution(all_counts),
        'class_token_count': class_token_distribution,
        'support_mean': (
            all_support.float().mean(0).tolist()
            if all_support.numel() else [0.0] * index.SUPPORT_DIM
        ),
        'role_token_activation_rate': {
            name: float(role_activations[position] / denominator)
            for position, name in enumerate(ROLE_NAMES)
        },
        'motif_token_activation_rate': {
            name: float(motif_activations[position] / denominator)
            for position, name in enumerate(MOTIF_NAMES)
        },
        'valid_evidence_tokens': valid_token_count,
        'query_seconds': query_seconds,
        'elapsed_seconds': time.monotonic() - split_started,
        'loader_shuffle': True,
        'loader_generator': None,
        'batch_size': int(config['train']['batch_size']),
        'configured_iter_per_epoch': int(config[
            'train' if split == 'train' else 'val']['iter_per_epoch']),
        'diagnostic_batch_cap': max_batches,
    }
    row['coverage_decision'] = coverage_decision(
        row, spec['coverage_gate'])
    return row


def audit_task(repo, spec, task, commit):
    relative_config, config = load_config(repo, spec, task)
    audit_protocol(repo, spec, task, config)
    seed = int(task['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    started = time.monotonic()
    rss_before = peak_rss_mb()
    dataset_root = Path(config['dataset']['dir']) / 'AML'
    dataset = AMLDataset(
        root=str(dataset_root),
        name=task['dataset'],
        reverse_mp=bool(config['dataset'].get('reverse_mp', False)),
        add_ports=bool(config['dataset'].get('add_ports', False)),
        tier_evidence=True,
    )
    load_seconds = time.monotonic() - started
    test_store = dataset['test'][TASK]
    index_started = time.monotonic()
    index = TemporalIncidentIndex(
        edge_index=test_store.edge_index,
        timestamps=test_store.timestamps,
        raw_edge_attr=test_store.raw_edge_attr,
    )
    index_seconds = time.monotonic() - index_started

    rows = []
    for split in spec['splits']:
        row = audit_split(index, dataset[split], split, config, spec)
        row.update({
            'dataset': task['dataset'],
            'seed': seed,
            'git_commit': commit,
            'config': relative_config,
            'sampling_protocol': 'dynamic_random',
        })
        rows.append(row)

    sidecar = Path(dataset.tier_sidecar_path)
    decisions = [row['coverage_decision'] for row in rows]
    overall = (
        'pass' if all(decision == 'pass' for decision in decisions)
        else 'inconclusive' if any(
            decision.startswith('inconclusive') for decision in decisions)
        else 'fail'
    )
    task_summary = {
        'dataset': task['dataset'],
        'model': 'TIER',
        'variant': spec['variant'],
        'seed': seed,
        'git_commit': commit,
        'config': relative_config,
        'checkpoint': 'not_applicable_no_training',
        'val_selected_test_f1': None,
        'raw_best_test_f1': None,
        'delta_val_selected_f1': None,
        'delta_raw_best_f1': None,
        'sampling_protocol': 'dynamic_random',
        'experiment_type': 'coverage_audit_no_training',
        'coverage_decision': overall,
        'dataset_load_seconds': load_seconds,
        'index_build_seconds': index_seconds,
        'total_elapsed_seconds': time.monotonic() - started,
        'peak_rss_before_mb': rss_before,
        'peak_rss_after_mb': peak_rss_mb(),
        'sidecar': str(sidecar),
        'sidecar_bytes': sidecar.stat().st_size,
    }
    return rows, task_summary


def fmt(value):
    return '-' if value is None else f'{value:.4f}'


def write_summary(path, rows, manifests, commit):
    lines = [
        '# TIER Phase 0 Evidence Coverage Audit',
        '',
        '## Material Passport',
        '',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run',
        '- Verification Status: EXECUTED',
        f'- Git Commit: `{commit}`',
        '- Sampling Protocol: `dynamic_random`',
        '- Model Training: none',
        '',
        '## Coverage',
        '',
        '| Dataset | Split | Samples | Positives | Coverage | Positive coverage | Tokens mean | Tokens p90 | Decision |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | "
            f"{row['sampled_targets']} | {row['class_counts']['1']} | "
            f"{fmt(row['coverage_rate'])} | "
            f"{fmt(row['class_coverage_rate']['1'])} | "
            f"{fmt(row['token_count']['mean'])} | "
            f"{fmt(row['token_count']['p90'])} | "
            f"{row['coverage_decision']} |"
        )
    lines.extend([
        '',
        '## Evidence Activation',
        '',
        '| Dataset | Split | Reverse role | Reciprocal | Relay | Cycle |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in rows:
        roles = row['role_token_activation_rate']
        motifs = row['motif_token_activation_rate']
        lines.append(
            f"| {row['dataset']} | {row['split']} | "
            f"{roles['reverse']:.4f} | {motifs['reciprocal']:.4f} | "
            f"{motifs['relay']:.4f} | {motifs['cycle']:.4f} |"
        )
    lines.extend([
        '',
        '## Resource Summary',
        '',
        '| Dataset | Load (s) | Index (s) | Total (s) | Peak RSS (MB) | Sidecar (bytes) | Decision |',
        '|---|---:|---:|---:|---:|---:|---|',
    ])
    for item in manifests:
        lines.append(
            f"| {item['dataset']} | {item['dataset_load_seconds']:.2f} | "
            f"{item['index_build_seconds']:.2f} | "
            f"{item['total_elapsed_seconds']:.2f} | "
            f"{item['peak_rss_after_mb']:.1f} | "
            f"{item['sidecar_bytes']} | {item['coverage_decision']} |"
        )
    lines.extend([
        '',
        'A Phase 0 pass only authorizes evidence-only qualification. It is not',
        'evidence of predictive improvement and does not contain an F1 result.',
        '',
    ])
    path.write_text('\n'.join(lines))


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--spec', type=Path,
        default=repo / 'run' / 'tier_phase0_coverage_spec.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    commit = verify_repository(repo, spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifests = []
    for task in spec['tasks']:
        task_rows, manifest = audit_task(repo, spec, task, commit)
        rows.extend(task_rows)
        manifests.append(manifest)
        print(json.dumps(manifest, sort_keys=True), flush=True)

    (args.output_dir / 'coverage_rows.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    (args.output_dir / 'experiment_manifest.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n'
                for row in manifests))
    write_summary(
        args.output_dir / 'phase0_summary.md', rows, manifests, commit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
