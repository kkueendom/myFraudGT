#!/usr/bin/env python3
"""Run paired, dynamic-random diagnostics on a trained COSTAR checkpoint."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric import seed_everything

import fraudGT  # noqa: F401 - register project modules
from fraudGT.analysis.costar_evidence import (
    contribution_audit,
    error_subset,
    evaluate_variants,
    f1,
    gradient_group_summary,
)
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.model_builder import create_model
from fraudGT.train.custom_train import _costar_training_terms
from fraudGT.train.custom_train import _set_evidence_gate_epoch


CAPTURE_KEYS = (
    "edge_id", "labels", "base_margin", "anchor_margin", "final_margin",
    "prototype_margin_delta", "costar_margin_delta",
    "total_evidence_margin_delta", "prototype_alpha", "prototype_ready",
    "prototype_reliability", "prototype_support_margin", "router_current",
    "router_ema", "router_consensus", "router_consistency",
    "residual_weight", "fallback_mask",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-commit", default="4abe58c6")
    parser.add_argument("--required-last-epoch", type=int, default=499)
    parser.add_argument("--gradient-batches", type=int, default=16)
    return parser.parse_args()


def git_commit(repo):
    return subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=repo,
        text=True).strip()


def protocol_audit(repo):
    if bool(cfg.val.fixed_target_panel):
        raise RuntimeError("val.fixed_target_panel must be False")
    sampler = (repo / "fraudGT" / "sampler" / "custom_sampler.py").read_text()
    loader = (repo / "fraudGT" / "graphgym" / "loader.py").read_text()
    forbidden = ("_fixed_target_panel", "reset_generator",
                 "generator=reset_generator")
    if any(token in sampler for token in forbidden):
        raise RuntimeError("fixed-panel or sampler RNG restore logic detected")
    if "shuffle=shuffle" not in sampler:
        raise RuntimeError("LinkNeighborLoader does not forward shuffle")
    if "def create_loader(dataset = None, shuffle = True" not in loader:
        raise RuntimeError("create_loader no longer defaults to shuffle=True")


def configure(args, repo):
    set_cfg(cfg)
    config_args = SimpleNamespace(
        cfg_file=args.cfg,
        opts=["seed", str(args.seed), "val.fixed_target_panel", "False"],
    )
    load_cfg(cfg, config_args)
    cfg.seed = args.seed
    cfg.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)
    protocol_audit(repo)


def find_costar_head(model):
    heads = [module for module in model.modules()
             if getattr(module, "use_costar", False)]
    if len(heads) != 1:
        raise RuntimeError(f"expected one COSTAR head, found {len(heads)}")
    return heads[0]


@torch.no_grad()
def collect_split(model, head, loader, split):
    model.eval()
    captured = {key: [] for key in CAPTURE_KEYS}
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        model(batch)
        analysis = head._costar_analysis
        if analysis is None:
            raise RuntimeError("COSTAR analysis capture was not populated")
        for key in CAPTURE_KEYS:
            captured[key].append(analysis[key].detach().cpu())
    return {
        key: torch.cat(values, dim=0).numpy()
        for key, values in captured.items()
    }


def gradient_audit(model, loader, batch_limit, epoch):
    if batch_limit < 1:
        raise ValueError("gradient batch limit must be positive")
    _set_evidence_gate_epoch(model, epoch)
    iterator = iter(loader)
    records = []
    for _ in range(batch_limit):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        model.train()
        model.zero_grad(set_to_none=True)
        batch.split = "train"
        batch.to(torch.device(cfg.device))
        _, labels = model(batch)
        anchor_logits, adapter_loss = _costar_training_terms(model)
        anchor_loss, _ = compute_loss(anchor_logits, labels)
        total = anchor_loss + adapter_loss
        total.backward()
        records.append({
            "anchor_loss": float(anchor_loss.detach()),
            "adapter_loss": float(adapter_loss.detach()),
            "total_loss": float(total.detach()),
            "positive_count": int((labels == 1).sum().detach()),
            "groups": gradient_group_summary(model.named_parameters()),
        })

    group_names = (
        "costar_router", "prototype_branch", "z_base_decoder",
        "encoder_and_other")
    groups = {}
    for name in group_names:
        rows = [record["groups"][name] for record in records]
        l2 = np.asarray([row["l2_norm"] for row in rows], dtype=float)
        groups[name] = {
            "parameter_numel_with_grad": max(
                (row["numel"] for row in rows), default=0),
            "mean_l2_norm": float(l2.mean()) if l2.size else 0.0,
            "max_l2_norm": float(l2.max()) if l2.size else 0.0,
            "nonzero_batch_fraction": (
                float((l2 > 0).mean()) if l2.size else 0.0),
            "mean_abs_gradient": float(np.mean([
                row["mean_abs"] for row in rows])) if rows else 0.0,
        }
    return {
        "batch_count": len(records),
        "batches_with_positives": sum(
            record["positive_count"] > 0 for record in records),
        "mean_anchor_loss": float(np.mean([
            record["anchor_loss"] for record in records])) if records else 0.0,
        "mean_adapter_loss": float(np.mean([
            record["adapter_loss"] for record in records])) if records else 0.0,
        "groups": groups,
    }


def read_rows(path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "epoch" in row:
            rows.append(row)
    return rows


def delta_status(delta):
    if delta >= 0.005:
        return "gain_ge_0.005"
    if delta <= -0.005:
        return "loss_ge_0.005"
    return "possible_dynamic_sampling_variation"


def formal_result(run_dir, repo, required_last_epoch):
    train = read_rows(run_dir / "train" / "stats.json")
    val = [row for row in read_rows(run_dir / "val" / "stats.json")
           if "f1" in row]
    test = [row for row in read_rows(run_dir / "test" / "stats.json")
            if "f1" in row]
    if not train or not val or not test:
        raise RuntimeError("formal run is missing train/val/test rows")
    last_epoch = int(train[-1]["epoch"])
    if last_epoch < required_last_epoch:
        raise RuntimeError(
            f"formal run is partial: {last_epoch} < {required_last_epoch}")
    test_by_epoch = {int(row["epoch"]): row for row in test}
    selected_val = max(val, key=lambda row: float(row["f1"]))
    selected_epoch = int(selected_val["epoch"])
    selected_test = test_by_epoch[selected_epoch]
    raw_test = max(test, key=lambda row: float(row["f1"]))
    baseline = json.loads(
        (repo / "run" / "dynamic_random_a2_baseline.json").read_text()
    )["datasets"][cfg.dataset.name]
    selected = float(selected_test["f1"])
    raw = float(raw_test["f1"])
    delta_selected = selected - baseline["val_selected_test_f1"]
    delta_raw = raw - baseline["raw_best_test_f1"]
    return {
        "last_epoch": last_epoch,
        "selected_epoch": selected_epoch,
        "val_selected_test_f1": selected,
        "a2_val_selected_test_f1": baseline["val_selected_test_f1"],
        "delta_val_selected_f1": delta_selected,
        "val_selected_status": delta_status(delta_selected),
        "raw_best_epoch": int(raw_test["epoch"]),
        "raw_best_test_f1": raw,
        "a2_raw_best_test_f1": baseline["raw_best_test_f1"],
        "delta_raw_best_f1": delta_raw,
        "raw_best_status": delta_status(delta_raw),
        "sampling_protocol": "dynamic_random",
    }


def paired_permutations(val_count, test_count):
    return (
        torch.randperm(val_count).numpy(),
        torch.randperm(test_count).numpy(),
    )


def render_markdown(result):
    lines = [
        "## Material Passport",
        "",
        "- Type: Experiment validation report",
        "- Verification Status: ANALYZED",
        f"- Dataset: {result['experiment']['dataset']}",
        "- Sampling protocol: dynamic_random",
        f"- Model commit: `{result['experiment']['model_commit']}`",
        f"- Diagnostic commit: `{result['experiment']['diagnostic_commit']}`",
        f"- Checkpoint: `{result['experiment']['checkpoint']}`",
        "",
        "## Formal Result",
        "",
        "| Metric | COSTAR | Initial A2 | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    formal = result["formal_result"]
    lines.extend([
        "| Val-selected Test F1 | "
        f"{formal['val_selected_test_f1']:.5f} | "
        f"{formal['a2_val_selected_test_f1']:.5f} | "
        f"{formal['delta_val_selected_f1']:+.5f} | "
        f"{formal['val_selected_status']} |",
        "| Raw-best Test F1 | "
        f"{formal['raw_best_test_f1']:.5f} | "
        f"{formal['a2_raw_best_test_f1']:.5f} | "
        f"{formal['delta_raw_best_f1']:+.5f} | "
        f"{formal['raw_best_status']} |",
        "",
        "## Paired Diagnostic F1",
        "",
        "Each row uses a threshold selected on the diagnostic validation sample "
        "and applies it to the paired diagnostic test sample.",
        "",
        "| Variant | Val F1 | Test F1 | Threshold |",
        "|---|---:|---:|---:|",
    ])
    for name, row in result["paired_diagnostic_f1"].items():
        lines.append(
            f"| {name} | {row['val_f1']:.5f} | {row['test_f1']:.5f} | "
            f"{row['threshold']:.6f} |")
    subset = result["a2_error_subset"]
    changes = result["contribution"]["costar_vs_a2_decision_changes"]
    contribution = result["contribution"]
    lines.extend([
        "",
        "## Complementarity",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| A2 wrong / evidence right | {subset['a2_wrong_evidence_right']} |",
        f"| A2 right / evidence wrong | {subset['a2_right_evidence_wrong']} |",
        f"| Both wrong | {subset['both_wrong']} |",
        f"| Both right | {subset['both_right']} |",
        "| Evidence correction rate within A2 errors | "
        f"{subset['correction_rate_within_a2_errors']:.6f} |",
        "| Evidence damage rate within A2 correct | "
        f"{subset['damage_rate_within_a2_correct']:.6f} |",
        f"| Net corrected minus broken | {subset['net_corrected_minus_broken']} |",
        "",
        "## Contribution",
        "",
        "- COSTAR hard gate present: no",
        "- Applied nontrivial correction rate: "
        f"{contribution['applied_correction_rate']:.6f}",
        "- Diagnostic confidence/open rate: "
        f"{contribution['diagnostic_open_rate']:.6f}",
        f"- COSTAR changed A2 decisions: {changes['changed_count']} / "
        f"{changes['sample_count']} ({changes['changed_rate']:.6f})",
        f"- Changed decisions corrected: {changes['corrected_count']}",
        f"- Changed decisions broken: {changes['broken_count']}",
        "- Full quantiles, relative contributions, weights, and gradient norms "
        "are in the adjacent JSON file.",
        "",
        "## Interpretation Gate",
        "",
        "Interpretation is deferred until both datasets have completed and all "
        "four diagnostics have been compared. A formal F1 gain alone is not "
        "evidence that COSTAR caused the gain.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    configure(args, repo)

    loaders, dataset = create_loader(returnDataset=True)
    model = create_model(dataset=dataset)
    checkpoint = torch.load(args.checkpoint, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = find_costar_head(model)

    val = collect_split(model, head, loaders[1], "val")
    test = collect_split(model, head, loaders[2], "test")
    val_perm, test_perm = paired_permutations(
        len(val["labels"]), len(test["labels"]))
    variant_metrics, _, test_variants = evaluate_variants(
        val, test, val_perm, test_perm)
    subset = error_subset(
        test["labels"],
        test_variants["a2_anchor"],
        test_variants["evidence_only_total"],
        variant_metrics["a2_anchor"]["threshold"],
        variant_metrics["evidence_only_total"]["threshold"],
    )
    contribution = contribution_audit(
        test, variant_metrics["costar_full"]["threshold"])
    gradients = gradient_audit(
        model, loaders[0], args.gradient_batches, args.required_last_epoch)

    result = {
        "experiment": {
            "dataset": cfg.dataset.name,
            "model": "COSTAR",
            "variant": "COSTAR-evidence-diagnostics",
            "seed": args.seed,
            "model_commit": args.model_commit,
            "diagnostic_commit": git_commit(repo),
            "config": str(Path(args.cfg).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "formal_run_dir": str(Path(args.formal_run_dir).resolve()),
            "sampling_protocol": "dynamic_random",
            "val_sample_count": int(len(val["labels"])),
            "test_sample_count": int(len(test["labels"])),
        },
        "formal_result": formal_result(
            Path(args.formal_run_dir), repo, args.required_last_epoch),
        "paired_diagnostic_f1": variant_metrics,
        "a2_error_subset": subset,
        "contribution": contribution,
        "gradient_audit": gradients,
        "shuffle_test": {},
    }
    shared_threshold = variant_metrics["costar_full"]["threshold"]
    normal_f1 = f1(
        test["labels"], test_variants["costar_full"], shared_threshold)
    shuffled_f1 = f1(
        test["labels"], test_variants["evidence_shuffled"],
        shared_threshold)
    off_f1 = f1(
        test["labels"], test_variants["evidence_off"], shared_threshold)
    result["shuffle_test"] = {
        "shared_normal_threshold": shared_threshold,
        "normal_test_f1": normal_f1,
        "shuffled_test_f1": shuffled_f1,
        "off_test_f1": off_f1,
        "normal_minus_shuffled": normal_f1 - shuffled_f1,
        "normal_minus_off": normal_f1 - off_f1,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(result))
    print(json.dumps({
        "output": str(output),
        "dataset": cfg.dataset.name,
        "formal": result["formal_result"],
        "shuffle": result["shuffle_test"],
        "changed": contribution["costar_vs_a2_decision_changes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
