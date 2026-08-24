#!/usr/bin/env python3
"""Summarize the minimal CDVT ablation and mechanism evidence package."""

import argparse
import json
import statistics
from pathlib import Path


DATASETS = ("Small-LI", "Medium-LI", "Large-LI")
CORE_VARIANTS = ("account_only", "event_only", "dual_view")
SAMPLING_BAND = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def final_path(dataset, phase1_root, phase2_root):
    root = phase1_root if dataset in {"Small-LI", "Large-LI"} else phase2_root
    return root / f"{dataset}_dual_view_seed42" / "manifest.json"


def core_path(dataset, variant, phase1_root, phase2_root, ablation_root):
    if variant == "dual_view":
        return final_path(dataset, phase1_root, phase2_root)
    root = phase1_root if dataset in {"Small-LI", "Large-LI"} else ablation_root
    return root / f"{dataset}_{variant}_seed42" / "manifest.json"


def followup_path(dataset, variant, ablation_root):
    return ablation_root / f"{dataset}_{variant}_seed42" / "manifest.json"


def protocol_settings(payload):
    snapshot = payload.get("config_snapshot", {})
    cdvt = snapshot.get("cdvt", {})
    return {
        "history_k": int(payload.get(
            "history_k", cdvt.get("history_k", -1))),
        "use_relation_types": payload.get(
            "use_relation_types", cdvt.get("use_relation_types", True)),
    }


def validate_manifest(
    payload, path, dataset, variant, architecture, history_k=4,
    use_relation_types=True, require_ablation_phase=False
):
    expected = {
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": architecture,
        "seed": 42,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, "
                f"found {payload.get(key)!r}")
    if float(payload.get("lambda_cons", -1)) != 0.0:
        raise ValueError(f"{path}: ablation must use lambda_cons=0")
    if require_ablation_phase and payload.get("phase") != "CDVT_ablation":
        raise ValueError(f"{path}: new control must be CDVT_ablation")
    snapshot = payload.get("config_snapshot", {})
    train = snapshot.get("train", {})
    val = snapshot.get("val", {})
    settings = protocol_settings(payload)
    protocol = (
        snapshot.get("dataset", {}).get("tier_evidence") is True
        and train.get("sampler") == "link_neighbor"
        and int(train.get("iter_per_epoch", -1)) == 256
        and int(train.get("batch_size", -1)) == 2048
        and int(train.get("eval_period", -1)) == 4
        and int(val.get("iter_per_epoch", -1)) == 256
        and val.get("fixed_target_panel") is False
        and settings["history_k"] == history_k
        and settings["use_relation_types"] is use_relation_types
    )
    if not protocol:
        raise ValueError(f"{path}: manifest violates the requested protocol")


def load_manifest(path, missing, allow_incomplete):
    if not path.is_file():
        missing.append(str(path))
        if allow_incomplete:
            return None
        raise FileNotFoundError(f"missing experiment manifest: {path}")
    return json.loads(path.read_text())


def compact_result(payload, path):
    event = payload["best_event"]
    steps = int(event["test_steps"])
    return {
        "val_selected_test_f1": float(payload["val_selected_test_f1"]),
        "raw_best_test_f1": float(payload["raw_best_test_f1"]),
        "val_selected_epoch": int(payload["val_selected_epoch"]),
        "parameter_count": int(payload["parameter_count"]),
        "peak_gpu_memory_bytes": int(payload["peak_gpu_memory_bytes"]),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "test_seconds_per_batch": float(event["test_seconds"]) / max(steps, 1),
        "git_commit": str(payload["git_commit"]),
        "manifest": str(path.resolve()),
    }


def build_summary(
    phase1_root, phase2_root, ablation_root, allow_incomplete=False
):
    phase1_root = Path(phase1_root)
    phase2_root = Path(phase2_root)
    ablation_root = Path(ablation_root)
    missing = []
    core = []
    final_payloads = {}
    for dataset in DATASETS:
        for variant in CORE_VARIANTS:
            path = core_path(
                dataset, variant, phase1_root, phase2_root, ablation_root)
            payload = load_manifest(path, missing, allow_incomplete)
            if payload is None:
                continue
            validate_manifest(
                payload, path, dataset, variant, variant,
                require_ablation_phase=(
                    dataset == "Medium-LI" and variant != "dual_view"))
            result = compact_result(payload, path)
            result["dataset"] = dataset
            result["variant"] = variant
            core.append(result)
            if variant == "dual_view":
                final_payloads[dataset] = payload

    relation = []
    sensitivity = []
    for dataset in DATASETS:
        for variant, history_k, use_relations, target in (
            ("dual_view_no_relation", 4, False, relation),
            ("dual_view_k2", 2, True, sensitivity),
        ):
            path = followup_path(dataset, variant, ablation_root)
            payload = load_manifest(path, missing, allow_incomplete)
            if payload is None:
                continue
            validate_manifest(
                payload, path, dataset, variant, "dual_view",
                history_k=history_k,
                use_relation_types=use_relations,
                require_ablation_phase=True,
            )
            result = compact_result(payload, path)
            result["dataset"] = dataset
            result["variant"] = variant
            target.append(result)

    mechanisms = []
    for dataset, payload in final_payloads.items():
        test = payload["best_event"]["test"]
        values = {
            condition: float(test[condition]["f1"])
            for condition in ("normal", "shuffled", "off")
        }
        if abs(values["normal"] - float(
            payload["val_selected_test_f1"])) > 1e-12:
            raise ValueError(
                f"{dataset}: manifest and normal intervention F1 differ")
        margin = values["normal"] - max(values["shuffled"], values["off"])
        mechanisms.append({
            "dataset": dataset,
            **values,
            "normal_margin_vs_strongest_control": margin,
            "supports_aligned_event_context": margin >= SAMPLING_BAND,
        })

    for row in relation:
        final = next((item for item in core if (
            item["dataset"] == row["dataset"]
            and item["variant"] == "dual_view")), None)
        row["delta_full_minus_no_relation"] = (
            final["val_selected_test_f1"] - row["val_selected_test_f1"]
            if final else None)
    for row in sensitivity:
        final = next((item for item in core if (
            item["dataset"] == row["dataset"]
            and item["variant"] == "dual_view")), None)
        row["delta_k2_minus_k4"] = (
            row["val_selected_test_f1"] - final["val_selected_test_f1"]
            if final else None)

    core_means = {}
    for variant in CORE_VARIANTS:
        values = [row["val_selected_test_f1"] for row in core
                  if row["variant"] == variant]
        core_means[variant] = statistics.fmean(values) if values else None
    return {
        "sampling_protocol": "dynamic_random",
        "selection_metric": "validation F1",
        "core_ablation": core,
        "core_val_selected_means": core_means,
        "mechanism_controls": mechanisms,
        "mechanism_support_count": sum(
            row["supports_aligned_event_context"] for row in mechanisms),
        "no_relation": relation,
        "history_k2": sensitivity,
        "missing_manifests": sorted(set(missing)),
        "complete": not missing,
    }


def markdown(summary):
    lines = [
        "# CDVT Ablation and Mechanism Summary",
        "",
        "## Core architecture ablation",
        "",
        "| Dataset | Variant | Val-selected test F1 | Parameters | "
        "Peak GPU memory (bytes) | Three-condition eval seconds / batch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["core_ablation"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} "
            f"| {row['val_selected_test_f1']:.5f} "
            f"| {row['parameter_count']} "
            f"| {row['peak_gpu_memory_bytes']} "
            f"| {row['test_seconds_per_batch']:.5f} |")
    lines.extend([
        "",
        "## Event alignment interventions",
        "",
        "| Dataset | Normal | Shuffled | Off | Normal margin | Supports mechanism |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in summary["mechanism_controls"]:
        lines.append(
            f"| {row['dataset']} | {row['normal']:.5f} "
            f"| {row['shuffled']:.5f} | {row['off']:.5f} "
            f"| {row['normal_margin_vs_strongest_control']:+.5f} "
            f"| {row['supports_aligned_event_context']} |")
    lines.extend([
        "",
        "## Relation-type ablation",
        "",
        "| Dataset | No-relation F1 | Full minus no-relation |",
        "|---|---:|---:|",
    ])
    for row in summary["no_relation"]:
        delta = row["delta_full_minus_no_relation"]
        lines.append(
            f"| {row['dataset']} | {row['val_selected_test_f1']:.5f} "
            f"| {delta:+.5f} |" if delta is not None else
            f"| {row['dataset']} | {row['val_selected_test_f1']:.5f} | TBD |")
    lines.extend([
        "",
        "## History-size sensitivity",
        "",
        "| Dataset | K=2 F1 | K=2 minus K=4 |",
        "|---|---:|---:|",
    ])
    for row in summary["history_k2"]:
        delta = row["delta_k2_minus_k4"]
        lines.append(
            f"| {row['dataset']} | {row['val_selected_test_f1']:.5f} "
            f"| {delta:+.5f} |" if delta is not None else
            f"| {row['dataset']} | {row['val_selected_test_f1']:.5f} | TBD |")
    if summary["missing_manifests"]:
        lines.extend([
            "",
            "Incomplete manifests:",
            *[f"- `{path}`" for path in summary["missing_manifests"]],
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = build_summary(
        args.phase1_root,
        args.phase2_root,
        args.ablation_root,
        allow_incomplete=args.allow_incomplete,
    )
    if args.write:
        args.ablation_root.mkdir(parents=True, exist_ok=True)
        (args.ablation_root / "ablation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.ablation_root / "ablation_summary.md").write_text(
            markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
