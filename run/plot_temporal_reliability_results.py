#!/usr/bin/env python3
"""Create publication figures for the temporal reliability benchmark."""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATASETS = (
    "Small-LI",
    "Small-HI",
    "Medium-LI",
    "Medium-HI",
    "Large-LI",
    "Large-HI",
)
SEEDS = (42, 43, 44)
SEED_COLORS = {
    42: "#0F766E",
    43: "#DC6B45",
    44: "#4F46A5",
}
COMPONENT_COLORS = {
    "Model seed": "#334155",
    "Dynamic stream": "#0F766E",
    "Within-stream event": "#DC6B45",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#475569",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0F172A",
        "axes.labelcolor": "#0F172A",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_manifests(root):
    paths = sorted(glob.glob(
        str(root / "**" / "nested_stability_manifest.json"),
        recursive=True,
    ))
    manifests = [json.loads(Path(path).read_text()) for path in paths]
    if len(manifests) != 36:
        raise RuntimeError(
            f"expected 36 nested manifests, found {len(manifests)}")
    return paths, manifests


def save_figure(fig, output_dir, stem):
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 320
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_variance_components(summary, output_dir):
    labels = list(DATASETS)
    model = []
    stream = []
    event = []
    for dataset in labels:
        row = summary["datasets"][dataset]
        total = float(row["variance_total"])
        if total <= 0:
            raise ValueError(f"nonpositive variance total for {dataset}")
        model.append(float(row["variance_model_seed"]) / total)
        stream.append(float(row["variance_stream"]) / total)
        event.append(float(row["variance_event"]) / total)

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for name, values in (
        ("Model seed", model),
        ("Dynamic stream", stream),
        ("Within-stream event", event),
    ):
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=COMPONENT_COLORS[name],
            label=name,
        )
        left += np.asarray(values)
    for index, dataset in enumerate(labels):
        sampling = 100.0 * summary["datasets"][dataset]["sampling_share"]
        ax.text(
            1.012,
            index,
            f"{sampling:.1f}% sampling",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#334155",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of estimated F1 variance")
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.42, 1.01),
        ncol=3,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.tight_layout()
    return save_figure(
        fig, output_dir, "fig_variance_components")


def _events_by_dataset_and_seed(manifests):
    values = {
        dataset: {seed: [] for seed in SEEDS}
        for dataset in DATASETS
    }
    for manifest in manifests:
        dataset = manifest["dataset"]
        seed = int(manifest["model_seed"])
        for event in manifest["events"]:
            values[dataset][seed].append(float(event["test"]["f1"]))
    for dataset in DATASETS:
        for seed in SEEDS:
            if len(values[dataset][seed]) != 8:
                raise RuntimeError(
                    f"expected 8 events for {dataset} seed {seed}")
    return values


def plot_event_distributions(summary, manifests, output_dir):
    values = _events_by_dataset_and_seed(manifests)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.25, 4.65),
        sharex=True,
        constrained_layout=True,
    )
    rng = np.random.default_rng(20260726)
    for ax, dataset in zip(axes.flat, DATASETS):
        grouped = [values[dataset][seed] for seed in SEEDS]
        box = ax.boxplot(
            grouped,
            positions=(1, 2, 3),
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0F172A", "linewidth": 1.2},
            whiskerprops={"color": "#64748B", "linewidth": 0.8},
            capprops={"color": "#64748B", "linewidth": 0.8},
            boxprops={"edgecolor": "#64748B", "linewidth": 0.8},
        )
        for patch, seed in zip(box["boxes"], SEEDS):
            patch.set_facecolor(SEED_COLORS[seed])
            patch.set_alpha(0.2)
        for position, seed in enumerate(SEEDS, start=1):
            jitter = rng.uniform(-0.12, 0.12, len(values[dataset][seed]))
            ax.scatter(
                position + jitter,
                values[dataset][seed],
                s=14,
                color=SEED_COLORS[seed],
                edgecolors="white",
                linewidths=0.35,
                alpha=0.9,
                zorder=3,
            )
        baseline = float(
            summary["datasets"][dataset]["initial_a2_diagnostic"])
        ax.axhline(
            baseline,
            color="#B91C1C",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="Historical initial A2",
        )
        ax.set_title(dataset, loc="left", fontweight="semibold")
        ax.set_xticks((1, 2, 3))
        ax.set_xticklabels(("42", "43", "44"))
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Val-selected test F1")
    for ax in axes[1, :]:
        ax.set_xlabel("Model-training seed")
    axes[0, 0].legend(loc="lower left")
    return save_figure(
        fig, output_dir, "fig_dynamic_event_distributions")


def plot_reporting_diagnostics(summary, output_dir):
    inflation = [
        summary["datasets"][dataset]["median_raw_max_inflation"]
        for dataset in DATASETS
    ]
    threshold_sd = [
        summary["datasets"][dataset]["val_threshold_std"]
        for dataset in DATASETS
    ]
    x = np.arange(len(DATASETS))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    axes[0].bar(x, inflation, color="#DC6B45", width=0.66)
    axes[0].axhline(
        0.01,
        color="#334155",
        linestyle=(0, (4, 2)),
        linewidth=0.9,
    )
    axes[0].text(
        5.45,
        0.0115,
        "0.01 F1",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    axes[0].set_ylabel("Median raw-event max inflation")
    axes[0].set_title("Best-event reporting inflation", loc="left")

    axes[1].bar(x, threshold_sd, color="#0F766E", width=0.66)
    axes[1].set_ylabel("Validation threshold SD")
    axes[1].set_title("Threshold instability", loc="left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=35, ha="right")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    return save_figure(
        fig, output_dir, "fig_reporting_diagnostics")


def main():
    args = parse_args()
    configure_style()
    summary = json.loads(args.summary.read_text())
    manifest_paths, manifests = load_manifests(args.manifest_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_variance_components(summary, args.output_dir))
    outputs.extend(plot_event_distributions(
        summary, manifests, args.output_dir))
    outputs.extend(plot_reporting_diagnostics(summary, args.output_dir))
    source_manifest = {
        "summary": str(args.summary.resolve()),
        "summary_sha256": sha256(args.summary),
        "manifest_root": str(args.manifest_root.resolve()),
        "manifest_count": len(manifest_paths),
        "manifest_sha256": {
            str(Path(path).resolve()): sha256(Path(path))
            for path in manifest_paths
        },
        "outputs": [str(path.resolve()) for path in outputs],
    }
    source_path = args.output_dir / "figure_source_manifest.json"
    source_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "figures": len(outputs),
        "source_manifest": str(source_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
