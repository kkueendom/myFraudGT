#!/usr/bin/env python3
"""Materialize one immutable CDVT experiment config from a frozen base."""

import argparse
from pathlib import Path

import yaml


VARIANTS = {
    "account_only": {
        "model_type": "GTModel",
        "architecture": "account_only",
        "history_k": 4,
        "use_relation_types": True,
    },
    "event_only": {
        "model_type": "CDVTModel",
        "architecture": "event_only",
        "history_k": 4,
        "use_relation_types": True,
    },
    "dual_view": {
        "model_type": "CDVTModel",
        "architecture": "dual_view",
        "history_k": 4,
        "use_relation_types": True,
    },
    "dual_view_no_relation": {
        "model_type": "CDVTModel",
        "architecture": "dual_view",
        "history_k": 4,
        "use_relation_types": False,
    },
    "dual_view_k2": {
        "model_type": "CDVTModel",
        "architecture": "dual_view",
        "history_k": 2,
        "use_relation_types": True,
    },
}


def materialize(base_path, output_path, seed, variant):
    base_path = Path(base_path)
    output_path = Path(output_path)
    if variant not in VARIANTS:
        raise ValueError(f"unknown experiment variant: {variant}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite config: {output_path}")
    payload = yaml.safe_load(base_path.read_text())
    settings = VARIANTS[variant]
    payload["seed"] = int(seed)
    payload["model"]["type"] = settings["model_type"]
    payload["cdvt"]["variant"] = settings["architecture"]
    payload["cdvt"]["lambda_cons"] = 0.0
    payload["cdvt"]["history_k"] = settings["history_k"]
    payload["cdvt"]["use_relation_types"] = settings[
        "use_relation_types"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    materialize(args.base, args.output, args.seed, args.variant)


if __name__ == "__main__":
    main()
