import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run.cdvt_additive_summary import build_summary as build_additive
from run.cdvt_ablation_summary import build_summary as build_ablation
from run.cdvt_phase3_gate import require_phase2_gate
from run.cdvt_phase3_summary import (
    build_summary as build_phase3,
    markdown as phase3_markdown,
)
from run.cdvt_protocol import INITIAL_A2, PE_FRAUDGT_PAPER
from run.cdvt_runtime_summary import build_summary as build_runtime


ALL_DATASETS = (
    "Small-LI", "Small-HI", "Medium-LI",
    "Medium-HI", "Large-LI", "Large-HI",
)


def config_snapshot(history_k=4, use_relation_types=True):
    return {
        "dataset": {"tier_evidence": True},
        "train": {
            "sampler": "link_neighbor",
            "iter_per_epoch": 256,
            "batch_size": 2048,
            "eval_period": 4,
        },
        "val": {
            "iter_per_epoch": 256,
            "fixed_target_panel": False,
        },
        "cdvt": {
            "history_k": history_k,
            "use_relation_types": use_relation_types,
        },
    }


def write_manifest(
    root, dataset, variant, seed, val_f1, raw_f1=None,
    architecture=None, phase="CDVT_phase1", history_k=4,
    use_relation_types=True, interventions=None
):
    architecture = architecture or variant
    raw_f1 = val_f1 + 0.01 if raw_f1 is None else raw_f1
    task = root / f"{dataset}_{variant}_seed{seed}"
    task.mkdir(parents=True)
    interventions = interventions or {
        "normal": val_f1,
        "shuffled": val_f1 - 0.01,
        "off": val_f1 - 0.02,
    }
    payload = {
        "phase": phase,
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": architecture,
        "lambda_cons": 0.0,
        "seed": seed,
        "git_commit": "abc12345",
        "config": str(task / "config.yaml"),
        "checkpoint": str(task / "best_val.ckpt"),
        "val_selected_epoch": 80,
        "val_selected_test_f1": val_f1,
        "raw_best_epoch": 84,
        "raw_best_test_f1": raw_f1,
        "history_k": history_k,
        "use_relation_types": use_relation_types,
        "parameter_count": 1000,
        "peak_gpu_memory_bytes": 2000,
        "elapsed_seconds": 30.0,
        "best_event": {
            "test": {
                name: {"f1": score}
                for name, score in interventions.items()
            },
            "test_seconds": 8.0,
            "test_steps": 4,
        },
        "config_snapshot": config_snapshot(
            history_k=history_k,
            use_relation_types=use_relation_types,
        ),
    }
    (task / "manifest.json").write_text(json.dumps(payload))
    return payload


def write_benchmark(root, dataset, variant, seconds_per_batch):
    task = root / f"{dataset}_{variant}_seed42"
    task.mkdir(parents=True)
    payload = {
        "phase": "CDVT_runtime",
        "sampling_protocol": "dynamic_random",
        "dataset": dataset,
        "variant": variant,
        "architecture_variant": variant,
        "seed": 42,
        "steps": 256,
        "targets": 4096,
        "elapsed_seconds": 256 * seconds_per_batch,
        "seconds_per_batch": seconds_per_batch,
        "seconds_per_target": seconds_per_batch / 16,
        "parameter_count": 2000 if variant == "dual_view" else 1000,
        "peak_gpu_memory_bytes": (
            4000 if variant == "dual_view" else 2000),
    }
    (task / "benchmark.json").write_text(json.dumps(payload))


class CDVTFollowupTest(unittest.TestCase):
    def populate_phase2(self, phase1, phase2, deltas):
        for dataset, delta in zip(ALL_DATASETS, deltas):
            root = phase1 if dataset in {"Small-LI", "Large-LI"} else phase2
            a2 = INITIAL_A2[dataset]
            write_manifest(
                root,
                dataset,
                "dual_view",
                42,
                PE_FRAUDGT_PAPER[dataset] + delta,
                a2["raw_best_test_f1"] + delta,
                phase=(
                    "CDVT_phase1" if root == phase1 else "CDVT_phase2"),
            )

    def test_phase2_gate_requires_four_wins_and_positive_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1, phase2 = root / "phase1", root / "phase2"
            phase1.mkdir()
            phase2.mkdir()
            self.populate_phase2(
                phase1, phase2, (0.01, 0.01, 0.01, 0.01, -0.002, -0.002))
            summary = require_phase2_gate(phase1, phase2)
            self.assertTrue(summary["phase2_gate"]["advance_to_phase3"])

    def test_importing_cli_scripts_does_not_depend_on_working_directory(self):
        repository = Path(__file__).resolve().parents[1]
        scripts = (
            repository / "run" / "cdvt_phase3_gate.py",
            repository / "run" / "cdvt_phase3_summary.py",
            repository / "run" / "cdvt_additive_summary.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                completed = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=directory,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"{script.name}: {completed.stderr}",
                )

    def test_phase3_summary_uses_real_seed_rows_and_sample_std(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1, phase2, phase3, ablation = (
                root / "phase1",
                root / "phase2",
                root / "phase3",
                root / "ablation",
            )
            for path in (phase1, phase2, phase3, ablation):
                path.mkdir()
            for dataset in ("Small-LI", "Medium-LI", "Large-LI"):
                dual_seed42_root = (
                    phase1 if dataset in {"Small-LI", "Large-LI"}
                    else phase2)
                account_seed42_root = (
                    phase1 if dataset in {"Small-LI", "Large-LI"}
                    else ablation)
                for seed, account_f1 in (
                    (42, 0.50), (43, 0.51), (44, 0.52)
                ):
                    account_target = (
                        account_seed42_root if seed == 42 else phase3)
                    dual_target = (
                        dual_seed42_root if seed == 42 else phase3)
                    seed42_phase = (
                        "CDVT_phase1"
                        if dataset in {"Small-LI", "Large-LI"}
                        else "CDVT_phase2"
                    )
                    account_seed42_phase = (
                        "CDVT_phase1"
                        if dataset in {"Small-LI", "Large-LI"}
                        else "CDVT_ablation"
                    )
                    write_manifest(
                        account_target,
                        dataset,
                        "account_only",
                        seed,
                        account_f1,
                        phase=(
                            "CDVT_phase3"
                            if seed != 42
                            else account_seed42_phase
                        ),
                    )
                    write_manifest(
                        dual_target,
                        dataset,
                        "dual_view",
                        seed,
                        account_f1 + 0.02,
                        phase=(
                            "CDVT_phase3" if seed != 42
                            else seed42_phase),
                    )
            summary = build_phase3(
                phase1, phase2, phase3, ablation)
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["rows"]), 18)
            self.assertEqual(len(summary["paired_rows"]), 9)
            for group in summary["datasets"]:
                self.assertEqual(group["seeds"], [42, 43, 44])
                self.assertTrue(math.isclose(
                    group["fraudgt"]["val_selected"]["sample_std"],
                    0.01,
                ))
                self.assertTrue(math.isclose(
                    group["cdvt"]["val_selected"]["sample_std"],
                    0.01,
                ))
                self.assertTrue(math.isclose(
                    group["paired_delta"]["val_selected"]["mean"],
                    0.02,
                ))
                self.assertTrue(math.isclose(
                    group["paired_delta"]["val_selected"]["sample_std"],
                    0.0,
                ))

    def test_incomplete_phase3_omits_unmatched_seed_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1, phase2, phase3, ablation = (
                root / "phase1",
                root / "phase2",
                root / "phase3",
                root / "ablation",
            )
            for path in (phase1, phase2, phase3, ablation):
                path.mkdir()
            write_manifest(
                phase1, "Small-LI", "account_only", 42, 0.40)
            write_manifest(
                phase1, "Small-LI", "dual_view", 42, 0.45)
            write_manifest(
                phase3, "Small-LI", "account_only", 43, 0.90,
                phase="CDVT_phase3")
            summary = build_phase3(
                phase1, phase2, phase3, ablation,
                allow_incomplete=True)
            small = summary["datasets"][0]
            self.assertEqual(small["seeds"], [42])
            self.assertEqual(
                small["available_seeds"]["account_only"], [42, 43])
            self.assertEqual(
                small["available_seeds"]["dual_view"], [42])
            self.assertEqual(
                small["fraudgt"]["val_selected"]["mean"], 0.40)
            self.assertNotIn(
                "## Three-seed aggregate", phase3_markdown(summary))

    def test_ablation_summary_reuses_final_and_extracts_interventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1, phase2, ablation = (
                root / "phase1", root / "phase2", root / "ablation")
            for path in (phase1, phase2, ablation):
                path.mkdir()
            for dataset in ("Small-LI", "Medium-LI", "Large-LI"):
                standard_root = (
                    phase1 if dataset in {"Small-LI", "Large-LI"}
                    else phase2)
                controls_root = (
                    phase1 if dataset in {"Small-LI", "Large-LI"}
                    else ablation)
                write_manifest(
                    standard_root, dataset, "dual_view", 42, 0.60,
                    interventions={
                        "normal": 0.60, "shuffled": 0.58, "off": 0.57},
                    phase=(
                        "CDVT_phase1" if standard_root == phase1
                        else "CDVT_phase2"),
                )
                for variant, f1 in (("account_only", 0.55),
                                    ("event_only", 0.54)):
                    write_manifest(
                        controls_root, dataset, variant, 42, f1,
                        phase=(
                            "CDVT_phase1" if controls_root == phase1
                            else "CDVT_ablation"),
                    )
                write_manifest(
                    ablation, dataset, "dual_view_no_relation", 42, 0.57,
                    architecture="dual_view", phase="CDVT_ablation",
                    use_relation_types=False)
                write_manifest(
                    ablation, dataset, "dual_view_k2", 42, 0.59,
                    architecture="dual_view", phase="CDVT_ablation",
                    history_k=2)
            summary = build_ablation(phase1, phase2, ablation)
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["core_ablation"]), 9)
            self.assertEqual(summary["mechanism_support_count"], 3)
            self.assertTrue(all(math.isclose(
                row["delta_full_minus_no_relation"], 0.03)
                for row in summary["no_relation"]))
            self.assertTrue(all(math.isclose(
                row["delta_k2_minus_k4"], -0.01)
                for row in summary["history_k2"]))

    def test_queues_are_gate_bound_and_do_not_repeat_final_seed42(self):
        phase3 = Path("run/cdvt_phase3_queue.sh").read_text()
        ablation = Path("run/cdvt_ablation_queue.sh").read_text()
        self.assertLess(
            phase3.index("cdvt_phase3_gate.py"), phase3.index("mkdir -p"))
        self.assertEqual(phase3.count("Small-LI:43:"), 2)
        self.assertEqual(phase3.count("Large-LI:44:"), 2)
        self.assertIn("Small-LI:43:account_only", phase3)
        self.assertIn("Large-LI:44:dual_view", phase3)
        self.assertNotIn("Small-LI:42", phase3)
        self.assertIn('"tasks":12', phase3)
        self.assertIn("CDVT_POLL_SECONDS:-300", phase3)
        self.assertLess(
            ablation.index("cdvt_phase3_gate.py"), ablation.index("mkdir -p"))
        self.assertIn("Medium-LI:account_only", ablation)
        self.assertIn("dual_view_no_relation", ablation)
        self.assertIn("dual_view_k2", ablation)
        self.assertNotIn("Small-LI:dual_view ", ablation)
        self.assertIn('"tasks":8', ablation)

    def test_runtime_summary_compares_normal_only_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("Small-LI", "Medium-LI", "Large-LI"):
                write_benchmark(root, dataset, "account_only", 0.5)
                write_benchmark(root, dataset, "dual_view", 1.0)
            summary = build_runtime(root)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["benchmark_mode"], "normal_only")
            self.assertEqual(len(summary["comparisons"]), 3)
            self.assertTrue(all(
                row["parameter_ratio_cdvt_vs_account"] == 2.0
                and row["memory_ratio_cdvt_vs_account"] == 2.0
                and row["latency_ratio_cdvt_vs_account"] == 2.0
                for row in summary["comparisons"]))

    def test_additive_summary_isolates_cross_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1, phase2, additive = (
                root / "phase1", root / "phase2", root / "additive")
            for path in (phase1, phase2, additive):
                path.mkdir()
            for dataset in ("Small-LI", "Medium-LI", "Large-LI"):
                final_root = (
                    phase1 if dataset in {"Small-LI", "Large-LI"}
                    else phase2)
                write_manifest(
                    final_root, dataset, "dual_view", 42, 0.60,
                    phase=(
                        "CDVT_phase1" if final_root == phase1
                        else "CDVT_phase2"),
                )
                write_manifest(
                    additive,
                    dataset,
                    "causal_event_add",
                    42,
                    0.58,
                    architecture="additive_view",
                    phase="CDVT_ablation",
                )
            summary = build_additive(phase1, phase2, additive)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["cross_attention_win_count"], 3)
            self.assertTrue(math.isclose(
                summary["mean_delta_cross_attention_minus_additive"],
                0.02,
            ))

    def test_additive_queue_waits_for_followup_and_has_three_tasks(self):
        source = Path("run/cdvt_additive_queue.sh").read_text()
        self.assertIn("CDVT_FOLLOWUP_ROOT", source)
        self.assertLess(
            source.index("queue_complete.json"),
            source.index("mkdir -p \"$root/configs\""),
        )
        self.assertIn("tasks=(Small-LI Medium-LI Large-LI)", source)
        self.assertIn('"tasks":3', source)
        self.assertIn("--variant additive_view", source)
        self.assertIn("--experiment-label causal_event_add", source)
        self.assertIn("CDVT_POLL_SECONDS:-300", source)

    def test_runtime_queue_requires_complete_ablation_and_uses_six_tasks(self):
        source = Path("run/cdvt_runtime_queue.sh").read_text()
        self.assertLess(
            source.index("cdvt_ablation_summary.py"),
            source.index("mkdir -p"),
        )
        self.assertIn('"mode":"normal_only"', source)
        self.assertIn('"tasks":6', source)
        self.assertIn("--max-batches 256", source)
        self.assertIn("CDVT_POLL_SECONDS:-300", source)

    def test_unified_followup_queue_owns_all_gpu_assignment(self):
        source = Path("run/cdvt_followup_queue.sh").read_text()
        task_rows = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("'phase3|", "'ablation|"))
        ]
        self.assertEqual(len(task_rows), 20)
        self.assertEqual(len(set(task_rows)), 20)
        self.assertIn('"training_tasks":20', source)
        self.assertIn('"phase3_tasks":12', source)
        self.assertIn('"runtime_tasks":6', source)
        self.assertIn("gpu_is_idle", source)
        self.assertIn("cdvt_runtime_queue.sh", source)
        self.assertNotIn("Small-LI|42|dual_view'", source)
        self.assertNotIn("Small-LI|42|account_only'", source)


if __name__ == "__main__":
    unittest.main()
