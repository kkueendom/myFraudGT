import unittest
from pathlib import Path

import torch

from run.cdvt_phase1_screen import bernoulli_js, common_positions, evaluate


class CDVTPhase1ScreenTest(unittest.TestCase):
    def test_validation_skips_counterfactuals_but_test_keeps_them(self):
        class Batch:
            def to(self, _device):
                return self

        class Model:
            def __init__(self):
                self.normal_calls = 0
                self.counterfactual_calls = 0

            def eval(self):
                return self

            def _output(self, offset=0.0):
                diagnostics = {
                    "target_edge_ids": torch.tensor([3]),
                    "event_count": torch.tensor([2]),
                    "fusion_gain_norm": torch.tensor([0.25]),
                }
                return (
                    torch.tensor([offset]), torch.tensor([1]), diagnostics,
                )

            def forward_details(self, _batch, condition):
                self.normal_calls += 1
                self.assert_normal(condition)
                return self._output()

            def forward_counterfactuals(self, _batch):
                self.counterfactual_calls += 1
                return {
                    "normal": self._output(),
                    "shuffled": self._output(0.5),
                    "off": self._output(-0.5),
                }

            @staticmethod
            def assert_normal(condition):
                if condition != "normal":
                    raise AssertionError(condition)

        model = Model()
        validation = evaluate(
            model, [Batch()], torch.device("cpu"), "val", ("normal",))
        self.assertEqual(set(validation["scores"]), {"normal"})
        self.assertEqual(model.normal_calls, 1)
        self.assertEqual(model.counterfactual_calls, 0)

        test = evaluate(model, [Batch()], torch.device("cpu"), "test")
        self.assertEqual(
            set(test["scores"]), {"normal", "shuffled", "off"})
        self.assertEqual(model.normal_calls, 1)
        self.assertEqual(model.counterfactual_calls, 1)

    def test_common_positions_align_independently_ordered_views(self):
        first = torch.tensor([7, 2, 9, 4])
        second = torch.tensor([4, 7, 3, 2])
        first_positions, second_positions = common_positions(first, second)
        self.assertTrue(torch.equal(
            first[first_positions], second[second_positions]))
        self.assertEqual(set(first[first_positions].tolist()), {2, 4, 7})

    def test_common_positions_accepts_an_empty_view(self):
        nonempty = torch.tensor([2, 7])
        empty = torch.empty(0, dtype=torch.long)
        for first, second in ((nonempty, empty), (empty, nonempty)):
            first_positions, second_positions = common_positions(first, second)
            self.assertEqual(first_positions.numel(), 0)
            self.assertEqual(second_positions.numel(), 0)

    def test_js_is_symmetric_zero_for_equal_and_positive_otherwise(self):
        first = torch.tensor([-2.0, 0.0, 2.0])
        same = bernoulli_js(first, first)
        second = torch.tensor([2.0, 0.0, -2.0])
        forward = bernoulli_js(first, second)
        reverse = bernoulli_js(second, first)
        self.assertAlmostEqual(float(same), 0.0, places=7)
        self.assertGreater(float(forward), 0.0)
        self.assertAlmostEqual(float(forward), float(reverse), places=7)

    def test_runner_does_not_create_or_restore_rng_state(self):
        source = Path("run/cdvt_phase1_screen.py").read_text()
        forbidden = (
            "torch." + "Generator(",
            "get_" + "rng_state(",
            "set_" + "rng_state(",
            "fixed_target_panel" + "=True",
        )
        self.assertFalse(any(token in source for token in forbidden))
        self.assertIn("shuffle=True", source)
        self.assertIn('"sampling_protocol": "dynamic_random"', source)

    def test_runner_limits_threads_and_writes_epoch_progress(self):
        runner = Path("run/cdvt_phase1_screen.py").read_text()
        launcher = Path("run/cdvt_phase1_6gpu.sh").read_text()
        self.assertIn("torch.set_num_threads(int(cfg.num_threads))", runner)
        self.assertIn("torch.set_num_interop_threads(1)", runner)
        self.assertIn('args.output_dir / "progress.json"', runner)
        self.assertIn("progress_tmp.replace(progress_path)", runner)
        for variable in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(launcher.count(variable), 2)

    def test_runner_records_explicit_experiment_phase(self):
        source = Path("run/cdvt_phase1_screen.py").read_text()
        self.assertIn('"phase": args.phase', source)
        self.assertIn('default="CDVT_phase1"', source)

    def test_formal_launcher_has_eight_seed42_tasks_on_seven_gpus(self):
        source = Path("run/cdvt_phase1_6gpu.sh").read_text()
        launch_rows = [
            line for line in source.splitlines()
            if line.startswith("launch ")
        ]
        self.assertEqual(len(launch_rows), 6)
        self.assertEqual(len(set(launch_rows)), 6)
        self.assertIn('"tasks":8', source)
        self.assertEqual(source.count("run_account_task "), 2)
        for gpu in range(7):
            self.assertIn(f"CUDA_VISIBLE_DEVICES={gpu}", source.replace(
                'CUDA_VISIBLE_DEVICES="$gpu"',
                " ".join(f"CUDA_VISIBLE_DEVICES={i}" for i in range(6)),
            ))
        self.assertIn("--max-epochs 500", source)
        self.assertNotIn("seed44", source)


if __name__ == "__main__":
    unittest.main()
