import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DynamicRandomProtocolTest(unittest.TestCase):
    def test_sampler_is_original_dynamic_path(self):
        source = (ROOT / "fraudGT" / "sampler" / "custom_sampler.py").read_text()
        self.assertIn("shuffle=shuffle", source)
        self.assertNotIn("_fixed_target_panel", source)
        self.assertNotIn("reset_generator", source)
        self.assertNotIn("generator=reset_generator", source)

    def test_initial_a2_has_all_six_datasets(self):
        baseline = json.loads((
            ROOT / "run" / "dynamic_random_a2_baseline.json").read_text())
        self.assertEqual(baseline["sampling_protocol"], "dynamic_random")
        self.assertEqual(len(baseline["datasets"]), 6)
        for metrics in baseline["datasets"].values():
            self.assertIn("val_selected_test_f1", metrics)
            self.assertIn("raw_best_test_f1", metrics)

    def test_pair_queue_uses_spec_config_template(self):
        source = (ROOT / "run" / "dynamic_random_pair_queue.py").read_text()
        self.assertIn('CONFIG_TEMPLATE.format(dataset=dataset)', source)
        self.assertIn('"config": config_path', source)
        self.assertNotIn(
            'f"configs/evidence_gate_v4/AML-{dataset}.yaml"', source)


if __name__ == "__main__":
    unittest.main()
