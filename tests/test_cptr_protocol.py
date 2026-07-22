import json
import unittest
from pathlib import Path


class CPTRProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]

    def test_pair_spec_is_dynamic_random_and_commit_lockable(self):
        spec = json.loads(
            (self.repo / 'run/dynamic_random_pair_spec.json').read_text())
        self.assertEqual(spec['sampling_protocol'], 'dynamic_random')
        self.assertEqual(spec['method_tag'], 'CPTR')
        self.assertEqual(spec['variant'], 'CPTR')
        self.assertEqual(
            spec['expected_branch'], 'feature/cptr-threshold-transfer')
        self.assertEqual(
            spec['config_template'], 'configs/cptr/AML-{dataset}.yaml')
        self.assertEqual(
            spec['tasks'], [['Small-LI', 42], ['Large-LI', 44]])
        self.assertEqual(
            spec['overrides'], ['val.fixed_target_panel', 'False'])

    def test_configs_preserve_sampler_and_use_cptr(self):
        for dataset in ('Small-LI', 'Large-LI'):
            source = (
                self.repo / f'configs/cptr/AML-{dataset}.yaml').read_text()
            self.assertIn('sampler: link_neighbor', source)
            self.assertIn('fixed_target_panel: False', source)
            self.assertIn('edge_decoding: cptr', source)
            self.assertIn('max_epoch: 500', source)
            self.assertNotIn('fixed_panel_seed', source)
            self.assertNotIn('generator', source)
            self.assertNotIn('restore', source)

    def test_initial_a2_table_remains_the_only_comparison_table(self):
        baseline = json.loads(
            (self.repo / 'run/dynamic_random_a2_baseline.json').read_text())
        self.assertEqual(baseline['sampling_protocol'], 'dynamic_random')
        self.assertEqual(baseline['source'], 'initial_A2')
        self.assertEqual(len(baseline['datasets']), 6)


if __name__ == '__main__':
    unittest.main()
