import json
import unittest
from pathlib import Path

from run import prospective_pair_queue as queue


class ProspectivePairQueueTest(unittest.TestCase):
    def test_spec_is_fixed_to_two_li_datasets(self):
        spec = json.loads((
            Path(__file__).resolve().parents[1] /
            'run' / 'prospective_pair_spec.json').read_text())
        self.assertEqual(
            spec['tasks'], [['Small-LI', 42], ['Large-LI', 44]])
        self.assertIn('val.fixed_target_panel', spec['overrides'])

    def test_gpu_zero_is_never_eligible(self):
        self.assertNotIn(0, queue.GPU_ALLOWLIST)

    def test_output_name_contains_method_and_commit(self):
        stem = queue.run_stem('Small-LI', 42, 'deadbeef')
        self.assertEqual(
            stem,
            'AML-Small-LI-UPRCPair500-Seed42-deadbeef')


if __name__ == '__main__':
    unittest.main()
