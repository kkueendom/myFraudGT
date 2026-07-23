import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'tier_phase0_coverage_audit',
    ROOT / 'run' / 'tier_phase0_coverage_audit.py',
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TierPhase0AuditTest(unittest.TestCase):
    def test_tensor_distribution(self):
        result = AUDIT.tensor_distribution(torch.tensor([0, 1, 3, 4]))
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 2.0)
        self.assertEqual(result['median'], 2.0)
        self.assertEqual(result['max'], 4.0)

    def test_coverage_gate(self):
        gate = {
            'min_overall_coverage': 0.25,
            'min_positive_coverage': 0.10,
            'min_positive_samples': 10,
        }
        row = {
            'class_counts': {'0': 90, '1': 10},
            'coverage_rate': 0.5,
            'class_coverage_rate': {'0': 0.5, '1': 0.2},
        }
        self.assertEqual(AUDIT.coverage_decision(row, gate), 'pass')
        row['class_coverage_rate']['1'] = 0.05
        self.assertEqual(
            AUDIT.coverage_decision(row, gate),
            'fail_positive_coverage',
        )
        row['class_counts']['1'] = 4
        self.assertEqual(
            AUDIT.coverage_decision(row, gate),
            'inconclusive_too_few_positive_samples',
        )


if __name__ == '__main__':
    unittest.main()
