import runpy
import unittest
from pathlib import Path


class CostarQueueTest(unittest.TestCase):
    def test_pair500_queue_is_revision_locked_and_never_uses_gpu0(self):
        repo = Path(__file__).resolve().parents[1]
        path = repo / 'run/costar_formal_queue.py'
        queue = runpy.run_path(str(path))
        self.assertEqual(queue['MAX_EPOCH'], 500)
        self.assertEqual(queue['DONE_EPOCH'], 499)
        self.assertEqual(
            queue['EXPECTED_BRANCH'],
            'feature/costar-orthogonal-f1-router')
        self.assertEqual(
            queue['TASKS'], [('Large-LI', 44), ('Small-LI', 42)])
        self.assertNotIn(0, queue['GPU_ALLOWLIST'])

        source = path.read_text()
        self.assertIn('costar_formal500.lock.json', source)
        self.assertIn('configs/costar/AML-{dataset}.yaml', source)
        self.assertIn('"model.edge_decoding", "costar"', source)
        self.assertIn('"optim.max_epoch", str(MAX_EPOCH)', source)
        self.assertIn('"train.auto_resume", "False"', source)


if __name__ == '__main__':
    unittest.main()
