import json
import unittest
from pathlib import Path

from run.nested_dynamic_stability_audit import expand_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]


class NestedDynamicStabilityTest(unittest.TestCase):
    def test_task_expansion_is_balanced_and_unique(self):
        spec = json.loads(
            (REPO_ROOT / "run" / "nested_dynamic_stability_spec.json")
            .read_text()
        )
        tasks = expand_tasks(spec)
        self.assertEqual(len(tasks), 36)
        keys = {
            (
                task["dataset"],
                task["model_seed"],
                task["audit_seed"],
            )
            for task in tasks
        }
        self.assertEqual(len(keys), 36)
        for dataset in spec["datasets"]:
            selected = [
                task for task in tasks
                if task["dataset"] == dataset["dataset"]
            ]
            self.assertEqual(len(selected), 6)
            self.assertEqual(
                {task["model_seed"] for task in selected},
                {42, 43, 44},
            )
            self.assertEqual(
                {task["audit_seed"] for task in selected},
                {93001, 93002},
            )

    def test_all_checkpoints_are_epoch_499(self):
        spec = json.loads(
            (REPO_ROOT / "run" / "nested_dynamic_stability_spec.json")
            .read_text()
        )
        for task in expand_tasks(spec):
            self.assertEqual(Path(task["checkpoint"]).stem, "499")
            self.assertEqual(task["repeats"], 4)


if __name__ == "__main__":
    unittest.main()
