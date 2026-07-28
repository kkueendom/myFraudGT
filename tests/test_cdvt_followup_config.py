import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the training environment has PyYAML
    yaml = None

if yaml is not None:
    from run.cdvt_materialize_config import materialize


@unittest.skipIf(yaml is None, "PyYAML is unavailable")
class CDVTFollowupConfigTest(unittest.TestCase):
    def write_base(self, root):
        path = root / "base.yaml"
        path.write_text(yaml.safe_dump({
            "seed": 42,
            "model": {"type": "CDVTModel"},
            "cdvt": {
                "variant": "dual_view",
                "lambda_cons": 0.0,
                "history_k": 4,
            },
        }, sort_keys=False))
        return path

    def test_materializes_seed_and_no_relation_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "no_relation.yaml"
            payload = materialize(
                self.write_base(root), output, 44,
                "dual_view_no_relation")
            self.assertEqual(payload["seed"], 44)
            self.assertEqual(payload["model"]["type"], "CDVTModel")
            self.assertEqual(payload["cdvt"]["variant"], "dual_view")
            self.assertFalse(payload["cdvt"]["use_relation_types"])
            self.assertEqual(payload["cdvt"]["history_k"], 4)

    def test_k2_and_account_only_configs_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_base(root)
            k2 = materialize(base, root / "k2.yaml", 42, "dual_view_k2")
            account = materialize(
                base, root / "account.yaml", 42, "account_only")
            self.assertEqual(k2["cdvt"]["history_k"], 2)
            self.assertTrue(k2["cdvt"]["use_relation_types"])
            self.assertEqual(account["model"]["type"], "GTModel")
            self.assertEqual(account["cdvt"]["variant"], "account_only")

    def test_refuses_to_overwrite_a_materialized_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_base(root)
            output = root / "config.yaml"
            materialize(base, output, 43, "dual_view")
            with self.assertRaises(FileExistsError):
                materialize(base, output, 44, "dual_view")


if __name__ == "__main__":
    unittest.main()
