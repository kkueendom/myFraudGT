import ast
import unittest
from pathlib import Path

import torch

import run.export_figure11_event_stats as exporter


class Figure11EventStatsExporterTest(unittest.TestCase):
    def test_histogram_contains_every_value_through_cap(self):
        rows = exporter.histogram_rows(torch.tensor([1, 2, 2, 48]))
        self.assertEqual(len(rows), 48)
        self.assertEqual([row["event_count"] for row in rows],
                         list(range(1, 49)))
        self.assertEqual(rows[0]["target_count"], 1)
        self.assertEqual(rows[1]["target_count"], 2)
        self.assertEqual(rows[-1]["target_count"], 1)
        self.assertAlmostEqual(
            sum(row["target_share"] for row in rows), 1.0)

    def test_relation_rows_use_implementation_id_mapping(self):
        rows = exporter.relation_rows(torch.tensor([2, 1, 1, 2]))
        self.assertEqual(
            [row["relation_name"] for row in rows],
            ["O→O", "I→I", "O→I", "I→O"],
        )
        self.assertEqual(
            sum(row["edge_occurrence_count"] for row in rows), 6)
        self.assertAlmostEqual(
            sum(row["edge_occurrence_share"] for row in rows), 1.0)

    def test_statistics_validation_covers_required_totals(self):
        target_ids = torch.tensor([10, 20, 30, 40])
        event_counts = torch.tensor([1, 2, 2, 48])
        histogram = exporter.histogram_rows(event_counts)
        relations = exporter.relation_rows(torch.tensor([2, 1, 1, 2]))
        checks, contexts, at_cap, total_edges = (
            exporter.validate_statistics(
                target_ids,
                event_counts,
                histogram,
                relations,
                target_identity_checks=4,
                relation_range_checks=6,
            )
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(contexts.tolist(), [0, 1, 1, 47])
        self.assertEqual(at_cap, 1)
        self.assertEqual(total_edges, 6)

    def test_exporter_does_not_access_label_attribute(self):
        source = Path(exporter.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        accessed_attributes = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("y", accessed_attributes)


if __name__ == "__main__":
    unittest.main()
