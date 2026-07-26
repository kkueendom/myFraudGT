import unittest

from run.summarize_tier_dynamic_reliability import (
    classify_mechanism,
)


class SummarizeTierDynamicReliabilityTest(unittest.TestCase):
    def classify(
        self,
        shuffle,
        off,
        delta,
        corrected_gt,
        corrected_le,
        changed,
    ):
        return classify_mechanism(
            shuffle,
            off,
            delta,
            corrected_gt,
            corrected_le,
            changed,
            16,
        )

    def test_classifies_useful_aligned_evidence(self):
        self.assertEqual(
            self.classify(0.02, 0.03, 0.01, 13, 3, 60),
            "useful_aligned_evidence",
        )

    def test_classifies_sensitive_but_harmful(self):
        self.assertEqual(
            self.classify(0.02, 0.03, -0.01, 4, 12, 80),
            "sensitive_but_harmful",
        )

    def test_classifies_used_but_unaligned(self):
        self.assertEqual(
            self.classify(0.005, 0.02, 0.01, 10, 6, 80),
            "used_but_unaligned",
        )

    def test_classifies_inactive_evidence(self):
        self.assertEqual(
            self.classify(0.005, 0.004, 0.01, 10, 6, 80),
            "inactive_evidence",
        )


if __name__ == "__main__":
    unittest.main()

