import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_meets_day7_split_and_evidence_requirements(self):
        questions = json.loads((ROOT / "data" / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 24)
        self.assertEqual(len({question["id"] for question in questions}), 24)

        counts = Counter(
            (question["split"], question.get("answerable", bool(question.get("gold"))))
            for question in questions
        )
        self.assertEqual(counts[("dev", True)], 15)
        self.assertEqual(counts[("dev", False)], 2)
        self.assertEqual(counts[("test", True)], 5)
        self.assertEqual(counts[("test", False)], 2)

        for question in questions:
            self.assertIn(question["split"], {"dev", "test"})
            if question.get("answerable") is False:
                self.assertEqual(question["expected_behavior"], "search_and_abstain")
                self.assertTrue(question["absence_reason"].strip())
            else:
                self.assertTrue(question["expected_answer"].strip())
                self.assertTrue(question["gold"])
                for evidence in question["gold"]:
                    self.assertTrue(evidence["evidence"].strip())
                    self.assertIsInstance(evidence["page"], int)


if __name__ == "__main__":
    unittest.main()
