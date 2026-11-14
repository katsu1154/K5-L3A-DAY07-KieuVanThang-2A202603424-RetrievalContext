import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.run_agent_benchmark import summarize_traces


class AgentBenchmarkTests(unittest.TestCase):
    def test_summary_keeps_prior_failed_attempts_and_marks_cost_incomplete(self):
        config = {"split": "test", "method": "hybrid_rrf"}
        traces = [
            {
                "status": "abstain",
                "search_count": 2,
                "duration_ms": 5000,
                "usage": {
                    "api_calls": 3,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "reported_cost_usd": 0.001,
                },
                "previous_attempts": [
                    {
                        "error": {"type": "RuntimeError"},
                        "duration_ms": 3000,
                        "usage": {
                            "api_calls": 2,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "reported_cost_usd": None,
                        },
                    }
                ],
            }
        ]
        summary = summarize_traces(config, traces)
        self.assertEqual(summary["api_calls"], 5)
        self.assertEqual(summary["failed_attempt_count"], 1)
        self.assertEqual(summary["known_reported_cost_usd"], 0.001)
        self.assertIsNone(summary["reported_cost_usd"])
        self.assertIsNone(summary["total_tokens"])


if __name__ == "__main__":
    unittest.main()
