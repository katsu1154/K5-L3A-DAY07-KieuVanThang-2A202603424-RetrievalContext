import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from starter.openrouter_controller import OpenRouterController


class FakeCompletions:
    def __init__(self, tool_calls, usage=None):
        self.tool_calls = tool_calls
        self.usage = usage
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(tool_calls=self.tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)


class OpenRouterControllerTests(unittest.TestCase):
    def controller_for(self, name, arguments, usage=None):
        completions = FakeCompletions(
            [
                SimpleNamespace(
                    function=SimpleNamespace(name=name, arguments=arguments)
                )
            ],
            usage=usage,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions,
            )
        )
        return OpenRouterController(client=client), completions

    def test_search_call_is_normalized_and_uses_openrouter_model(self):
        controller, completions = self.controller_for(
            "search", '{"query":"Leonardo"}'
        )

        action = controller.decide(
            "Who was Leonardo?",
            [{"queries": ["artist"], "context": "[chunk-1] Leonardo was an artist."}],
        )

        self.assertEqual(action, {"action": "search", "query": "Leonardo"})
        self.assertIn("Leonardo was an artist.", completions.request["messages"][1]["content"])
        self.assertEqual(completions.request["model"], "google/gemini-4-hydrogen")
        self.assertEqual(completions.request["timeout"], 90.0)
        self.assertEqual(completions.request["tool_choice"], "auto")
        self.assertEqual(controller.api_call_count, 1)

    def test_records_provider_usage_when_available(self):
        usage = SimpleNamespace(
            prompt_tokens=41,
            completion_tokens=7,
            total_tokens=48,
            cost=0.0003,
        )
        controller, _ = self.controller_for("answer", '{"answer":"Yes","citations":["c"]}', usage)
        controller.decide("Question?", [])
        self.assertEqual(
            controller.usage_records,
            [{
                "model": "google/gemini-4-hydrogen",
                "prompt_tokens": 41,
                "completion_tokens": 7,
                "total_tokens": 48,
                "reported_cost_usd": 0.0003,
            }],
        )

    def test_answer_call_preserves_citations(self):
        controller, _ = self.controller_for(
            "answer",
            '{"answer":"He was an artist.","citations":["chunk-1"]}',
        )

        self.assertEqual(
            controller.decide("Who was Leonardo?", []),
            {
                "action": "answer",
                "answer": "He was an artist. [chunk-1]",
                "citations": ["chunk-1"],
            },
        )

    def test_rejects_missing_api_key_and_malformed_tool_calls(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                OpenRouterController()

        controller, _ = self.controller_for("search", '{"query":" "}')
        with self.assertRaisesRegex(ValueError, "non-empty query"):
            controller.decide("Question?", [])

        controller, _ = self.controller_for("search", "not-json")
        with self.assertRaisesRegex(ValueError, "invalid function-call JSON"):
            controller.decide("Question?", [])


if __name__ == "__main__":
    unittest.main()
