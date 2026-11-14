"""Run one question through OpenRouter and the Day 6 Chroma retriever."""
import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

from starter.agentic import run_agent
from starter.day6_retriever import Day6ChromaRetriever
from starter.openrouter_controller import OpenRouterController


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    parser = argparse.ArgumentParser(description="Run the Day 7 OpenRouter + Day 6 RAG agent")
    parser.add_argument("question", help="Question to answer from the Day 6 corpus")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()

    print("[Day 6] Opening vector index...", file=sys.stderr, flush=True)
    retriever = Day6ChromaRetriever()
    print("[Day 6] Vector index ready.", file=sys.stderr, flush=True)
    controller = OpenRouterController()
    result = run_agent(
        args.question,
        controller,
        retriever,
        top_k=args.top_k,
        max_searches=args.max_searches,
        budget=args.budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
