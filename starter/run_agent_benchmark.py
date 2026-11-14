"""Run real OpenRouter agent evaluations for one benchmark split."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from dotenv import load_dotenv

from starter.agentic import run_agent
from starter.baseline import TfidfIndex, validate
from starter.day6_retriever import Day6ChromaRetriever
from starter.evaluate_retrieval import load_indexed_chunks, reciprocal_rank_fusion
from starter.openrouter_controller import OpenRouterController


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dense_day6", "tfidf", "hybrid_rrf")
Search = Callable[[str, dict[str, Any] | None, int], list[dict[str, Any]]]


class SearchAdapter:
    def __init__(self, search: Search) -> None:
        self._search = search

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return self._search(query, filters, top_k)


def build_searches(
    dense: Day6ChromaRetriever,
    lexical: TfidfIndex,
    chunk_count: int,
    *,
    candidate_k: int,
    rrf_k: int,
) -> dict[str, Search]:
    retrieval_depth = min(candidate_k, chunk_count)

    def dense_search(
        query: str, filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        return dense.search(query, filters=filters, top_k=top_k)

    def tfidf_search(
        query: str, filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        return lexical.search(query, top_k=top_k, filters=filters)

    def hybrid_search(
        query: str, filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        dense_results = dense.search(
            query, filters=filters, top_k=retrieval_depth
        )
        lexical_results = lexical.search(
            query, top_k=retrieval_depth, filters=filters
        )
        return reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            rrf_k=rrf_k,
            top_k=top_k,
        )

    return {
        "dense_day6": dense_search,
        "tfidf": tfidf_search,
        "hybrid_rrf": hybrid_search,
    }


def summarize_usage(
    controller: OpenRouterController,
    call_start: int,
    usage_start: int,
) -> dict[str, Any]:
    api_calls = controller.api_call_count - call_start
    records = controller.usage_records[usage_start:]

    def token_total(field: str) -> int | None:
        values = [record[field] for record in records]
        if not values or any(type(value) is not int for value in values):
            return None
        return sum(values)

    costs = [record["reported_cost_usd"] for record in records]
    cost_is_complete = (
        len(records) == api_calls
        and bool(records)
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in costs
        )
    )
    return {
        "model": controller.model,
        "api_calls": api_calls,
        "responses_with_usage": len(records),
        "prompt_tokens": token_total("prompt_tokens"),
        "completion_tokens": token_total("completion_tokens"),
        "total_tokens": token_total("total_tokens"),
        "reported_cost_usd": sum(costs) if cost_is_complete else None,
    }


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize_traces(config: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {
        status: sum(trace["status"] == status for trace in traces)
        for status in ("answered", "abstain", "error")
    }
    current_costs = [trace["usage"]["reported_cost_usd"] for trace in traces]
    previous_attempts = [
        attempt
        for trace in traces
        for attempt in trace.get("previous_attempts", [])
    ]
    previous_costs = [
        attempt.get("usage", {}).get("reported_cost_usd")
        for attempt in previous_attempts
    ]
    known_costs = [
        value
        for value in current_costs + previous_costs
        if isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    ]
    all_costs = current_costs + previous_costs
    cost_is_complete = bool(all_costs) and all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        for value in all_costs
    )

    def token_total(field: str) -> int | None:
        values = [trace["usage"][field] for trace in traces]
        values.extend(
            attempt.get("usage", {}).get(field)
            for attempt in previous_attempts
        )
        if not values or any(type(value) is not int for value in values):
            return None
        return sum(values)

    durations = [trace["duration_ms"] for trace in traces]
    durations.extend(
        attempt.get("duration_ms")
        for attempt in previous_attempts
        if isinstance(attempt.get("duration_ms"), (int, float))
    )
    previous_api_calls = sum(
        attempt.get("usage", {}).get("api_calls", 0)
        for attempt in previous_attempts
    )
    return {
        **config,
        "case_count": len(traces),
        "status_counts": status_counts,
        "search_count": sum(trace["search_count"] for trace in traces),
        "api_calls": (
            sum(trace["usage"]["api_calls"] for trace in traces)
            + previous_api_calls
        ),
        "prompt_tokens": token_total("prompt_tokens"),
        "completion_tokens": token_total("completion_tokens"),
        "total_tokens": token_total("total_tokens"),
        "known_reported_cost_usd": sum(known_costs),
        "reported_cost_usd": sum(all_costs) if cost_is_complete else None,
        "failed_attempt_count": len(previous_attempts),
        "total_duration_seconds": sum(durations) / 1000 if durations else None,
        "median_duration_ms": statistics.median(durations) if durations else None,
        "mean_duration_ms": statistics.mean(durations) if durations else None,
        "review_status": "pending_manual_review",
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run real OpenRouter agent evaluations on a benchmark split"
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"), default="B")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "starter" / "results" / "agent_eval",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial run without repeating completed model calls",
    )
    args = parser.parse_args()
    if min(args.top_k, args.max_searches, args.budget, args.candidate_k) <= 0:
        raise ValueError("top-k, max-searches, budget, and candidate-k must be positive")
    if args.candidate_k < args.top_k or args.rrf_k < 0:
        raise ValueError("candidate-k must be >= top-k and rrf-k must be non-negative")
    if args.max_consecutive_errors <= 0:
        raise ValueError("max-consecutive-errors must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be unique")

    controller = OpenRouterController()
    benchmark = json.loads(
        (ROOT / "data" / "benchmark.json").read_text(encoding="utf-8")
    )
    questions = [
        question for question in benchmark if question["split"] == args.split
    ]
    if not questions:
        raise ValueError(f"Benchmark has no {args.split} questions")

    dense = Day6ChromaRetriever(strategy=args.strategy)
    chunks = load_indexed_chunks(dense)
    validate(chunks, [question for question in questions if question.get("gold")])
    index_started = time.perf_counter()
    lexical = TfidfIndex(chunks)
    index_build_seconds = time.perf_counter() - index_started
    searches = build_searches(
        dense,
        lexical,
        len(chunks),
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )

    output_root = args.output.resolve() / args.split
    completed = 0
    errors = 0
    for method in args.methods:
        method_dir = output_root / method
        cases_path = method_dir / "cases.json"
        traces_path = method_dir / "traces.json"
        summary_path = method_dir / "summary.json"
        config_path = method_dir / "run_config.json"
        config = {
            "split": args.split,
            "method": method,
            "collection": f"day6_{args.strategy}",
            "model": controller.model,
            "top_k": args.top_k,
            "max_searches": args.max_searches,
            "context_budget_chars": args.budget,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
        }

        if method_dir.exists() and not args.resume:
            raise FileExistsError(
                f"{method_dir} exists; use --resume to avoid repeating API calls"
            )
        existing_config = read_json(config_path, None)
        if existing_config is not None and existing_config != config:
            raise ValueError(
                f"Run configuration differs from saved config in {config_path}"
            )
        cases = read_json(cases_path, [])
        traces = read_json(traces_path, [])
        failed_ids = {
            trace["benchmark_id"]
            for trace in traces
            if trace["status"] == "error"
        }
        previous_attempts: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            previous_attempts.setdefault(
                trace["benchmark_id"], []
            ).extend(trace.get("previous_attempts", []))
            if trace["status"] == "error":
                previous_attempts.setdefault(trace["benchmark_id"], []).append(
                    {
                        "error": trace.get("error", {}),
                        "usage": trace.get("usage", {}),
                        "duration_ms": trace.get("duration_ms"),
                    }
                )
        if args.resume and failed_ids:
            traces = [
                trace
                for trace in traces
                if not (
                    trace["benchmark_id"] in failed_ids
                    and trace["status"] == "error"
                )
            ]
            cases = [
                case for case in cases if case["benchmark_id"] not in failed_ids
            ]
        completed_ids = {
            trace["benchmark_id"]
            for trace in traces
            if trace["status"] != "error"
        }
        write_json(config_path, config)
        write_json(cases_path, cases)
        write_json(traces_path, traces)

        retriever = SearchAdapter(searches[method])
        consecutive_errors = 0
        for question in questions:
            if question["id"] in completed_ids:
                continue
            query_id = f"{method}__{question['id']}"
            call_start = controller.api_call_count
            usage_start = len(controller.usage_records)
            started = time.perf_counter()
            error_record = None
            try:
                result = run_agent(
                    question["question"],
                    controller,
                    retriever,
                    filters=question.get("filter"),
                    max_searches=args.max_searches,
                    top_k=args.top_k,
                    budget=args.budget,
                )
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                result = {
                    "status": "error",
                    "answer": None,
                    "citations": [],
                    "events": [],
                }
                error_record = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                print(
                    f"[ERROR] {query_id}: {error_record['type']}: "
                    f"{error_record['message']}",
                    file=sys.stderr,
                    flush=True,
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            usage = summarize_usage(controller, call_start, usage_start)
            answerable = bool(question.get("gold"))
            trace = {
                "query_id": query_id,
                "benchmark_id": question["id"],
                "split": args.split,
                "method": method,
                "question": question["question"],
                "answerable": answerable,
                "expected_answer": question.get("expected_answer"),
                "gold": question.get("gold", []),
                "absence_reason": question.get("absence_reason"),
                "status": result["status"],
                "answer": result.get("answer"),
                "citations": result.get("citations", []),
                "events": result.get("events", []),
                "search_count": sum(
                    event.get("event") == "search"
                    for event in result.get("events", [])
                ),
                "duration_ms": elapsed_ms,
                "usage": usage,
            }
            if error_record is not None:
                trace["error"] = error_record
            if previous_attempts.get(question["id"]):
                trace["previous_attempts"] = previous_attempts[question["id"]]
            case = {
                "id": query_id,
                "benchmark_id": question["id"],
                "split": args.split,
                "method": method,
                "answerable": answerable,
                "question": question["question"],
            }
            cases.append(case)
            traces.append(trace)
            completed_ids.add(question["id"])
            completed += 1
            errors += result["status"] == "error"
            write_json(cases_path, cases)
            write_json(traces_path, traces)
            write_json(
                summary_path,
                summarize_traces(
                    {**config, "index_build_seconds": index_build_seconds}, traces
                ),
            )
            print(
                f"[{method}] {question['id']}: {result['status']}; "
                f"{usage['api_calls']} API call(s); {elapsed_ms:.0f} ms",
                file=sys.stderr,
                flush=True,
            )
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Stopped after {consecutive_errors} consecutive errors; "
                    f"partial traces are saved under {method_dir}"
                )
        write_json(
            summary_path,
            summarize_traces(
                {**config, "index_build_seconds": index_build_seconds}, traces
            ),
        )

    print(
        json.dumps(
            {
                "split": args.split,
                "methods": list(args.methods),
                "new_cases_completed": completed,
                "errors": errors,
                "output": str(output_root),
                "note": "Answer, citation, and abstention labels require manual review.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
