"""Compare dense, TF-IDF, and RRF retrieval on the Day 7 benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from dotenv import load_dotenv

from starter.baseline import TfidfIndex, hit, pack_context, validate
from starter.day6_retriever import Day6ChromaRetriever


ROOT = Path(__file__).resolve().parents[1]
TOP_K = 3
CANDIDATE_K = 50
RRF_K = 60
CONTEXT_BUDGET_CHARS = 800
SELECTION_RULE = (
    "Select using dev Evidence Hit@3, then dev Hit@1, then dev Context Evidence "
    "Rate, then lower median retrieval latency. Test is not used for selection."
)

Search = Callable[[str, dict[str, Any] | None, int], list[dict[str, Any]]]


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    *,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    if rrf_k < 0 or top_k <= 0:
        raise ValueError("rrf_k must be non-negative and top_k must be positive")

    fused: dict[str, dict[str, Any]] = {}
    for ranking_name, results in (
        ("dense_rank", dense_results),
        ("tfidf_rank", lexical_results),
    ):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            item = fused.setdefault(
                chunk_id,
                {**result, "score": 0.0, "strategy": "hybrid_rrf"},
            )
            item["score"] += 1.0 / (rrf_k + rank)
            item[ranking_name] = rank

    return sorted(fused.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]


def load_indexed_chunks(retriever: Day6ChromaRetriever) -> list[dict[str, Any]]:
    stored = retriever.store.get_chunks(retriever.strategy, {})
    chunks: list[dict[str, Any]] = []
    for row in stored:
        metadata = row["metadata"] or {}
        source_id = metadata.get("document_id")
        source = metadata.get("source")
        page = metadata.get("page")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"Chunk {row['chunk_id']!r} has no document_id metadata")
        if not isinstance(source, str) or not source.strip() or not isinstance(page, int):
            raise ValueError(f"Chunk {row['chunk_id']!r} has invalid source/page metadata")
        chunks.append(
            {
                "id": row["chunk_id"],
                "text": row["text"],
                "source_id": source_id,
                "source": source,
                "page": page,
                "metadata": metadata,
            }
        )
    if not chunks:
        raise ValueError(f"Day 6 collection day6_{retriever.strategy} is empty")
    return chunks


def rate(traces: list[dict[str, Any]], field: str) -> dict[str, Any]:
    answerable = [trace for trace in traces if trace["answerable"]]
    hits = sum(bool(trace[field]) for trace in answerable)
    return {
        "hits": hits,
        "total": len(answerable),
        "rate": hits / len(answerable) if answerable else None,
    }


def evaluate_method(
    method: str,
    search: Search,
    questions: list[dict[str, Any]],
    *,
    context_budget_chars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for question in questions:
        filters = question.get("filter")
        started = time.perf_counter()
        results = search(question["question"], filters, TOP_K)
        search_ms = (time.perf_counter() - started) * 1000
        context, context_ids = pack_context(results, budget=context_budget_chars)
        answerable = bool(question.get("gold"))
        marked = [
            {
                **result,
                **({"hit": hit(result, question["gold"])} if answerable else {}),
            }
            for result in results
        ]
        context_hit = (
            any(result["hit"] and result["id"] in context_ids for result in marked)
            if answerable
            else None
        )
        traces.append(
            {
                "query_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "answerable": answerable,
                "method": method,
                "filters": filters,
                "top_k": TOP_K,
                "context_budget_chars": context_budget_chars,
                "gold": question.get("gold", []),
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "search_ms": search_ms,
                "hit1": bool(marked and marked[0].get("hit", False)) if answerable else None,
                "hit3": any(result.get("hit", False) for result in marked) if answerable else None,
                "context_hit": context_hit,
            }
        )

    answerable_count = sum(trace["answerable"] for trace in traces)
    summary = {
        "method": method,
        "split": questions[0]["split"] if questions else None,
        "question_count": len(traces),
        "answerable_count": answerable_count,
        "unanswerable_count": len(traces) - answerable_count,
        "hit_at_1": rate(traces, "hit1"),
        "evidence_hit_at_3": rate(traces, "hit3"),
        "context_evidence_rate": rate(traces, "context_hit"),
        "median_search_ms": statistics.median(trace["search_ms"] for trace in traces)
        if traces
        else None,
        "mean_search_ms": statistics.mean(trace["search_ms"] for trace in traces)
        if traces
        else None,
        "context_budget_chars": context_budget_chars,
        "service_cost_usd": 0.0,
    }
    return summary, traces


def select_method(summaries: dict[str, dict[str, Any]]) -> str:
    def score(method: str) -> tuple[float, float, float, float]:
        summary = summaries[method]
        return (
            summary["evidence_hit_at_3"]["rate"] or 0.0,
            summary["hit_at_1"]["rate"] or 0.0,
            summary["context_evidence_rate"]["rate"] or 0.0,
            -(summary["median_search_ms"] or 0.0),
        )

    return max(summaries, key=score)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Compare Day 7 retrieval methods on dev, then evaluate frozen choices on test"
    )
    parser.add_argument("--strategy", choices=("A", "B", "C"), default="B")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET_CHARS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "starter" / "results" / "retrieval_eval",
    )
    args = parser.parse_args()
    if args.candidate_k < TOP_K or args.rrf_k < 0 or args.context_budget <= 0:
        raise ValueError("candidate-k must be at least 3, rrf-k non-negative, and budget positive")

    benchmark = json.loads((ROOT / "data" / "benchmark.json").read_text(encoding="utf-8"))
    dev_questions = [question for question in benchmark if question["split"] == "dev"]
    test_questions = [question for question in benchmark if question["split"] == "test"]
    answerable_dev = [question for question in dev_questions if question.get("gold")]
    answerable_test = [question for question in test_questions if question.get("gold")]
    if len(answerable_dev) < 15 or len(answerable_test) < 5:
        raise ValueError("Benchmark must contain at least 15 answerable dev and 5 answerable test cases")

    retriever = Day6ChromaRetriever(strategy=args.strategy)
    chunks = load_indexed_chunks(retriever)
    validate(chunks, answerable_dev)
    validate(chunks, answerable_test)

    index_started = time.perf_counter()
    lexical = TfidfIndex(chunks)
    tfidf_build_seconds = time.perf_counter() - index_started

    def dense_search(query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return retriever.search(query, filters=filters, top_k=top_k)

    def tfidf_search(query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return lexical.search(query, top_k=top_k, filters=filters)

    def hybrid_search(query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        dense_results = retriever.search(
            query, filters=filters, top_k=min(args.candidate_k, len(chunks))
        )
        lexical_results = lexical.search(
            query, top_k=min(args.candidate_k, len(chunks)), filters=filters
        )
        return reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            rrf_k=args.rrf_k,
            top_k=top_k,
        )

    methods: dict[str, Search] = {
        "dense_day6": dense_search,
        "tfidf": tfidf_search,
        "hybrid_rrf": hybrid_search,
    }
    output = args.output.resolve()

    dev_summaries: dict[str, dict[str, Any]] = {}
    dev_traces: list[dict[str, Any]] = []
    for name, search in methods.items():
        summary, traces = evaluate_method(
            name,
            search,
            dev_questions,
            context_budget_chars=args.context_budget,
        )
        summary["index_build_seconds"] = tfidf_build_seconds if name != "dense_day6" else 0.0
        if name == "hybrid_rrf":
            summary["candidate_k"] = args.candidate_k
            summary["rrf_k"] = args.rrf_k
        dev_summaries[name] = summary
        dev_traces.extend(traces)

    selected_method = select_method(dev_summaries)
    dev_report = {
        "split": "dev",
        "collection": f"day6_{args.strategy}",
        "chunk_count": len(chunks),
        "selection_rule": SELECTION_RULE,
        "selected_method": selected_method,
        "methods": dev_summaries,
    }
    write_json(output / "dev_summary.json", dev_report)
    write_json(output / "dev_traces.json", dev_traces)

    test_method_names = list(dict.fromkeys(("dense_day6", selected_method)))
    test_summaries: dict[str, dict[str, Any]] = {}
    test_traces: list[dict[str, Any]] = []
    for name in test_method_names:
        summary, traces = evaluate_method(
            name,
            methods[name],
            test_questions,
            context_budget_chars=args.context_budget,
        )
        summary["index_build_seconds"] = tfidf_build_seconds if name == "tfidf" else 0.0
        if name == "hybrid_rrf":
            summary["candidate_k"] = args.candidate_k
            summary["rrf_k"] = args.rrf_k
        test_summaries[name] = summary
        test_traces.extend(traces)

    test_report = {
        "split": "test",
        "collection": f"day6_{args.strategy}",
        "baseline_method": "dense_day6",
        "selected_on_dev": selected_method,
        "selection_rule": SELECTION_RULE,
        "methods": test_summaries,
    }
    write_json(output / "test_summary.json", test_report)
    write_json(output / "test_traces.json", test_traces)

    print(json.dumps({"dev": dev_report, "test": test_report, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
