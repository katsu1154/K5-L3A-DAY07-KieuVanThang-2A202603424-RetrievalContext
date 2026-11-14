import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.evaluate_retrieval import reciprocal_rank_fusion


def result(chunk_id, score):
    return {
        "id": chunk_id,
        "text": chunk_id,
        "source_id": "doc",
        "source": "doc.txt",
        "page": 1,
        "metadata": {},
        "score": score,
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def test_rrf_promotes_chunks_ranked_by_both_retrievers(self):
        fused = reciprocal_rank_fusion(
            [result("a", 0.9), result("b", 0.8)],
            [result("b", 0.4), result("c", 0.3)],
            rrf_k=60,
            top_k=3,
        )
        self.assertEqual([item["id"] for item in fused], ["b", "a", "c"])
        self.assertEqual(fused[0]["strategy"], "hybrid_rrf")
        self.assertEqual((fused[0]["dense_rank"], fused[0]["tfidf_rank"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
