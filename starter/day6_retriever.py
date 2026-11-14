"""Adapter from the Day 6 persistent Chroma index to the Day 7 Retriever contract."""
import os
from pathlib import Path
import sys
from typing import Any


class Day6ChromaRetriever:
    def __init__(
        self,
        day6_starter_path: str | Path | None = None,
        strategy: str | None = None,
    ) -> None:
        configured_root = day6_starter_path or os.getenv("DAY6_STARTER_PATH")
        if configured_root is None or not str(configured_root).strip():
            raise ValueError(
                "Set DAY6_STARTER_PATH to the Day 6 starter directory or pass it explicitly"
            )

        self.day6_starter_path = Path(configured_root).expanduser().resolve()
        if not (self.day6_starter_path / "src" / "store.py").is_file():
            raise FileNotFoundError(
                f"Day 6 source was not found under {self.day6_starter_path}"
            )

        if str(self.day6_starter_path) not in sys.path:
            sys.path.insert(0, str(self.day6_starter_path))

        from dotenv import load_dotenv

        load_dotenv(self.day6_starter_path / ".env")

        configured_store = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")
        store_path = Path(configured_store).expanduser()
        if not store_path.is_absolute():
            store_path = self.day6_starter_path / store_path
        self.store_path = store_path.resolve()
        if not (self.store_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                f"Day 6 Chroma database was not found at {self.store_path}; "
                "run the Day 6 ingestion pipeline first"
            )

        from src.embeddings import ChromaLocalEmbedder
        from src.store import ChromaVectorStore

        self.embedder = ChromaLocalEmbedder(
            os.getenv("EMBEDDING_MODEL", ChromaLocalEmbedder.default_model)
        )
        self.strategy = strategy or os.getenv("DAY6_STRATEGY", "B")
        self.store = ChromaVectorStore(self.store_path, self.embedder.dimensions)
        count = self.store.count(self.strategy)
        if count == 0:
            raise ValueError(
                f"Day 6 collection day6_{self.strategy} is empty at {self.store_path}"
            )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(
            strategy=self.strategy,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        results: list[dict[str, Any]] = []
        for hit in hits:
            metadata = hit["metadata"] or {}
            source_id = metadata.get("document_id")
            source = metadata.get("source")
            page = metadata.get("page")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"Chunk {hit['chunk_id']!r} has no document_id metadata")
            if not isinstance(source, str) or not source.strip() or page is None:
                raise ValueError(
                    f"Chunk {hit['chunk_id']!r} is missing source or page metadata"
                )
            results.append(
                {
                    "id": hit["chunk_id"],
                    "text": hit["text"],
                    "source_id": source_id,
                    "source": source,
                    "page": page,
                    "metadata": metadata,
                    "score": hit["score"],
                }
            )
        return results
