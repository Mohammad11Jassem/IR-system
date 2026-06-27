from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.query_refinement.models import QueryRefinementConfig
from src.query_refinement.token_utils import is_meaningful_token, tokenize_terms, unique_preserve_order
from src.preprocessing import DocumentProcessor

class SemanticHistoryRefiner:
    """
    Selects semantically similar previous queries and extracts useful terms.

    This implements the user-history part of Query Refinement without coupling
    it to a UI or database. A caller can pass history queries directly, or load
    them from a file via load_history_file().
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: QueryRefinementConfig | None = None,
    ):
        self.model_name = model_name
        self.config = config or QueryRefinementConfig()
        self.model = SentenceTransformer(model_name)
        self.document_processor = DocumentProcessor()

    @staticmethod
    def load_history_file(path: str | Path | None) -> list[str]:
        if not path:
            return []

        history_path = Path(path)
        if not history_path.exists():
            raise FileNotFoundError(f"Query history file not found: {history_path}")

        suffix = history_path.suffix.lower()

        if suffix == ".json":
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                history = []
                for item in data:
                    if isinstance(item, str):
                        history.append(item)
                    elif isinstance(item, dict):
                        value = item.get("query") or item.get("text") or item.get("q")
                        if value:
                            history.append(str(value))
                return [q.strip() for q in history if q and q.strip()]

            raise ValueError("JSON history must be a list of strings or objects with query/text.")

        # Default: one query per line.
        return [
            line.strip()
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def refine(
        self,
        current_query: str,
        history_queries: list[str],
        existing_terms: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not history_queries:
            return [], []

        query_embedding = self.model.encode(current_query, normalize_embeddings=True)
        history_embeddings = self.model.encode(history_queries, normalize_embeddings=True)

        similarities = np.dot(history_embeddings, query_embedding)
        ranked = sorted(
            zip(history_queries, similarities),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        selected_history: list[dict[str, Any]] = []
        for query, score in ranked:
            score = float(score)
            if score < self.config.history_similarity_threshold:
                continue

            selected_history.append({"query": query, "similarity": score})
            if len(selected_history) >= self.config.top_history_queries:
                break

        if not selected_history:
            return [], []

        existing = set(existing_terms)
        candidate_terms: list[str] = []

        for item in selected_history:
            # for token in tokenize_terms(item["query"]):
            for token in self.document_processor.process(item["query"]):
                if token in existing:
                    continue
                if not is_meaningful_token(
                    token=token,
                    stop_terms=self.config.stop_terms,
                    min_token_length=self.config.min_token_length,
                    allow_numeric_tokens=self.config.allow_numeric_tokens,
                ):
                    continue
                candidate_terms.append(token)

        terms = unique_preserve_order(candidate_terms)[: self.config.max_history_terms]
        return terms, selected_history
