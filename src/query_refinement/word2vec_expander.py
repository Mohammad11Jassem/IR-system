from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from gensim.models import Word2Vec

from src.query_refinement.models import QueryRefinementConfig
from src.query_refinement.token_utils import (
    document_to_text,
    is_meaningful_token,
    tokenize_terms,
)

from src.preprocessing import DocumentProcessor


class Word2VecQueryExpander:
    """
    Word2Vec query expansion with optional context validation.

    Raw Word2Vec is useful for generating candidate terms, but candidates should
    not be blindly added. For context-aware expansion, a candidate is accepted
    only when it is supported by the first-stage retrieval context, typically
    PRF/BM25 top documents.
    """

    def __init__(
        self,
        word2vec_index_dir: str | Path,
        config: QueryRefinementConfig | None = None,
    ):
        self.index_dir = Path(word2vec_index_dir)
        self.model_path = self.index_dir / "word2vec.model"
        self.config = config or QueryRefinementConfig()
        self.document_processor = DocumentProcessor()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Word2Vec model not found: {self.model_path}")

        self.model = Word2Vec.load(str(self.model_path))

    def _context_stats(
        self,
        feedback_documents: list[dict[str, Any]] | None,
    ) -> tuple[set[str], Counter[str]]:
        feedback_documents = list(feedback_documents or [])[: self.config.context_feedback_docs]
        vocab: set[str] = set()
        df: Counter[str] = Counter()

        for doc in feedback_documents:
            # tokens = set(tokenize_terms(document_to_text(doc)))
            tokens = set(self.document_processor.process(document_to_text(doc)))
            for token in tokens:
                if not is_meaningful_token(
                    token=token,
                    stop_terms=self.config.stop_terms,
                    min_token_length=self.config.min_token_length,
                    allow_numeric_tokens=self.config.allow_numeric_tokens,
                ):
                    continue
                vocab.add(token)
                df[token] += 1

        return vocab, df

    def expand(
        self,
        seed_terms: list[str],
        existing_terms: list[str] | None = None,
        feedback_documents: list[dict[str, Any]] | None = None,
        allowed_terms: set[str] | None = None,
        allowed_term_scores: dict[str, float] | None = None,
        context_aware: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        existing = set(existing_terms or seed_terms)
        candidates: dict[str, float] = {}
        raw_candidates: dict[str, float] = {}
        rejected: dict[str, str] = {}
        allowed_term_scores = allowed_term_scores or {}

        context_vocab: set[str] = set()
        context_df: Counter[str] = Counter()
        if context_aware:
            context_vocab, context_df = self._context_stats(feedback_documents)

        for token in seed_terms:
            token = token.lower().strip()
            if not is_meaningful_token(
                token=token,
                stop_terms=self.config.stop_terms,
                min_token_length=self.config.min_token_length,
                allow_numeric_tokens=self.config.allow_numeric_tokens,
            ):
                continue

            if token not in self.model.wv:
                continue

            for word, similarity in self.model.wv.most_similar(
                token,
                topn=self.config.word2vec_topn,
            ):
                word = str(word).lower().strip()
                similarity = float(similarity)
                raw_candidates[word] = max(raw_candidates.get(word, 0.0), similarity)

                if similarity < self.config.min_word2vec_similarity:
                    rejected[word] = "low_similarity"
                    continue
                if word in existing:
                    rejected[word] = "already_existing"
                    continue
                if not is_meaningful_token(
                    token=word,
                    stop_terms=self.config.stop_terms,
                    min_token_length=self.config.min_token_length,
                    allow_numeric_tokens=self.config.allow_numeric_tokens,
                ):
                    rejected[word] = "not_meaningful"
                    continue

                if context_aware:
                    if word not in context_vocab:
                        rejected[word] = "not_in_feedback_context"
                        continue
                    if context_df[word] < self.config.context_min_doc_frequency:
                        rejected[word] = "weak_context_df"
                        continue
                    if self.config.context_require_prf_support and allowed_terms is not None:
                        if word not in allowed_terms:
                            rejected[word] = "not_supported_by_prf"
                            continue

                context_score = float(allowed_term_scores.get(word, 0.0))
                # Combine semantic similarity with context evidence. If no
                # context score exists, the score falls back to similarity.
                combined_score = similarity
                if context_aware:
                    combined_score = (0.65 * context_score) + (0.35 * similarity)
                candidates[word] = max(candidates.get(word, 0.0), combined_score)

        ranked_terms = [
            term
            for term, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        ]
        selected = ranked_terms[: self.config.max_expansion_terms]

        debug = {
            "context_aware": context_aware,
            "raw_candidates": dict(sorted(raw_candidates.items(), key=lambda item: (-item[1], item[0]))[:50]),
            "accepted_scores": {term: candidates[term] for term in ranked_terms[:50]},
            "selected_terms": selected,
            "rejected_sample": dict(list(rejected.items())[:50]),
            "context_vocab_size": len(context_vocab),
        }
        return selected, debug
