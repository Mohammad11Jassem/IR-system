from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from src.query_refinement.models import QueryRefinementConfig
from src.query_refinement.token_utils import (
    document_to_text,
    is_gene_like,
    is_meaningful_token,
    tokenize_terms,
)


class PRFRefiner:
    """
    Pseudo Relevance Feedback query expansion.

    The refiner assumes the top documents returned by the first-stage retriever
    are pseudo-relevant. It extracts high-quality candidate terms from those
    documents and adds only a small number of terms to reduce query drift.
    """

    def __init__(self, config: QueryRefinementConfig | None = None):
        self.config = config or QueryRefinementConfig()

    def extract_terms(
        self,
        original_terms: list[str],
        feedback_documents: list[dict[str, Any]] | None,
        existing_terms: list[str] | None = None,
        max_terms: int | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        feedback_documents = list(feedback_documents or [])[: self.config.prf_feedback_docs]
        existing = set(existing_terms or original_terms)
        original_set = set(original_terms)
        max_terms = self.config.max_prf_terms if max_terms is None else max_terms

        if not feedback_documents:
            return [], {
                "reason": "no_feedback_documents",
                "candidate_scores": {},
                "doc_frequency": {},
                "feedback_doc_count": 0,
            }

        term_tf: Counter[str] = Counter()
        term_df: Counter[str] = Counter()
        rank_weight_sum: defaultdict[str, float] = defaultdict(float)

        for rank, doc in enumerate(feedback_documents, start=1):
            text = document_to_text(doc)
            tokens = tokenize_terms(text)
            if not tokens:
                continue

            filtered_tokens = []
            for token in tokens:
                if token in existing or token in original_set:
                    continue
                if not is_meaningful_token(
                    token=token,
                    stop_terms=self.config.stop_terms,
                    min_token_length=self.config.min_token_length,
                    allow_numeric_tokens=self.config.allow_numeric_tokens,
                ):
                    continue
                filtered_tokens.append(token)

            if not filtered_tokens:
                continue

            counts = Counter(filtered_tokens)
            term_tf.update(counts)
            for term in counts:
                term_df[term] += 1
                # Higher-ranked feedback docs are more trusted.
                rank_weight_sum[term] += 1.0 / math.log2(rank + 1.0)

        feedback_doc_count = max(1, len(feedback_documents))
        candidate_scores: dict[str, float] = {}

        for term, tf in term_tf.items():
            df = term_df[term]
            if df < self.config.prf_min_doc_frequency:
                continue
            if df / feedback_doc_count > self.config.prf_max_doc_frequency_ratio:
                continue

            coverage = df / feedback_doc_count
            rank_bonus = rank_weight_sum[term]
            gene_bonus = 1.20 if is_gene_like(term) else 1.0
            length_bonus = 1.05 if len(term) >= 5 else 1.0

            # The score combines frequency, coverage across feedback docs, and
            # preference for appearances in higher-ranked documents.
            score = (math.log1p(tf) * (1.0 + coverage) * (1.0 + rank_bonus))
            score *= gene_bonus * length_bonus
            candidate_scores[term] = float(score)

        ranked = [
            term
            for term, _ in sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        selected = ranked[:max_terms]
        debug = {
            "feedback_doc_count": len(feedback_documents),
            "candidate_scores": {term: candidate_scores[term] for term in ranked[:50]},
            "doc_frequency": {term: int(term_df[term]) for term in ranked[:50]},
            "selected_terms": selected,
        }
        return selected, debug
