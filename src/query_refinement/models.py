from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RefinementMethod = Literal[
    "word2vec",                 # legacy raw Word2Vec expansion
    "context_word2vec",         # Word2Vec candidates validated by PRF/top-doc context
    "prf",                      # pseudo relevance feedback only
    "prf_word2vec",             # PRF terms + context-aware Word2Vec terms
    "history",                  # optional history-based refinement
    "history_word2vec",         # optional history + raw Word2Vec
    "prf_history_word2vec",     # PRF + history + context-aware Word2Vec
]


@dataclass
class QueryRefinementConfig:
    """
    Configuration for biomedical query refinement.

    The recommended methods for the final project are:
    - prf
    - context_word2vec
    - prf_word2vec

    Raw Word2Vec expansion is kept only for comparison because it may cause
    query drift when similar terms are added without validating them against
    the current query context.
    """

    method: RefinementMethod = "prf"

    # Global query expansion controls
    max_expansion_terms: int = 5
    max_final_terms: int = 40

    # Word2Vec expansion controls
    word2vec_topn: int = 8
    min_word2vec_similarity: float = 0.60

    # PRF controls
    prf_feedback_docs: int = 10
    max_prf_terms: int = 5
    prf_min_doc_frequency: int = 1
    prf_max_doc_frequency_ratio: float = 0.90

    # Context-aware Word2Vec validation controls
    context_feedback_docs: int = 10
    context_min_doc_frequency: int = 1
    context_require_prf_support: bool = True
    context_candidate_pool_size: int = 40

    # Safety gate controls. When enabled, the wrapper compares top results for
    # the original query and refined query. If the refined query causes a large
    # result-set shift, the system falls back to the original query.
    enable_safety_gate: bool = True
    safety_top_k: int = 10
    safety_min_overlap_ratio: float = 0.20

    # Semantic history controls
    history_similarity_threshold: float = 0.65
    top_history_queries: int = 2
    max_history_terms: int = 5

    # Query construction controls
    original_term_weight: int = 2

    # Biomedical-safe filtering
    min_token_length: int = 2
    allow_numeric_tokens: bool = False
    stop_terms: set[str] = field(default_factory=lambda: {
        # Generic task words common in TREC Genomics queries.
        "provide", "information", "describe", "procedure", "procedures",
        "method", "methods", "role", "roles", "process", "different",
        "exact", "specific", "used", "using", "use", "take", "takes",
        "place", "data", "documentation", "knowledge", "about", "with",
        "without", "within", "through", "into", "from", "that", "this",
        "these", "those", "their", "there", "what", "when", "where",
        "how", "the", "and", "or", "of", "in", "on", "for", "to", "a",
        "an", "is", "are", "be", "by", "as", "at", "it", "its", "they",
        # Very common biomedical article words that are usually bad expansion terms.
        "study", "studies", "result", "results", "background", "objective",
        "objectives", "conclusion", "conclusions", "patient", "patients",
        "case", "cases", "control", "controls", "analysis", "analyses",
        "sample", "samples", "significant", "significantly", "observed",
        "reported", "found", "showed", "shown", "suggest", "suggested",
        "associated", "association", "effect", "effects", "level", "levels",
        "expression", "expressed", "activity", "activities", "human", "humans",
        "cell", "cells", "protein", "proteins", "gene", "genes", "mutation",
        "mutations", "disease", "diseases",
    })


@dataclass
class QueryRefinementResult:
    original_query: str
    processed_query: str
    refined_query: str
    method: str
    original_terms: list[str]
    history_terms: list[str] = field(default_factory=list)
    prf_terms: list[str] = field(default_factory=list)
    word2vec_terms: list[str] = field(default_factory=list)
    selected_history: list[dict[str, Any]] = field(default_factory=list)
    prf_debug: dict[str, Any] = field(default_factory=dict)
    word2vec_debug: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.refined_query.strip() != self.processed_query.strip()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed"] = self.changed
        return data
