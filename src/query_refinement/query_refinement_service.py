from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.preprocessing import QueryProcessor
from src.preprocessing import DocumentProcessor
from src.query_refinement.history_refiner import SemanticHistoryRefiner
from src.query_refinement.models import QueryRefinementConfig, QueryRefinementResult
from src.query_refinement.prf_refiner import PRFRefiner
from src.query_refinement.token_utils import top_doc_ids, tokenize_terms, unique_preserve_order
from src.query_refinement.word2vec_expander import Word2VecQueryExpander


PRF_METHODS = {"prf", "prf_word2vec", "prf_history_word2vec"}
CONTEXT_W2V_METHODS = {"context_word2vec", "prf_word2vec", "prf_history_word2vec"}
W2V_METHODS = {"word2vec", "context_word2vec", "history_word2vec", "prf_word2vec", "prf_history_word2vec"}
HISTORY_METHODS = {"history", "history_word2vec", "prf_history_word2vec"}


class QueryRefinementService:
    """
    Production-style Query Refinement service.

    Recommended methods:
    - prf: expands using Pseudo Relevance Feedback from first-stage top docs.
    - context_word2vec: generates Word2Vec candidates and validates them using
      the first-stage retrieval context.
    - prf_word2vec: combines PRF terms and context-aware Word2Vec terms.

    Raw word2vec is preserved for comparison, but it is not recommended as the
    final refinement strategy because it can cause query drift.
    """

    def __init__(
        self,
        config: QueryRefinementConfig | None = None,
        word2vec_index_dir: str | Path | None = None,
        history_queries: list[str] | None = None,
        history_file: str | Path | None = None,
        sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.config = config or QueryRefinementConfig()
        # self.query_processor = QueryProcessor()
        self.document_processor = DocumentProcessor()
        self.history_queries = history_queries or SemanticHistoryRefiner.load_history_file(history_file)

        self.history_refiner: SemanticHistoryRefiner | None = None
        if self.config.method in HISTORY_METHODS:
            self.history_refiner = SemanticHistoryRefiner(
                model_name=sentence_model_name,
                config=self.config,
            )

        self.prf_refiner: PRFRefiner | None = None
        if self.config.method in PRF_METHODS or self.config.method in CONTEXT_W2V_METHODS:
            self.prf_refiner = PRFRefiner(config=self.config)

        self.word2vec_expander: Word2VecQueryExpander | None = None
        if self.config.method in W2V_METHODS:
            if not word2vec_index_dir:
                raise ValueError("word2vec_index_dir is required for Word2Vec query expansion.")
            self.word2vec_expander = Word2VecQueryExpander(
                word2vec_index_dir=word2vec_index_dir,
                config=self.config,
            )

    def needs_feedback(self) -> bool:
        """Whether this method requires first-stage top documents."""
        return self.config.method in PRF_METHODS or self.config.method in CONTEXT_W2V_METHODS

    def refine(
        self,
        query: str,
        feedback_documents: list[dict[str, Any]] | None = None,
    ) -> QueryRefinementResult:
        original_query = "" if query is None else str(query)
        # processed_query = self.query_processor.process(original_query)
        # original_terms = tokenize_terms(processed_query)
        processed_query = self.document_processor.process_to_text(original_query)
        original_terms = self.document_processor.process(original_query)

        # Preserve the user's original intent by repeating original terms.
        final_terms: list[str] = []
        for _ in range(max(1, self.config.original_term_weight)):
            final_terms.extend(original_terms)

        history_terms: list[str] = []
        selected_history: list[dict[str, Any]] = []
        prf_terms: list[str] = []
        prf_debug: dict[str, Any] = {}
        w2v_terms: list[str] = []
        w2v_debug: dict[str, Any] = {}

        existing_terms = unique_preserve_order(final_terms)

        if self.history_refiner is not None:
            history_terms, selected_history = self.history_refiner.refine(
                current_query=processed_query or original_query,
                history_queries=self.history_queries,
                existing_terms=existing_terms,
            )
            final_terms.extend(history_terms)
            existing_terms = unique_preserve_order(final_terms)

        # PRF can either directly add terms or only provide a validation context
        # for context-aware Word2Vec.
        prf_support_terms: list[str] = []
        prf_support_scores: dict[str, float] = {}
        if self.prf_refiner is not None:
            pool_size = max(self.config.context_candidate_pool_size, self.config.max_prf_terms)
            prf_support_terms, prf_debug = self.prf_refiner.extract_terms(
                original_terms=original_terms,
                feedback_documents=feedback_documents,
                existing_terms=existing_terms,
                max_terms=pool_size,
            )
            prf_support_scores = dict(prf_debug.get("candidate_scores", {}))

            if self.config.method in PRF_METHODS:
                prf_terms = prf_support_terms[: self.config.max_prf_terms]
                final_terms.extend(prf_terms)
                existing_terms = unique_preserve_order(final_terms)

        if self.word2vec_expander is not None:
            seed_terms = unique_preserve_order(original_terms + history_terms + prf_terms)
            context_aware = self.config.method in CONTEXT_W2V_METHODS
            allowed_terms = set(prf_support_terms) if context_aware else None
            w2v_terms, w2v_debug = self.word2vec_expander.expand(
                seed_terms=seed_terms,
                existing_terms=existing_terms,
                feedback_documents=feedback_documents,
                allowed_terms=allowed_terms,
                allowed_term_scores=prf_support_scores,
                context_aware=context_aware,
            )
            final_terms.extend(w2v_terms)

        # Keep repeated original terms but deduplicate appended expansion terms.
        original_weighted_len = len(original_terms) * max(1, self.config.original_term_weight)
        weighted_original = final_terms[:original_weighted_len]
        added_terms = unique_preserve_order(final_terms[original_weighted_len:])
        final_terms = weighted_original + added_terms
        final_terms = [term for term in final_terms if term]
        final_terms = final_terms[: self.config.max_final_terms]

        refined_query = " ".join(final_terms).strip()

        return QueryRefinementResult(
            original_query=original_query,
            processed_query=processed_query,
            refined_query=refined_query or processed_query,
            method=self.config.method,
            original_terms=original_terms,
            history_terms=history_terms,
            prf_terms=prf_terms,
            word2vec_terms=w2v_terms,
            selected_history=selected_history,
            prf_debug=prf_debug,
            word2vec_debug=w2v_debug,
        )


class QueryRefinedRetriever:
    """
    Decorator/wrapper that applies QueryRefinementService before retrieval.

    For PRF and context-aware methods, the wrapper first retrieves top documents
    using the original query, passes those documents to the refinement service,
    then retrieves again using the refined query. A safety gate can fall back to
    the original query if the refined query drifts too far.
    """

    def __init__(self, base_retriever, refinement_service: QueryRefinementService):
        self.base_retriever = base_retriever
        self.refinement_service = refinement_service

    def _overlap_ratio(self, original_output: dict[str, Any], refined_output: dict[str, Any]) -> float:
        k = self.refinement_service.config.safety_top_k
        original_ids = set(top_doc_ids(original_output, k))
        refined_ids = set(top_doc_ids(refined_output, k))
        if not original_ids or not refined_ids:
            return 0.0
        return len(original_ids.intersection(refined_ids)) / max(1, min(len(original_ids), len(refined_ids)))

    def search(self, query: str) -> dict:
        start = time.time()
        config = self.refinement_service.config

        initial_output = None
        feedback_documents = None

        if self.refinement_service.needs_feedback() or config.enable_safety_gate:
            initial_output = self.base_retriever.search(query)
            feedback_documents = (initial_output or {}).get("results", [])[: config.prf_feedback_docs]

        refinement = self.refinement_service.refine(
            query=query,
            feedback_documents=feedback_documents,
        )

        if not refinement.changed:
            output = initial_output or self.base_retriever.search(query)
            safety_info = {
                "enabled": config.enable_safety_gate,
                "accepted": True,
                "reason": "query_not_changed",
                "overlap_ratio": None,
            }
        else:
            refined_output = self.base_retriever.search(refinement.refined_query)
            output = refined_output
            safety_info = {
                "enabled": config.enable_safety_gate,
                "accepted": True,
                "reason": "accepted",
                "overlap_ratio": None,
            }

            if config.enable_safety_gate and initial_output is not None:
                overlap = self._overlap_ratio(initial_output, refined_output)
                safety_info["overlap_ratio"] = overlap
                safety_info["min_overlap_ratio"] = config.safety_min_overlap_ratio
                safety_info["top_k"] = config.safety_top_k
                if overlap < config.safety_min_overlap_ratio:
                    output = initial_output
                    safety_info["accepted"] = False
                    safety_info["reason"] = "low_result_overlap_fallback_to_original"

        if not isinstance(output, dict):
            output = {"results": output}

        output["query"] = refinement.original_query
        output["refined_query"] = refinement.refined_query
        output["refinement"] = refinement.to_dict()
        output["refinement_changed"] = refinement.changed
        output["refinement_safety"] = safety_info
        output["refinement_total_time_seconds"] = time.time() - start
        return output
