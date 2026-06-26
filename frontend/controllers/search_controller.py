from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_retriever  # Reuse the official project retriever factory.
from src.query_refinement import QueryRefinedRetriever, QueryRefinementConfig, QueryRefinementService
from frontend.controllers.dataset_controller import judge_output_with_qrels


@dataclass(frozen=True)
class SearchConfig:
    model: str
    query: str
    top_k: int
    bm25_k1: float
    bm25_b: float
    index_path: str
    db_path: str
    bert_index_dir: str
    bert_model_name: str
    word2vec_index_dir: str
    first_stage: str = "bm25"
    second_stage: str = "bert"
    candidate_k: int = 100
    parallel_models: tuple[str, ...] = ("bm25", "tfidf", "word2vec")
    fusion_method: str = "rrf"
    per_model_k: int = 100
    enable_refinement: bool = False
    refinement_method: str = "prf"
    refinement_prf_feedback_docs: int = 10
    refinement_max_prf_terms: int = 3
    refinement_max_expansion_terms: int = 2
    refinement_word2vec_topn: int = 8
    refinement_min_word2vec_similarity: float = 0.60
    refinement_original_term_weight: int = 2
    refinement_history_threshold: float = 0.65
    refinement_top_history_queries: int = 2
    refinement_max_history_terms: int = 5
    query_history_file: str | None = None
    query_mode: str = "Manual Query"
    official_dataset_id: str | None = None
    official_qid: str | None = None


MODEL_LABEL_TO_VALUE = {
    "BM25": "bm25",
    "TF-IDF": "tfidf",
    "Word2Vec": "word2vec",
    "BERT": "bert",
    "Serial Hybrid": "serial",
    "Parallel Hybrid": "parallel",
}


PARALLEL_LABEL_TO_VALUE = {
    "BM25": "bm25",
    "TF-IDF": "tfidf",
    "Word2Vec": "word2vec",
    "BERT": "bert",
}


@lru_cache(maxsize=12)
def _build_base_retriever_cached(
    model: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    index_path: str,
    db_path: str,
    bert_index_dir: str,
    bert_model_name: str,
    word2vec_index_dir: str,
    first_stage: str,
    second_stage: str,
    candidate_k: int,
    parallel_models: tuple[str, ...],
    fusion_method: str,
    per_model_k: int,
):
    """Build and cache retrievers so the UI does not reload indexes on every click."""
    return build_retriever(
        model=model,
        index_path=index_path,
        db_path=db_path,
        top_k=top_k,
        bert_index_dir=bert_index_dir,
        bert_model_name=bert_model_name,
        word2vec_index_dir=word2vec_index_dir,
        first_stage=first_stage,
        second_stage=second_stage,
        candidate_k=candidate_k,
        parallel_models=list(parallel_models),
        fusion_method=fusion_method,
        per_model_k=per_model_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only arguments supported by the installed project version."""
    try:
        params = inspect.signature(callable_obj).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        return kwargs


def _build_refinement_service(config: SearchConfig) -> QueryRefinementService:
    qr_kwargs = {
        "method": config.refinement_method,
        "prf_feedback_docs": config.refinement_prf_feedback_docs,
        "max_prf_terms": config.refinement_max_prf_terms,
        "max_expansion_terms": config.refinement_max_expansion_terms,
        "word2vec_topn": config.refinement_word2vec_topn,
        "min_word2vec_similarity": config.refinement_min_word2vec_similarity,
        "original_term_weight": config.refinement_original_term_weight,
        "history_similarity_threshold": config.refinement_history_threshold,
        "top_history_queries": config.refinement_top_history_queries,
        "max_history_terms": config.refinement_max_history_terms,
    }
    qr_config = QueryRefinementConfig(**_filter_kwargs(QueryRefinementConfig, qr_kwargs))

    service_kwargs = {
        "config": qr_config,
        "word2vec_index_dir": config.word2vec_index_dir,
        "history_file": config.query_history_file,
        "sentence_model_name": config.bert_model_name,
        # PRF needs the document store and the lexical first-stage retriever context.
        "db_path": config.db_path,
        "index_path": config.index_path,
    }
    return QueryRefinementService(**_filter_kwargs(QueryRefinementService, service_kwargs))


def build_search_config(
    query: str,
    model_label: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    index_path: str,
    db_path: str,
    bert_index_dir: str,
    bert_model_name: str,
    word2vec_index_dir: str,
    first_stage: str,
    second_stage: str,
    candidate_k: int,
    parallel_model_labels: Iterable[str],
    per_model_k: int,
    enable_refinement: bool,
    refinement_method: str,
    prf_feedback_docs: int,
    max_prf_terms: int,
    max_expansion_terms: int,
    word2vec_topn: int,
    min_word2vec_similarity: float,
    original_term_weight: int,
    query_history_file: str | None,
    query_mode: str = "Manual Query",
    official_dataset_id: str | None = None,
    official_qid: str | None = None,
) -> SearchConfig:
    model = MODEL_LABEL_TO_VALUE.get(model_label, model_label).lower().strip()
    parallel_models = tuple(PARALLEL_LABEL_TO_VALUE.get(x, x).lower().strip() for x in parallel_model_labels)
    return SearchConfig(
        model=model,
        query=query.strip(),
        top_k=int(top_k),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        index_path=index_path.strip(),
        db_path=db_path.strip(),
        bert_index_dir=bert_index_dir.strip(),
        bert_model_name=bert_model_name.strip(),
        word2vec_index_dir=word2vec_index_dir.strip(),
        first_stage=first_stage,
        second_stage=second_stage,
        candidate_k=int(candidate_k),
        parallel_models=parallel_models,
        per_model_k=int(per_model_k),
        enable_refinement=bool(enable_refinement),
        refinement_method=refinement_method,
        refinement_prf_feedback_docs=int(prf_feedback_docs),
        refinement_max_prf_terms=int(max_prf_terms),
        refinement_max_expansion_terms=int(max_expansion_terms),
        refinement_word2vec_topn=int(word2vec_topn),
        refinement_min_word2vec_similarity=float(min_word2vec_similarity),
        refinement_original_term_weight=int(original_term_weight),
        query_history_file=query_history_file.strip() or None if query_history_file else None,
        query_mode=query_mode,
        official_dataset_id=official_dataset_id.strip() if official_dataset_id else None,
        official_qid=official_qid.strip() if official_qid else None,
    )


def execute_search(config: SearchConfig) -> dict[str, Any]:
    if not config.query:
        raise ValueError("Query is required.")

    base_retriever = _build_base_retriever_cached(
        config.model,
        config.top_k,
        config.bm25_k1,
        config.bm25_b,
        config.index_path,
        config.db_path,
        config.bert_index_dir,
        config.bert_model_name,
        config.word2vec_index_dir,
        config.first_stage,
        config.second_stage,
        config.candidate_k,
        config.parallel_models,
        config.fusion_method,
        config.per_model_k,
    )

    retriever = base_retriever
    if config.enable_refinement:
        refinement_service = _build_refinement_service(config)
        retriever = QueryRefinedRetriever(
            base_retriever=base_retriever,
            refinement_service=refinement_service,
        )

    started = time.time()
    output = retriever.search(config.query)
    output.setdefault("ui_total_time_seconds", time.time() - started)

    if config.query_mode == "Official Dataset Query" and config.official_dataset_id and config.official_qid:
        output = judge_output_with_qrels(
            output=output,
            dataset_id=config.official_dataset_id,
            qid=config.official_qid,
        )

    return output


def output_to_json(output: dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, indent=2, default=str)
