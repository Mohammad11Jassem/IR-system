# import argparse
# import sys
# from pathlib import Path


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.datasets.loader import load_qrels, load_queries
# from src.evaluation import EvaluationConfig, EvaluationService
# from src.retrieval import (
#     BM25Retriever,
#     BertRetriever,
#     ParallelHybridRetriever,
#     SerialHybridRetriever,
#     TfidfRetriever,
#     Word2VecRetriever,
# )


# DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
# DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
# DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# def build_retriever(model_name: str, args):
#     model_name = model_name.lower().strip()

#     if model_name == "bm25":
#         return BM25Retriever(
#             index_path=args.index_path,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name in {"tfidf", "tf-idf", "tf_idf"}:
#         return TfidfRetriever(
#             index_path=args.index_path,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name in {"word2vec", "w2v"}:
#         return Word2VecRetriever(
#             index_dir=args.word2vec_index_dir,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name == "bert":
#         return BertRetriever(
#             index_dir=args.bert_index_dir,
#             db_path=args.db_path,
#             model_name=args.bert_model_name,
#             top_k=args.run_depth,
#         )

#     if model_name == "parallel_basic":
#         return ParallelHybridRetriever(
#             models=["bm25", "tfidf", "word2vec"],
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_index_dir=args.bert_index_dir,
#             bert_model_name=args.bert_model_name,
#             fusion_method="rrf",
#             per_model_k=args.per_model_k,
#             top_k=args.run_depth,
#             rrf_k=args.rrf_k,
#         )

#     if model_name == "parallel_full":
#         return ParallelHybridRetriever(
#             models=["bm25", "tfidf", "word2vec", "bert"],
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_index_dir=args.bert_index_dir,
#             bert_model_name=args.bert_model_name,
#             fusion_method="rrf",
#             per_model_k=args.per_model_k,
#             top_k=args.run_depth,
#             rrf_k=args.rrf_k,
#         )

#     if model_name == "serial_bm25_bert":
#         return SerialHybridRetriever(
#             first_stage="bm25",
#             second_stage="bert",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     if model_name == "serial_tfidf_bert":
#         return SerialHybridRetriever(
#             first_stage="tfidf",
#             second_stage="bert",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     if model_name == "serial_bm25_word2vec":
#         return SerialHybridRetriever(
#             first_stage="bm25",
#             second_stage="word2vec",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     raise ValueError(
#         f"Unsupported model: {model_name}. "
#         "Supported examples: bm25, tfidf, word2vec, bert, parallel_basic, "
#         "parallel_full, serial_bm25_bert, serial_tfidf_bert, serial_bm25_word2vec."
#     )


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Evaluate IR retrievers using MAP, Precision@K, Recall@K, and nDCG@K."
#     )

#     parser.add_argument(
#         "--models",
#         nargs="+",
#         default=["bm25", "tfidf", "word2vec", "parallel_basic"],
#         help="Models to evaluate. Example: --models bm25 tfidf word2vec parallel_basic",
#     )

#     parser.add_argument("--dataset", default="main")
#     parser.add_argument("--run-depth", type=int, default=100)
#     parser.add_argument("--precision-k", type=int, default=10)
#     parser.add_argument("--ndcg-k", type=int, default=10)
#     parser.add_argument(
#         "--recall-k",
#         type=int,
#         default=None,
#         help="Defaults to run-depth if omitted.",
#     )
#     parser.add_argument(
#         "--map-depth",
#         type=int,
#         default=None,
#         help="Defaults to run-depth if omitted.",
#     )
#     parser.add_argument("--min-relevance", type=int, default=1)

#     parser.add_argument("--candidate-k", type=int, default=100)
#     parser.add_argument("--per-model-k", type=int, default=100)
#     parser.add_argument("--rrf-k", type=int, default=60)

#     parser.add_argument("--index-path", default=DEFAULT_TERRIER_INDEX_PATH)
#     parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
#     parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
#     parser.add_argument("--bert-index-dir", default=DEFAULT_BERT_INDEX_DIR)
#     parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)

#     parser.add_argument("--output-dir", default="reports/evaluation")
#     parser.add_argument("--limit-queries", type=int, default=None)
#     parser.add_argument("--stop-on-error", action="store_true")
#     parser.add_argument("--no-charts", action="store_true")
#     parser.add_argument("--no-excel", action="store_true")

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     print("Loading queries and qrels...")
#     queries_df = load_queries(args.dataset)
#     qrels_df = load_qrels(args.dataset)

#     if args.limit_queries is not None:
#         queries_df = queries_df.head(args.limit_queries).copy()
#         allowed_qids = set(queries_df["query_id"].astype(str))
#         qrels_df = qrels_df[qrels_df["query_id"].astype(str).isin(allowed_qids)].copy()

#     print("Queries:", len(queries_df))
#     print("Qrels:", len(qrels_df))

#     retrievers = {}
#     for model_name in args.models:
#         normalized_name = model_name.lower().replace("-", "_")
#         print(f"Building retriever: {normalized_name}")
#         retrievers[normalized_name] = build_retriever(normalized_name, args)

#     config = EvaluationConfig(
#         run_depth=args.run_depth,
#         precision_k=args.precision_k,
#         recall_k=args.recall_k,
#         ndcg_k=args.ndcg_k,
#         map_depth=args.map_depth,
#         min_relevance=args.min_relevance,
#         continue_on_error=not args.stop_on_error,
#         save_charts=not args.no_charts,
#         save_excel=not args.no_excel,
#     )

#     service = EvaluationService(
#         output_dir=args.output_dir,
#         config=config,
#     )

#     result = service.evaluate_models(
#         retrievers=retrievers,
#         queries_df=queries_df,
#         qrels_df=qrels_df,
#     )

#     print("\nEvaluation finished.")
#     print("Summary CSV:", result["summary_csv"])
#     print("Summary XLSX:", result["summary_xlsx"])
#     print("Run files:")
#     for model, path in result["run_paths"].items():
#         print(f"  - {model}: {path}")


# if __name__ == "__main__":
#     main()




# import argparse
# import sys
# from pathlib import Path


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.datasets.loader import load_qrels, load_queries
# from src.evaluation import EvaluationConfig, EvaluationService
# from src.retrieval import (
#     BM25Retriever,
#     BertRetriever,
#     ParallelHybridRetriever,
#     SerialHybridRetriever,
#     TfidfRetriever,
#     Word2VecRetriever,
# )
# from src.query_refinement import (
#     QueryRefinedRetriever,
#     QueryRefinementConfig,
#     QueryRefinementService,
# )


# DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
# DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
# DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# def build_query_refinement_service(args) -> QueryRefinementService:
#     config = QueryRefinementConfig(
#         method=args.refinement_method,
#         max_expansion_terms=args.refinement_max_expansion_terms,
#         word2vec_topn=args.refinement_word2vec_topn,
#         min_word2vec_similarity=args.refinement_min_word2vec_similarity,
#         history_similarity_threshold=args.refinement_history_threshold,
#         top_history_queries=args.refinement_top_history_queries,
#         max_history_terms=args.refinement_max_history_terms,
#         original_term_weight=args.refinement_original_term_weight,
#     )

#     return QueryRefinementService(
#         config=config,
#         word2vec_index_dir=args.word2vec_index_dir,
#         history_file=args.query_history_file,
#         sentence_model_name=args.bert_model_name,
#     )


# def build_retriever(model_name: str, args):
#     model_name = model_name.lower().strip()

#     if model_name == "bm25":
#         return BM25Retriever(
#             index_path=args.index_path,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name in {"tfidf", "tf-idf", "tf_idf"}:
#         return TfidfRetriever(
#             index_path=args.index_path,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name in {"word2vec", "w2v"}:
#         return Word2VecRetriever(
#             index_dir=args.word2vec_index_dir,
#             db_path=args.db_path,
#             top_k=args.run_depth,
#         )

#     if model_name == "bert":
#         return BertRetriever(
#             index_dir=args.bert_index_dir,
#             db_path=args.db_path,
#             model_name=args.bert_model_name,
#             top_k=args.run_depth,
#         )

#     if model_name == "parallel_basic":
#         return ParallelHybridRetriever(
#             models=["bm25", "tfidf", "word2vec"],
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_index_dir=args.bert_index_dir,
#             bert_model_name=args.bert_model_name,
#             fusion_method="rrf",
#             per_model_k=args.per_model_k,
#             top_k=args.run_depth,
#             rrf_k=args.rrf_k,
#         )

#     if model_name == "parallel_full":
#         return ParallelHybridRetriever(
#             models=["bm25", "tfidf", "word2vec", "bert"],
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_index_dir=args.bert_index_dir,
#             bert_model_name=args.bert_model_name,
#             fusion_method="rrf",
#             per_model_k=args.per_model_k,
#             top_k=args.run_depth,
#             rrf_k=args.rrf_k,
#         )

#     if model_name == "serial_bm25_bert":
#         return SerialHybridRetriever(
#             first_stage="bm25",
#             second_stage="bert",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     if model_name == "serial_tfidf_bert":
#         return SerialHybridRetriever(
#             first_stage="tfidf",
#             second_stage="bert",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     if model_name == "serial_bm25_word2vec":
#         return SerialHybridRetriever(
#             first_stage="bm25",
#             second_stage="word2vec",
#             index_path=args.index_path,
#             db_path=args.db_path,
#             word2vec_index_dir=args.word2vec_index_dir,
#             bert_model_name=args.bert_model_name,
#             candidate_k=args.candidate_k,
#             top_k=args.run_depth,
#         )

#     raise ValueError(
#         f"Unsupported model: {model_name}. "
#         "Supported examples: bm25, tfidf, word2vec, bert, parallel_basic, "
#         "parallel_full, serial_bm25_bert, serial_tfidf_bert, serial_bm25_word2vec."
#     )


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Evaluate IR retrievers using MAP, Precision@K, Recall@K, and nDCG@K."
#     )

#     parser.add_argument(
#         "--models",
#         nargs="+",
#         default=["bm25", "tfidf", "word2vec", "parallel_basic"],
#         help="Models to evaluate. Example: --models bm25 tfidf word2vec parallel_basic",
#     )

#     parser.add_argument("--dataset", default="main")
#     parser.add_argument("--run-depth", type=int, default=100)
#     parser.add_argument("--precision-k", type=int, default=10)
#     parser.add_argument("--ndcg-k", type=int, default=10)
#     parser.add_argument(
#         "--recall-k",
#         type=int,
#         default=None,
#         help="Defaults to run-depth if omitted.",
#     )
#     parser.add_argument(
#         "--map-depth",
#         type=int,
#         default=None,
#         help="Defaults to run-depth if omitted.",
#     )
#     parser.add_argument("--min-relevance", type=int, default=1)

#     parser.add_argument("--candidate-k", type=int, default=100)
#     parser.add_argument("--per-model-k", type=int, default=100)
#     parser.add_argument("--rrf-k", type=int, default=60)

#     parser.add_argument("--index-path", default=DEFAULT_TERRIER_INDEX_PATH)
#     parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
#     parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
#     parser.add_argument("--bert-index-dir", default=DEFAULT_BERT_INDEX_DIR)
#     parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)

#     parser.add_argument("--output-dir", default="reports/evaluation")
#     parser.add_argument("--limit-queries", type=int, default=None)
#     parser.add_argument("--stop-on-error", action="store_true")
#     parser.add_argument("--no-charts", action="store_true")
#     parser.add_argument("--no-excel", action="store_true")

#     parser.add_argument(
#         "--enable-query-refinement",
#         action="store_true",
#         help="Apply Query Refinement before each retriever search.",
#     )
#     parser.add_argument(
#         "--refinement-method",
#         choices=["word2vec", "history", "history_word2vec"],
#         default="word2vec",
#         help="Query Refinement method.",
#     )
#     parser.add_argument(
#         "--query-history-file",
#         default=None,
#         help="Optional .txt or .json file containing previous user queries.",
#     )
#     parser.add_argument("--refinement-max-expansion-terms", type=int, default=5)
#     parser.add_argument("--refinement-word2vec-topn", type=int, default=5)
#     parser.add_argument("--refinement-min-word2vec-similarity", type=float, default=0.55)
#     parser.add_argument("--refinement-history-threshold", type=float, default=0.65)
#     parser.add_argument("--refinement-top-history-queries", type=int, default=2)
#     parser.add_argument("--refinement-max-history-terms", type=int, default=5)
#     parser.add_argument("--refinement-original-term-weight", type=int, default=1)

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     print("Loading queries and qrels...")
#     queries_df = load_queries(args.dataset)
#     qrels_df = load_qrels(args.dataset)

#     if args.limit_queries is not None:
#         queries_df = queries_df.head(args.limit_queries).copy()
#         allowed_qids = set(queries_df["query_id"].astype(str))
#         qrels_df = qrels_df[qrels_df["query_id"].astype(str).isin(allowed_qids)].copy()

#     print("Queries:", len(queries_df))
#     print("Qrels:", len(qrels_df))

#     retrievers = {}
#     refinement_service = None
#     if args.enable_query_refinement:
#         print(f"Query Refinement enabled: {args.refinement_method}")
#         refinement_service = build_query_refinement_service(args)

#     for model_name in args.models:
#         normalized_name = model_name.lower().replace("-", "_")
#         print(f"Building retriever: {normalized_name}")
#         retriever = build_retriever(normalized_name, args)
#         if refinement_service is not None:
#             retriever = QueryRefinedRetriever(
#                 base_retriever=retriever,
#                 refinement_service=refinement_service,
#             )
#         retrievers[normalized_name] = retriever

#     config = EvaluationConfig(
#         run_depth=args.run_depth,
#         precision_k=args.precision_k,
#         recall_k=args.recall_k,
#         ndcg_k=args.ndcg_k,
#         map_depth=args.map_depth,
#         min_relevance=args.min_relevance,
#         continue_on_error=not args.stop_on_error,
#         save_charts=not args.no_charts,
#         save_excel=not args.no_excel,
#     )

#     service = EvaluationService(
#         output_dir=args.output_dir,
#         config=config,
#     )

#     result = service.evaluate_models(
#         retrievers=retrievers,
#         queries_df=queries_df,
#         qrels_df=qrels_df,
#     )

#     print("\nEvaluation finished.")
#     print("Summary CSV:", result["summary_csv"])
#     print("Summary XLSX:", result["summary_xlsx"])
#     print("Run files:")
#     for model, path in result["run_paths"].items():
#         print(f"  - {model}: {path}")


# if __name__ == "__main__":
#     main()








import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.loader import load_qrels, load_queries
from src.evaluation import EvaluationConfig, EvaluationService
from src.retrieval import (
    BM25Retriever,
    BertRetriever,
    ParallelHybridRetriever,
    SerialHybridRetriever,
    TfidfRetriever,
    Word2VecRetriever,
)
from src.query_refinement import (
    QueryRefinedRetriever,
    QueryRefinementConfig,
    QueryRefinementService,
)


DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



def build_query_refinement_service(args) -> QueryRefinementService:
    config = QueryRefinementConfig(
        method=args.refinement_method,
        max_expansion_terms=args.refinement_max_expansion_terms,
        max_prf_terms=args.refinement_max_prf_terms,
        word2vec_topn=args.refinement_word2vec_topn,
        min_word2vec_similarity=args.refinement_min_word2vec_similarity,
        prf_feedback_docs=args.refinement_prf_feedback_docs,
        prf_min_doc_frequency=args.refinement_prf_min_doc_frequency,
        prf_max_doc_frequency_ratio=args.refinement_prf_max_doc_frequency_ratio,
        context_feedback_docs=args.refinement_context_feedback_docs,
        context_min_doc_frequency=args.refinement_context_min_doc_frequency,
        context_require_prf_support=not args.refinement_no_context_prf_support,
        context_candidate_pool_size=args.refinement_context_candidate_pool_size,
        enable_safety_gate=not args.disable_refinement_safety,
        safety_top_k=args.refinement_safety_top_k,
        safety_min_overlap_ratio=args.refinement_safety_min_overlap_ratio,
        history_similarity_threshold=args.refinement_history_threshold,
        top_history_queries=args.refinement_top_history_queries,
        max_history_terms=args.refinement_max_history_terms,
        original_term_weight=args.refinement_original_term_weight,
    )

    return QueryRefinementService(
        config=config,
        word2vec_index_dir=args.word2vec_index_dir,
        history_file=args.query_history_file,
        sentence_model_name=args.bert_model_name,
    )

def build_retriever(model_name: str, args):
    model_name = model_name.lower().strip()

    if model_name == "bm25":
        return BM25Retriever(
            index_path=args.index_path,
            db_path=args.db_path,
            top_k=args.run_depth,
        )

    if model_name in {"tfidf", "tf-idf", "tf_idf"}:
        return TfidfRetriever(
            index_path=args.index_path,
            db_path=args.db_path,
            top_k=args.run_depth,
        )

    if model_name in {"word2vec", "w2v"}:
        return Word2VecRetriever(
            index_dir=args.word2vec_index_dir,
            db_path=args.db_path,
            top_k=args.run_depth,
        )

    if model_name == "bert":
        return BertRetriever(
            index_dir=args.bert_index_dir,
            db_path=args.db_path,
            model_name=args.bert_model_name,
            top_k=args.run_depth,
        )

    if model_name == "parallel_basic":
        return ParallelHybridRetriever(
            models=["bm25", "tfidf", "word2vec"],
            index_path=args.index_path,
            db_path=args.db_path,
            word2vec_index_dir=args.word2vec_index_dir,
            bert_index_dir=args.bert_index_dir,
            bert_model_name=args.bert_model_name,
            fusion_method="rrf",
            per_model_k=args.per_model_k,
            top_k=args.run_depth,
            rrf_k=args.rrf_k,
        )

    if model_name == "parallel_full":
        return ParallelHybridRetriever(
            models=["bm25", "tfidf", "word2vec", "bert"],
            index_path=args.index_path,
            db_path=args.db_path,
            word2vec_index_dir=args.word2vec_index_dir,
            bert_index_dir=args.bert_index_dir,
            bert_model_name=args.bert_model_name,
            fusion_method="rrf",
            per_model_k=args.per_model_k,
            top_k=args.run_depth,
            rrf_k=args.rrf_k,
        )

    if model_name == "serial_bm25_bert":
        return SerialHybridRetriever(
            first_stage="bm25",
            second_stage="bert",
            index_path=args.index_path,
            db_path=args.db_path,
            word2vec_index_dir=args.word2vec_index_dir,
            bert_model_name=args.bert_model_name,
            candidate_k=args.candidate_k,
            top_k=args.run_depth,
        )

    if model_name == "serial_tfidf_bert":
        return SerialHybridRetriever(
            first_stage="tfidf",
            second_stage="bert",
            index_path=args.index_path,
            db_path=args.db_path,
            word2vec_index_dir=args.word2vec_index_dir,
            bert_model_name=args.bert_model_name,
            candidate_k=args.candidate_k,
            top_k=args.run_depth,
        )

    if model_name == "serial_bm25_word2vec":
        return SerialHybridRetriever(
            first_stage="bm25",
            second_stage="word2vec",
            index_path=args.index_path,
            db_path=args.db_path,
            word2vec_index_dir=args.word2vec_index_dir,
            bert_model_name=args.bert_model_name,
            candidate_k=args.candidate_k,
            top_k=args.run_depth,
        )

    raise ValueError(
        f"Unsupported model: {model_name}. "
        "Supported examples: bm25, tfidf, word2vec, bert, parallel_basic, "
        "parallel_full, serial_bm25_bert, serial_tfidf_bert, serial_bm25_word2vec."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IR retrievers using MAP, Precision@K, Recall@K, and nDCG@K."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bm25", "tfidf", "word2vec", "parallel_basic"],
        help="Models to evaluate. Example: --models bm25 tfidf word2vec parallel_basic",
    )

    parser.add_argument("--dataset", default="main")
    parser.add_argument("--run-depth", type=int, default=100)
    parser.add_argument("--precision-k", type=int, default=10)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help="Defaults to run-depth if omitted.",
    )
    parser.add_argument(
        "--map-depth",
        type=int,
        default=None,
        help="Defaults to run-depth if omitted.",
    )
    parser.add_argument("--min-relevance", type=int, default=1)

    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--per-model-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument("--index-path", default=DEFAULT_TERRIER_INDEX_PATH)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
    parser.add_argument("--bert-index-dir", default=DEFAULT_BERT_INDEX_DIR)
    parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)

    parser.add_argument("--output-dir", default="reports/evaluation")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--no-excel", action="store_true")

    parser.add_argument(
        "--enable-query-refinement",
        action="store_true",
        help="Apply Query Refinement before each retriever search.",
    )
    parser.add_argument(
        "--refinement-method",
        choices=[
            "word2vec",
            "context_word2vec",
            "prf",
            "prf_word2vec",
            "history",
            "history_word2vec",
            "prf_history_word2vec",
        ],
        default="prf",
        help="Query Refinement method. Recommended: prf or prf_word2vec.",
    )
    parser.add_argument(
        "--query-history-file",
        default=None,
        help="Optional .txt or .json file containing previous user queries.",
    )
    parser.add_argument("--refinement-max-expansion-terms", type=int, default=5)
    parser.add_argument("--refinement-max-prf-terms", type=int, default=5)
    parser.add_argument("--refinement-word2vec-topn", type=int, default=8)
    parser.add_argument("--refinement-min-word2vec-similarity", type=float, default=0.60)
    parser.add_argument("--refinement-prf-feedback-docs", type=int, default=10)
    parser.add_argument("--refinement-prf-min-doc-frequency", type=int, default=1)
    parser.add_argument("--refinement-prf-max-doc-frequency-ratio", type=float, default=0.90)
    parser.add_argument("--refinement-context-feedback-docs", type=int, default=10)
    parser.add_argument("--refinement-context-min-doc-frequency", type=int, default=1)
    parser.add_argument("--refinement-context-candidate-pool-size", type=int, default=40)
    parser.add_argument("--refinement-no-context-prf-support", action="store_true")
    parser.add_argument("--disable-refinement-safety", action="store_true")
    parser.add_argument("--refinement-safety-top-k", type=int, default=10)
    parser.add_argument("--refinement-safety-min-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--refinement-history-threshold", type=float, default=0.65)
    parser.add_argument("--refinement-top-history-queries", type=int, default=2)
    parser.add_argument("--refinement-max-history-terms", type=int, default=5)
    parser.add_argument("--refinement-original-term-weight", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading queries and qrels...")
    queries_df = load_queries(args.dataset)
    qrels_df = load_qrels(args.dataset)

    if args.limit_queries is not None:
        queries_df = queries_df.head(args.limit_queries).copy()
        allowed_qids = set(queries_df["query_id"].astype(str))
        qrels_df = qrels_df[qrels_df["query_id"].astype(str).isin(allowed_qids)].copy()

    print("Queries:", len(queries_df))
    print("Qrels:", len(qrels_df))

    retrievers = {}
    refinement_service = None
    if args.enable_query_refinement:
        print(f"Query Refinement enabled: {args.refinement_method}")
        refinement_service = build_query_refinement_service(args)

    for model_name in args.models:
        normalized_name = model_name.lower().replace("-", "_")
        print(f"Building retriever: {normalized_name}")
        retriever = build_retriever(normalized_name, args)
        if refinement_service is not None:
            retriever = QueryRefinedRetriever(
                base_retriever=retriever,
                refinement_service=refinement_service,
            )
        retrievers[normalized_name] = retriever

    config = EvaluationConfig(
        run_depth=args.run_depth,
        precision_k=args.precision_k,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        map_depth=args.map_depth,
        min_relevance=args.min_relevance,
        continue_on_error=not args.stop_on_error,
        save_charts=not args.no_charts,
        save_excel=not args.no_excel,
    )

    service = EvaluationService(
        output_dir=args.output_dir,
        config=config,
    )

    result = service.evaluate_models(
        retrievers=retrievers,
        queries_df=queries_df,
        qrels_df=qrels_df,
    )

    print("\nEvaluation finished.")
    print("Summary CSV:", result["summary_csv"])
    print("Summary XLSX:", result["summary_xlsx"])
    print("Run files:")
    for model, path in result["run_paths"].items():
        print(f"  - {model}: {path}")


if __name__ == "__main__":
    main()
