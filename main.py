# import argparse
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.retrieval import (
#     BM25Retriever,
#     TfidfRetriever,
#     BertRetriever,
#     Word2VecRetriever,
#     SerialHybridRetriever,
#     ParallelHybridRetriever,
# )


# DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
# DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"

# DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"


# def build_retriever(
#     model: str,
#     index_path: str,
#     db_path: str,
#     top_k: int,
#     bert_index_dir: str,
#     bert_model_name: str,
#     word2vec_index_dir: str,
#     first_stage: str,
#     second_stage: str,
#     candidate_k: int,
#     parallel_models: list[str],
#     fusion_method: str,
#     per_model_k: int,
# ):
#     model = model.lower().strip()

#     if model == "bm25":
#         return BM25Retriever(
#             index_path=index_path,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model in {"tfidf", "tf-idf", "tf_idf"}:
#         return TfidfRetriever(
#             index_path=index_path,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model == "bert":
#         return BertRetriever(
#             index_dir=bert_index_dir,
#             db_path=db_path,
#             model_name=bert_model_name,
#             top_k=top_k,
#         )

#     if model in {"word2vec", "w2v"}:
#         return Word2VecRetriever(
#             index_dir=word2vec_index_dir,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model in {"serial", "serial-hybrid", "serial_hybrid"}:
#         return SerialHybridRetriever(
#             first_stage=first_stage,
#             second_stage=second_stage,
#             index_path=index_path,
#             db_path=db_path,
#             word2vec_index_dir=word2vec_index_dir,
#             bert_model_name=bert_model_name,
#             candidate_k=candidate_k,
#             top_k=top_k,
#         )

#     if model in {"parallel", "parallel-hybrid", "parallel_hybrid"}:
#         return ParallelHybridRetriever(
#             models=parallel_models,
#             index_path=index_path,
#             db_path=db_path,
#             word2vec_index_dir=word2vec_index_dir,
#             bert_index_dir=bert_index_dir,
#             bert_model_name=bert_model_name,
#             fusion_method=fusion_method,
#             per_model_k=per_model_k,
#             top_k=top_k,
#         )

#     raise ValueError(f"Unsupported model: {model}")

# def print_results(output: dict):
#     print("=" * 80)
#     print(f"Query : {output['query']}")

#     if "processed_query" in output:
#         print(f"Processed Query: {output['processed_query']}")

#     if "tokens" in output:
#         print(f"Tokens: {output['tokens']}")

#     print(f"Model : {output['model']}")

#     if output.get("model") == "SERIAL_HYBRID":
#         print(f"First Stage : {output.get('first_stage')}")
#         print(f"Second Stage: {output.get('second_stage')}")
#         print(f"Candidate K : {output.get('candidate_k')}")
#         print(f"Top K       : {output.get('top_k')}")

#     if output.get("model") == "PARALLEL_HYBRID":
#         print(f"Models        : {output.get('models')}")
#         print(f"Fusion Method : {output.get('fusion_method')}")
#         print(f"Per Model K   : {output.get('per_model_k')}")
#         print(f"Top K         : {output.get('top_k')}")
#         print(f"RRF K         : {output.get('rrf_k')}")

#     print(f"Time  : {output['time_seconds']:.4f} seconds")
#     print("=" * 80)

#     results = output.get("results", [])

#     if not results:
#         print("No results found.")
#         return

#     # for item in results:
#     #     print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")

#     #     if "first_stage_rank" in item:
#     #         print(
#     #             f"First-stage rank={item['first_stage_rank']} "
#     #             f"| first-stage score={item['first_stage_score']:.4f} "
#     #             f"| rerank score={item['rerank_score']:.4f}"
#     #         )

#     #     print(f"Title: {item.get('title') or ''}")

#     #     abstract = item.get("abstract") or ""
#     #     if len(abstract) > 500:
#     #         abstract = abstract[:500] + "..."

#     #     print(f"Abstract: {abstract}")
#     #     print("-" * 80)


#     #     if "model_contributions" in item:
#     #         print("Model contributions:")

#     #         for model_name, contribution in item["model_contributions"].items():
#     #             print(
#     #                 f"  - {model_name}: "
#     #                 f"rank={contribution['rank']} "
#     #                 f"| score={contribution['score']:.4f} "
#     #                 f"| rrf={contribution['rrf_contribution']:.6f}"
#     #             )
                
#     for item in results:
#         print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")

#         if "first_stage_rank" in item:
#             print(
#                 f"First-stage rank={item['first_stage_rank']} "
#                 f"| first-stage score={item['first_stage_score']:.4f} "
#                 f"| rerank score={item['rerank_score']:.4f}"
#             )

#         if "model_contributions" in item:
#             print("Model contributions:")

#             for model_name, contribution in item["model_contributions"].items():
#                 print(
#                     f"  - {model_name}: "
#                     f"rank={contribution['rank']} "
#                     f"| score={contribution['score']:.4f} "
#                     f"| rrf={contribution['rrf_contribution']:.6f}"
#                 )

#         print(f"Title: {item.get('title') or ''}")

#         abstract = item.get("abstract") or ""
#         if len(abstract) > 500:
#             abstract = abstract[:500] + "..."

#         print(f"Abstract: {abstract}")
#         print("-" * 80)
                    
                
                
                
                
# def main():
#     parser = argparse.ArgumentParser(
#         description="Official IR Search CLI using single and hybrid retrieval models."
#     )

#     parser.add_argument(
#         "--model",
#         choices=[
#             "bm25",
#             "tfidf",
#             "tf-idf",
#             "bert",
#             "word2vec",
#             "w2v",
#             "serial",
#             "serial-hybrid",
#             "serial_hybrid",
#             "parallel",
#             "parallel-hybrid",
#             "parallel_hybrid",
#         ],
#         default="bm25",
#         help="Retrieval model or retrieval mode.",
#     )

#     parser.add_argument(
#         "--query",
#         required=True,
#         help="Search query.",
#     )

#     parser.add_argument(
#         "--top-k",
#         type=int,
#         default=10,
#         help="Number of returned final results.",
#     )

#     parser.add_argument(
#         "--candidate-k",
#         type=int,
#         default=100,
#         help="Number of candidates retrieved by first-stage model in Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--first-stage",
#         choices=["bm25", "tfidf", "word2vec"],
#         default="bm25",
#         help="First-stage retriever for Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--second-stage",
#         choices=["bert", "word2vec"],
#         default="bert",
#         help="Second-stage reranker for Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--index-path",
#         default=DEFAULT_TERRIER_INDEX_PATH,
#         help="Path to Terrier index directory. Used by BM25 and TF-IDF.",
#     )

#     parser.add_argument(
#         "--bert-index-dir",
#         default=DEFAULT_BERT_INDEX_DIR,
#         help="Path to BERT FAISS index directory. Used by standalone BERT and Parallel Hybrid when BERT is selected.",
#     )

#     parser.add_argument(
#         "--bert-model-name",
#         default=DEFAULT_BERT_MODEL_NAME,
#         help="SentenceTransformer model name. Used by BERT standalone and BERT reranker.",
#     )

#     parser.add_argument(
#         "--word2vec-index-dir",
#         default=DEFAULT_WORD2VEC_INDEX_DIR,
#         help="Path to Word2Vec FAISS index directory and trained Word2Vec model.",
#     )

#     parser.add_argument(
#         "--db-path",
#         default=DEFAULT_DB_PATH,
#         help="Path to SQLite document store.",
#     )
    
#     parser.add_argument(
#     "--parallel-models",
#     nargs="+",
#     choices=["bm25", "tfidf", "tf-idf", "word2vec", "w2v", "bert"],
#     default=["bm25", "tfidf", "word2vec"],
#     help="Models used in Parallel Hybrid.",
#     )

#     parser.add_argument(
#         "--fusion-method",
#         choices=["rrf"],
#         default="rrf",
#         help="Fusion method for Parallel Hybrid.",
#     )

#     parser.add_argument(
#         "--per-model-k",
#         type=int,
#         default=100,
#         help="Number of results retrieved independently by each model in Parallel Hybrid.",
#     )

#     args = parser.parse_args()


#     retriever = build_retriever(
#         model=args.model,
#         index_path=args.index_path,
#         db_path=args.db_path,
#         top_k=args.top_k,
#         bert_index_dir=args.bert_index_dir,
#         bert_model_name=args.bert_model_name,
#         word2vec_index_dir=args.word2vec_index_dir,
#         first_stage=args.first_stage,
#         second_stage=args.second_stage,
#         candidate_k=args.candidate_k,
#         parallel_models=args.parallel_models,
#         fusion_method=args.fusion_method,
#         per_model_k=args.per_model_k,
#     )

#     output = retriever.search(args.query)
#     print_results(output)


# if __name__ == "__main__":
#     main()





# import argparse
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.retrieval import (
#     BM25Retriever,
#     TfidfRetriever,
#     BertRetriever,
#     Word2VecRetriever,
#     SerialHybridRetriever,
#     ParallelHybridRetriever,
# )
# from src.query_refinement import (
#     QueryRefinedRetriever,
#     QueryRefinementConfig,
#     QueryRefinementService,
# )


# DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
# DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"

# DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"



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


# def build_retriever(
#     model: str,
#     index_path: str,
#     db_path: str,
#     top_k: int,
#     bert_index_dir: str,
#     bert_model_name: str,
#     word2vec_index_dir: str,
#     first_stage: str,
#     second_stage: str,
#     candidate_k: int,
#     parallel_models: list[str],
#     fusion_method: str,
#     per_model_k: int,
# ):
#     model = model.lower().strip()

#     if model == "bm25":
#         return BM25Retriever(
#             index_path=index_path,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model in {"tfidf", "tf-idf", "tf_idf"}:
#         return TfidfRetriever(
#             index_path=index_path,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model == "bert":
#         return BertRetriever(
#             index_dir=bert_index_dir,
#             db_path=db_path,
#             model_name=bert_model_name,
#             top_k=top_k,
#         )

#     if model in {"word2vec", "w2v"}:
#         return Word2VecRetriever(
#             index_dir=word2vec_index_dir,
#             db_path=db_path,
#             top_k=top_k,
#         )

#     if model in {"serial", "serial-hybrid", "serial_hybrid"}:
#         return SerialHybridRetriever(
#             first_stage=first_stage,
#             second_stage=second_stage,
#             index_path=index_path,
#             db_path=db_path,
#             word2vec_index_dir=word2vec_index_dir,
#             bert_model_name=bert_model_name,
#             candidate_k=candidate_k,
#             top_k=top_k,
#         )

#     if model in {"parallel", "parallel-hybrid", "parallel_hybrid"}:
#         return ParallelHybridRetriever(
#             models=parallel_models,
#             index_path=index_path,
#             db_path=db_path,
#             word2vec_index_dir=word2vec_index_dir,
#             bert_index_dir=bert_index_dir,
#             bert_model_name=bert_model_name,
#             fusion_method=fusion_method,
#             per_model_k=per_model_k,
#             top_k=top_k,
#         )

#     raise ValueError(f"Unsupported model: {model}")

# def print_results(output: dict):
#     print("=" * 80)
#     print(f"Query : {output['query']}")

#     if "processed_query" in output:
#         print(f"Processed Query: {output['processed_query']}")

#     if "refined_query" in output:
#         print(f"Refined Query  : {output['refined_query']}")

#     if output.get("refinement"):
#         refinement = output["refinement"]
#         print(f"Refinement    : {refinement.get('method')} | changed={refinement.get('changed')}")
#         print(f"History Terms : {refinement.get('history_terms', [])}")
#         print(f"W2V Terms     : {refinement.get('word2vec_terms', [])}")

#     if "tokens" in output:
#         print(f"Tokens: {output['tokens']}")

#     print(f"Model : {output['model']}")

#     if output.get("model") == "SERIAL_HYBRID":
#         print(f"First Stage : {output.get('first_stage')}")
#         print(f"Second Stage: {output.get('second_stage')}")
#         print(f"Candidate K : {output.get('candidate_k')}")
#         print(f"Top K       : {output.get('top_k')}")

#     if output.get("model") == "PARALLEL_HYBRID":
#         print(f"Models        : {output.get('models')}")
#         print(f"Fusion Method : {output.get('fusion_method')}")
#         print(f"Per Model K   : {output.get('per_model_k')}")
#         print(f"Top K         : {output.get('top_k')}")
#         print(f"RRF K         : {output.get('rrf_k')}")

#     print(f"Time  : {output['time_seconds']:.4f} seconds")
#     print("=" * 80)

#     results = output.get("results", [])

#     if not results:
#         print("No results found.")
#         return

#     # for item in results:
#     #     print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")

#     #     if "first_stage_rank" in item:
#     #         print(
#     #             f"First-stage rank={item['first_stage_rank']} "
#     #             f"| first-stage score={item['first_stage_score']:.4f} "
#     #             f"| rerank score={item['rerank_score']:.4f}"
#     #         )

#     #     print(f"Title: {item.get('title') or ''}")

#     #     abstract = item.get("abstract") or ""
#     #     if len(abstract) > 500:
#     #         abstract = abstract[:500] + "..."

#     #     print(f"Abstract: {abstract}")
#     #     print("-" * 80)


#     #     if "model_contributions" in item:
#     #         print("Model contributions:")

#     #         for model_name, contribution in item["model_contributions"].items():
#     #             print(
#     #                 f"  - {model_name}: "
#     #                 f"rank={contribution['rank']} "
#     #                 f"| score={contribution['score']:.4f} "
#     #                 f"| rrf={contribution['rrf_contribution']:.6f}"
#     #             )
                
#     for item in results:
#         print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")

#         if "first_stage_rank" in item:
#             print(
#                 f"First-stage rank={item['first_stage_rank']} "
#                 f"| first-stage score={item['first_stage_score']:.4f} "
#                 f"| rerank score={item['rerank_score']:.4f}"
#             )

#         if "model_contributions" in item:
#             print("Model contributions:")

#             for model_name, contribution in item["model_contributions"].items():
#                 print(
#                     f"  - {model_name}: "
#                     f"rank={contribution['rank']} "
#                     f"| score={contribution['score']:.4f} "
#                     f"| rrf={contribution['rrf_contribution']:.6f}"
#                 )

#         print(f"Title: {item.get('title') or ''}")

#         abstract = item.get("abstract") or ""
#         if len(abstract) > 500:
#             abstract = abstract[:500] + "..."

#         print(f"Abstract: {abstract}")
#         print("-" * 80)
                    
                
                
                
                
# def main():
#     parser = argparse.ArgumentParser(
#         description="Official IR Search CLI using single and hybrid retrieval models."
#     )

#     parser.add_argument(
#         "--model",
#         choices=[
#             "bm25",
#             "tfidf",
#             "tf-idf",
#             "bert",
#             "word2vec",
#             "w2v",
#             "serial",
#             "serial-hybrid",
#             "serial_hybrid",
#             "parallel",
#             "parallel-hybrid",
#             "parallel_hybrid",
#         ],
#         default="bm25",
#         help="Retrieval model or retrieval mode.",
#     )

#     parser.add_argument(
#         "--query",
#         required=True,
#         help="Search query.",
#     )

#     parser.add_argument(
#         "--top-k",
#         type=int,
#         default=10,
#         help="Number of returned final results.",
#     )

#     parser.add_argument(
#         "--candidate-k",
#         type=int,
#         default=100,
#         help="Number of candidates retrieved by first-stage model in Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--first-stage",
#         choices=["bm25", "tfidf", "word2vec"],
#         default="bm25",
#         help="First-stage retriever for Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--second-stage",
#         choices=["bert", "word2vec"],
#         default="bert",
#         help="Second-stage reranker for Serial Hybrid.",
#     )

#     parser.add_argument(
#         "--index-path",
#         default=DEFAULT_TERRIER_INDEX_PATH,
#         help="Path to Terrier index directory. Used by BM25 and TF-IDF.",
#     )

#     parser.add_argument(
#         "--bert-index-dir",
#         default=DEFAULT_BERT_INDEX_DIR,
#         help="Path to BERT FAISS index directory. Used by standalone BERT and Parallel Hybrid when BERT is selected.",
#     )

#     parser.add_argument(
#         "--bert-model-name",
#         default=DEFAULT_BERT_MODEL_NAME,
#         help="SentenceTransformer model name. Used by BERT standalone and BERT reranker.",
#     )

#     parser.add_argument(
#         "--word2vec-index-dir",
#         default=DEFAULT_WORD2VEC_INDEX_DIR,
#         help="Path to Word2Vec FAISS index directory and trained Word2Vec model.",
#     )

#     parser.add_argument(
#         "--db-path",
#         default=DEFAULT_DB_PATH,
#         help="Path to SQLite document store.",
#     )
    
#     parser.add_argument(
#     "--parallel-models",
#     nargs="+",
#     choices=["bm25", "tfidf", "tf-idf", "word2vec", "w2v", "bert"],
#     default=["bm25", "tfidf", "word2vec"],
#     help="Models used in Parallel Hybrid.",
#     )

#     parser.add_argument(
#         "--fusion-method",
#         choices=["rrf"],
#         default="rrf",
#         help="Fusion method for Parallel Hybrid.",
#     )

#     parser.add_argument(
#         "--enable-query-refinement",
#         action="store_true",
#         help="Apply Query Refinement before retrieval.",
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
#         help="Optional .txt or .json file containing previous user queries, one query per line or JSON list.",
#     )

#     parser.add_argument("--refinement-max-expansion-terms", type=int, default=5)
#     parser.add_argument("--refinement-word2vec-topn", type=int, default=5)
#     parser.add_argument("--refinement-min-word2vec-similarity", type=float, default=0.55)
#     parser.add_argument("--refinement-history-threshold", type=float, default=0.65)
#     parser.add_argument("--refinement-top-history-queries", type=int, default=2)
#     parser.add_argument("--refinement-max-history-terms", type=int, default=5)
#     parser.add_argument("--refinement-original-term-weight", type=int, default=1)
#     parser.add_argument(
#         "--per-model-k",
#         type=int,
#         default=100,
#         help="Number of results retrieved independently by each model in Parallel Hybrid.",
#     )

#     args = parser.parse_args()


#     retriever = build_retriever(
#         model=args.model,
#         index_path=args.index_path,
#         db_path=args.db_path,
#         top_k=args.top_k,
#         bert_index_dir=args.bert_index_dir,
#         bert_model_name=args.bert_model_name,
#         word2vec_index_dir=args.word2vec_index_dir,
#         first_stage=args.first_stage,
#         second_stage=args.second_stage,
#         candidate_k=args.candidate_k,
#         parallel_models=args.parallel_models,
#         fusion_method=args.fusion_method,
#         per_model_k=args.per_model_k,
#     )

#     if args.enable_query_refinement:
#         refinement_service = build_query_refinement_service(args)
#         retriever = QueryRefinedRetriever(
#             base_retriever=retriever,
#             refinement_service=refinement_service,
#         )

#     output = retriever.search(args.query)
#     print_results(output)


# if __name__ == "__main__":
#     main()








import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (
    BM25Retriever,
    TfidfRetriever,
    BertRetriever,
    Word2VecRetriever,
    SerialHybridRetriever,
    ParallelHybridRetriever,
)
from src.query_refinement import (
    QueryRefinedRetriever,
    QueryRefinementConfig,
    QueryRefinementService,
)


DEFAULT_TERRIER_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_medline"
DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
DEFAULT_BERT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"


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


def build_retriever(
    model: str,
    index_path: str,
    db_path: str,
    top_k: int,
    bert_index_dir: str,
    bert_model_name: str,
    word2vec_index_dir: str,
    first_stage: str,
    second_stage: str,
    candidate_k: int,
    parallel_models: list[str],
    fusion_method: str,
    per_model_k: int,
):
    model = model.lower().strip()

    if model == "bm25":
        return BM25Retriever(index_path=index_path, db_path=db_path, top_k=top_k)

    if model in {"tfidf", "tf-idf", "tf_idf"}:
        return TfidfRetriever(index_path=index_path, db_path=db_path, top_k=top_k)

    if model == "bert":
        return BertRetriever(
            index_dir=bert_index_dir,
            db_path=db_path,
            model_name=bert_model_name,
            top_k=top_k,
        )

    if model in {"word2vec", "w2v"}:
        return Word2VecRetriever(index_dir=word2vec_index_dir, db_path=db_path, top_k=top_k)

    if model in {"serial", "serial-hybrid", "serial_hybrid"}:
        return SerialHybridRetriever(
            first_stage=first_stage,
            second_stage=second_stage,
            index_path=index_path,
            db_path=db_path,
            word2vec_index_dir=word2vec_index_dir,
            bert_model_name=bert_model_name,
            candidate_k=candidate_k,
            top_k=top_k,
        )

    if model in {"parallel", "parallel-hybrid", "parallel_hybrid"}:
        return ParallelHybridRetriever(
            models=parallel_models,
            index_path=index_path,
            db_path=db_path,
            word2vec_index_dir=word2vec_index_dir,
            bert_index_dir=bert_index_dir,
            bert_model_name=bert_model_name,
            fusion_method=fusion_method,
            per_model_k=per_model_k,
            top_k=top_k,
        )

    raise ValueError(f"Unsupported model: {model}")


def print_results(output: dict):
    print("=" * 80)
    print(f"Query : {output['query']}")

    if "processed_query" in output:
        print(f"Processed Query: {output['processed_query']}")

    if "refined_query" in output:
        print(f"Refined Query  : {output['refined_query']}")

    if output.get("refinement"):
        refinement = output["refinement"]
        print(f"Refinement    : {refinement.get('method')} | changed={refinement.get('changed')}")
        print(f"History Terms : {refinement.get('history_terms', [])}")
        print(f"PRF Terms     : {refinement.get('prf_terms', [])}")
        print(f"W2V Terms     : {refinement.get('word2vec_terms', [])}")

    if output.get("refinement_safety"):
        safety = output["refinement_safety"]
        print(
            "Safety Gate   : "
            f"accepted={safety.get('accepted')} | "
            f"reason={safety.get('reason')} | "
            f"overlap={safety.get('overlap_ratio')}"
        )

    if "tokens" in output:
        print(f"Tokens: {output['tokens']}")

    print(f"Model : {output['model']}")

    if output.get("model") == "SERIAL_HYBRID":
        print(f"First Stage : {output.get('first_stage')}")
        print(f"Second Stage: {output.get('second_stage')}")
        print(f"Candidate K : {output.get('candidate_k')}")
        print(f"Top K       : {output.get('top_k')}")

    if output.get("model") == "PARALLEL_HYBRID":
        print(f"Models        : {output.get('models')}")
        print(f"Fusion Method : {output.get('fusion_method')}")
        print(f"Per Model K   : {output.get('per_model_k')}")
        print(f"Top K         : {output.get('top_k')}")
        print(f"RRF K         : {output.get('rrf_k')}")

    print(f"Time  : {output['time_seconds']:.4f} seconds")
    print("=" * 80)

    results = output.get("results", [])
    if not results:
        print("No results found.")
        return

    for item in results:
        print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")

        if "first_stage_rank" in item:
            print(
                f"First-stage rank={item['first_stage_rank']} "
                f"| first-stage score={item['first_stage_score']:.4f} "
                f"| rerank score={item['rerank_score']:.4f}"
            )

        if "model_contributions" in item:
            print("Model contributions:")
            for model_name, contribution in item["model_contributions"].items():
                print(
                    f"  - {model_name}: "
                    f"rank={contribution['rank']} "
                    f"| score={contribution['score']:.4f} "
                    f"| rrf={contribution['rrf_contribution']:.6f}"
                )

        print(f"Title: {item.get('title') or ''}")
        abstract = item.get("abstract") or ""
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        print(f"Abstract: {abstract}")
        print("-" * 80)


def add_refinement_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enable-query-refinement",
        action="store_true",
        help="Apply Query Refinement before retrieval.",
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
    parser.add_argument("--query-history-file", default=None)
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


def main():
    parser = argparse.ArgumentParser(
        description="Official IR Search CLI using single and hybrid retrieval models."
    )

    parser.add_argument(
        "--model",
        choices=[
            "bm25", "tfidf", "tf-idf", "bert", "word2vec", "w2v",
            "serial", "serial-hybrid", "serial_hybrid",
            "parallel", "parallel-hybrid", "parallel_hybrid",
        ],
        default="bm25",
        help="Retrieval model or retrieval mode.",
    )
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of returned final results.")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--first-stage", choices=["bm25", "tfidf", "word2vec"], default="bm25")
    parser.add_argument("--second-stage", choices=["bert", "word2vec"], default="bert")
    parser.add_argument("--parallel-models", nargs="+", choices=["bm25", "tfidf", "tf-idf", "word2vec", "w2v", "bert"], default=["bm25", "tfidf", "word2vec"])
    parser.add_argument("--fusion-method", choices=["rrf"], default="rrf")
    parser.add_argument("--per-model-k", type=int, default=100)
    parser.add_argument("--index-path", default=DEFAULT_TERRIER_INDEX_PATH)
    parser.add_argument("--bert-index-dir", default=DEFAULT_BERT_INDEX_DIR)
    parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)
    parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)

    add_refinement_args(parser)
    args = parser.parse_args()

    retriever = build_retriever(
        model=args.model,
        index_path=args.index_path,
        db_path=args.db_path,
        top_k=args.top_k,
        bert_index_dir=args.bert_index_dir,
        bert_model_name=args.bert_model_name,
        word2vec_index_dir=args.word2vec_index_dir,
        first_stage=args.first_stage,
        second_stage=args.second_stage,
        candidate_k=args.candidate_k,
        parallel_models=args.parallel_models,
        fusion_method=args.fusion_method,
        per_model_k=args.per_model_k,
    )

    if args.enable_query_refinement:
        refinement_service = build_query_refinement_service(args)
        retriever = QueryRefinedRetriever(
            base_retriever=retriever,
            refinement_service=refinement_service,
        )

    output = retriever.search(args.query)
    print_results(output)


if __name__ == "__main__":
    main()
