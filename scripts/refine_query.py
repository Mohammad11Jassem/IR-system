# import argparse
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.query_refinement import QueryRefinementConfig, QueryRefinementService


# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# def parse_args():
#     parser = argparse.ArgumentParser(description="Inspect Query Refinement output for one query.")
#     parser.add_argument("--query", required=True)
#     parser.add_argument(
#         "--method",
#         choices=["word2vec", "history", "history_word2vec"],
#         default="word2vec",
#     )
#     parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
#     parser.add_argument("--history-file", default=None)
#     parser.add_argument("--max-expansion-terms", type=int, default=5)
#     parser.add_argument("--word2vec-topn", type=int, default=5)
#     parser.add_argument("--min-word2vec-similarity", type=float, default=0.55)
#     parser.add_argument("--history-threshold", type=float, default=0.65)
#     parser.add_argument("--top-history-queries", type=int, default=2)
#     parser.add_argument("--max-history-terms", type=int, default=5)
#     parser.add_argument("--original-term-weight", type=int, default=1)
#     parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     config = QueryRefinementConfig(
#         method=args.method,
#         max_expansion_terms=args.max_expansion_terms,
#         word2vec_topn=args.word2vec_topn,
#         min_word2vec_similarity=args.min_word2vec_similarity,
#         history_similarity_threshold=args.history_threshold,
#         top_history_queries=args.top_history_queries,
#         max_history_terms=args.max_history_terms,
#         original_term_weight=args.original_term_weight,
#     )

#     service = QueryRefinementService(
#         config=config,
#         word2vec_index_dir=args.word2vec_index_dir,
#         history_file=args.history_file,
#         sentence_model_name=args.bert_model_name,
#     )

#     result = service.refine(args.query)

#     print("=" * 80)
#     print("Original Query :", result.original_query)
#     print("Processed Query:", result.processed_query)
#     print("Refined Query  :", result.refined_query)
#     print("Method         :", result.method)
#     print("Changed        :", result.changed)
#     print("=" * 80)
#     print("Original Terms :", result.original_terms)
#     print("History Terms  :", result.history_terms)
#     print("Word2Vec Terms :", result.word2vec_terms)
#     print("Selected History:", result.selected_history)


# if __name__ == "__main__":
#     main()


# import argparse
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.query_refinement import QueryRefinementConfig, QueryRefinementService


# DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
# DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# def parse_args():
#     parser = argparse.ArgumentParser(description="Inspect Query Refinement output for one query.")
#     parser.add_argument("--query", required=True)
#     parser.add_argument(
#         "--method",
#         choices=["word2vec", "history", "history_word2vec"],
#         default="word2vec",
#     )
#     parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
#     parser.add_argument("--history-file", default=None)
#     parser.add_argument("--max-expansion-terms", type=int, default=5)
#     parser.add_argument("--word2vec-topn", type=int, default=5)
#     parser.add_argument("--min-word2vec-similarity", type=float, default=0.55)
#     parser.add_argument("--history-threshold", type=float, default=0.65)
#     parser.add_argument("--top-history-queries", type=int, default=2)
#     parser.add_argument("--max-history-terms", type=int, default=5)
#     parser.add_argument("--original-term-weight", type=int, default=1)
#     parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     config = QueryRefinementConfig(
#         method=args.method,
#         max_expansion_terms=args.max_expansion_terms,
#         word2vec_topn=args.word2vec_topn,
#         min_word2vec_similarity=args.min_word2vec_similarity,
#         history_similarity_threshold=args.history_threshold,
#         top_history_queries=args.top_history_queries,
#         max_history_terms=args.max_history_terms,
#         original_term_weight=args.original_term_weight,
#     )

#     service = QueryRefinementService(
#         config=config,
#         word2vec_index_dir=args.word2vec_index_dir,
#         history_file=args.history_file,
#         sentence_model_name=args.bert_model_name,
#     )

#     result = service.refine(args.query)

#     print("=" * 80)
#     print("Original Query :", result.original_query)
#     print("Processed Query:", result.processed_query)
#     print("Refined Query  :", result.refined_query)
#     print("Method         :", result.method)
#     print("Changed        :", result.changed)
#     print("=" * 80)
#     print("Original Terms :", result.original_terms)
#     print("History Terms  :", result.history_terms)
#     print("Word2Vec Terms :", result.word2vec_terms)
#     print("Selected History:", result.selected_history)


# if __name__ == "__main__":
#     main()






import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_refinement import QueryRefinementConfig, QueryRefinementService


DEFAULT_WORD2VEC_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_word2vec_full"
DEFAULT_BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Query Refinement output for one query. PRF methods need feedback docs, so test them through main.py."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--method",
        choices=[
            "word2vec",
            "context_word2vec",
            "prf",
            "prf_word2vec",
            "history",
            "history_word2vec",
            "prf_history_word2vec",
        ],
        default="word2vec",
    )
    parser.add_argument("--word2vec-index-dir", default=DEFAULT_WORD2VEC_INDEX_DIR)
    parser.add_argument("--history-file", default=None)
    parser.add_argument("--max-expansion-terms", type=int, default=5)
    parser.add_argument("--max-prf-terms", type=int, default=5)
    parser.add_argument("--word2vec-topn", type=int, default=8)
    parser.add_argument("--min-word2vec-similarity", type=float, default=0.60)
    parser.add_argument("--original-term-weight", type=int, default=2)
    parser.add_argument("--bert-model-name", default=DEFAULT_BERT_MODEL_NAME)
    return parser.parse_args()


def main():
    args = parse_args()

    config = QueryRefinementConfig(
        method=args.method,
        max_expansion_terms=args.max_expansion_terms,
        max_prf_terms=args.max_prf_terms,
        word2vec_topn=args.word2vec_topn,
        min_word2vec_similarity=args.min_word2vec_similarity,
        original_term_weight=args.original_term_weight,
    )

    service = QueryRefinementService(
        config=config,
        word2vec_index_dir=args.word2vec_index_dir,
        history_file=args.history_file,
        sentence_model_name=args.bert_model_name,
    )

    result = service.refine(args.query)

    print("=" * 80)
    print("Original Query :", result.original_query)
    print("Processed Query:", result.processed_query)
    print("Refined Query  :", result.refined_query)
    print("Method         :", result.method)
    print("Changed        :", result.changed)
    print("=" * 80)
    print("Original Terms :", result.original_terms)
    print("History Terms  :", result.history_terms)
    print("PRF Terms      :", result.prf_terms)
    print("Word2Vec Terms :", result.word2vec_terms)
    print("Selected History:", result.selected_history)

    if args.method in {"prf", "context_word2vec", "prf_word2vec", "prf_history_word2vec"}:
        print("\nNote: This standalone script does not run an initial BM25 search, so PRF/context terms may be empty.")
        print("Use main.py with --enable-query-refinement to test PRF methods with real feedback documents.")


if __name__ == "__main__":
    main()
