import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.retrieval.terrier_retriever import TerrierRetriever


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["bm25", "tfidf", "tf-idf"],
        required=True,
        help="Retrieval model"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="User query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results"
    )

    parser.add_argument(
        "--index-path",
        default=r"E:\ir_project_artifacts\indexes\terrier_medline",
        help="Terrier index directory"
    )

    parser.add_argument(
        "--db-path",
        default=r"E:\ir_project_artifacts\documents.sqlite",
        help="SQLite document store path"
    )

    args = parser.parse_args()

    retriever = TerrierRetriever(
        index_path=args.index_path,
        db_path=args.db_path,
        model=args.model,
        top_k=args.top_k,
    )

    output = retriever.search(args.query)

    print("=" * 80)
    print(f"Query : {output['query']}")
    print(f"Model : {output['model']}")
    print(f"Time  : {output['time_seconds']:.4f} seconds")
    print("=" * 80)

    for item in output["results"]:
        print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")
        print(f"Title: {item['title']}")
        abstract = item["abstract"] or ""
        print(f"Abstract: {abstract[:500]}")
        print("-" * 80)


if __name__ == "__main__":
    main()