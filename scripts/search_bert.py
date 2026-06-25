import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import BertVectorizer, FaissVectorStore
from src.preprocessing import QueryProcessor
from src.storage.document_store import DocumentStore
from src.preprocessing import EmbeddingTextCleaner

def print_results(output: dict) -> None:
    print("=" * 80)
    print(f"Query          : {output['query']}")
    print(f"Processed Query: {output['processed_query']}")
    print(f"Model          : {output['model']}")
    print(f"Time           : {output['time_seconds']:.4f} seconds")
    print("=" * 80)

    if not output["results"]:
        print("No results found.")
        return

    for item in output["results"]:
        print(f"[{item['rank']}] doc_id={item['doc_id']} | score={item['score']:.4f}")
        print(f"Title: {item.get('title') or ''}")

        abstract = item.get("abstract") or ""
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."

        print(f"Abstract: {abstract}")
        print("-" * 80)


def search_bert(
    query: str,
    index_dir: str,
    db_path: str,
    model_name: str,
    top_k: int,
) -> dict:
    start = time.time()

    cleaner = EmbeddingTextCleaner(max_chars=None)
    processed_query = cleaner.clean(query)

    if not processed_query:
        return {
            "query": query,
            "processed_query": processed_query,
            "model": "BERT",
            "time_seconds": time.time() - start,
            "results": [],
        }

    vectorizer = BertVectorizer(
        model_name=model_name,
        normalize_embeddings=True,
    )

    store = FaissVectorStore.load(index_dir)
    document_store = DocumentStore(db_path)

    query_vector = vectorizer.encode_query(processed_query)
    hits = store.search(query_vector=query_vector, top_k=top_k)

    doc_ids = [doc_id for doc_id, _ in hits]
    docs = document_store.get_by_ids(doc_ids)
    docs_by_id = {str(doc["doc_id"]): doc for doc in docs}

    results = []

    for rank, (doc_id, score) in enumerate(hits, start=1):
        doc = docs_by_id.get(str(doc_id))

        if doc is None:
            continue

        results.append(
            {
                "rank": rank,
                "doc_id": str(doc_id),
                "score": float(score),
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
            }
        )

    return {
        "query": query,
        "processed_query": processed_query,
        "model": "BERT",
        "time_seconds": time.time() - start,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="Search query",
    )

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Path to BERT FAISS index directory",
    )

    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to documents.sqlite",
    )

    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of returned results",
    )

    args = parser.parse_args()

    output = search_bert(
        query=args.query,
        index_dir=args.index_dir,
        db_path=args.db_path,
        model_name=args.model_name,
        top_k=args.top_k,
    )

    print_results(output)


if __name__ == "__main__":
    main()