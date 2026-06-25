import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import FaissVectorStore, Word2VecVectorizer
from src.preprocessing import EmbeddingTextCleaner
from src.storage.document_store import DocumentStore


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")


def tokenize_for_word2vec(text: str) -> list[str]:
    """
    Same tokenizer used while building the Word2Vec index.
    """
    if not text:
        return []

    text = text.lower()
    return TOKEN_PATTERN.findall(text)


def print_results(output: dict) -> None:
    print("=" * 80)
    print(f"Query          : {output['query']}")
    print(f"Processed Query: {output['processed_query']}")
    print(f"Tokens         : {output['tokens']}")
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


def search_word2vec(
    query: str,
    index_dir: str,
    db_path: str,
    top_k: int,
) -> dict:
    start = time.time()

    index_dir_path = Path(index_dir)
    model_path = index_dir_path / "word2vec.model"

    if not model_path.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {model_path}")

    cleaner = EmbeddingTextCleaner(max_chars=None)
    processed_query = cleaner.clean(query)
    tokens = tokenize_for_word2vec(processed_query)

    if not tokens:
        return {
            "query": query,
            "processed_query": processed_query,
            "tokens": tokens,
            "model": "WORD2VEC",
            "time_seconds": time.time() - start,
            "results": [],
        }

    vectorizer = Word2VecVectorizer.load(str(model_path))
    vector_store = FaissVectorStore.load(index_dir)
    document_store = DocumentStore(db_path)

    query_vector = vectorizer.encode_tokens(tokens)

    if not query_vector.any():
        return {
            "query": query,
            "processed_query": processed_query,
            "tokens": tokens,
            "model": "WORD2VEC",
            "time_seconds": time.time() - start,
            "results": [],
        }

    hits = vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
    )

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
        "tokens": tokens,
        "model": "WORD2VEC",
        "time_seconds": time.time() - start,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Search using Word2Vec FAISS index."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Search query.",
    )

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Path to Word2Vec FAISS index directory.",
    )

    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to documents.sqlite.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of returned results.",
    )

    args = parser.parse_args()

    output = search_word2vec(
        query=args.query,
        index_dir=args.index_dir,
        db_path=args.db_path,
        top_k=args.top_k,
    )

    print_results(output)


if __name__ == "__main__":
    main()