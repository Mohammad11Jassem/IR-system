import re
import time
from pathlib import Path

from src.embeddings import FaissVectorStore, Word2VecVectorizer
from src.preprocessing import EmbeddingTextCleaner
from src.storage.document_store import DocumentStore


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")


def tokenize_for_word2vec(text: str) -> list[str]:
    if not text:
        return []

    text = text.lower()
    return TOKEN_PATTERN.findall(text)


class Word2VecRetriever:
    """
    Official Word2Vec dense retriever.

    Retrieval path:
    Query
      -> light embedding text cleaning
      -> biomedical-friendly tokenization
      -> average Word2Vec word vectors
      -> FAISS vector search
      -> DocumentStore fetch from SQLite
    """

    def __init__(
        self,
        index_dir: str,
        db_path: str,
        top_k: int = 10,
    ):
        self.index_dir = index_dir
        self.db_path = db_path
        self.top_k = top_k

        index_dir_path = Path(index_dir)
        model_path = index_dir_path / "word2vec.model"

        if not model_path.exists():
            raise FileNotFoundError(f"Word2Vec model not found: {model_path}")

        self.cleaner = EmbeddingTextCleaner(max_chars=None)
        self.vectorizer = Word2VecVectorizer.load(str(model_path))
        self.vector_store = FaissVectorStore.load(index_dir)
        self.document_store = DocumentStore(db_path)

    def search(self, query: str) -> dict:
        start = time.time()

        original_query = query
        processed_query = self.cleaner.clean(query)
        tokens = tokenize_for_word2vec(processed_query)

        if not tokens:
            return {
                "query": original_query,
                "processed_query": processed_query,
                "tokens": tokens,
                "model": "WORD2VEC",
                "time_seconds": time.time() - start,
                "results": [],
            }

        query_vector = self.vectorizer.encode_tokens(tokens)

        if not query_vector.any():
            return {
                "query": original_query,
                "processed_query": processed_query,
                "tokens": tokens,
                "model": "WORD2VEC",
                "time_seconds": time.time() - start,
                "results": [],
            }

        hits = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k,
        )

        doc_ids = [doc_id for doc_id, _ in hits]
        docs = self.document_store.get_by_ids(doc_ids)
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
            "query": original_query,
            "processed_query": processed_query,
            "tokens": tokens,
            "model": "WORD2VEC",
            "time_seconds": time.time() - start,
            "results": results,
        }