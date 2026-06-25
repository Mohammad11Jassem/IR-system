import time

from src.embeddings import BertVectorizer, FaissVectorStore
from src.preprocessing import EmbeddingTextCleaner
from src.storage.document_store import DocumentStore


class BertRetriever:
    """
    Official BERT dense retriever.

    Retrieval path:
    Query
      -> light embedding text cleaning
      -> SentenceTransformer embedding
      -> FAISS vector search
      -> DocumentStore fetch from SQLite
    """

    def __init__(
        self,
        index_dir: str,
        db_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
    ):
        self.index_dir = index_dir
        self.db_path = db_path
        self.model_name = model_name
        self.top_k = top_k

        self.cleaner = EmbeddingTextCleaner(max_chars=None)

        self.vectorizer = BertVectorizer(
            model_name=model_name,
            normalize_embeddings=True,
        )

        self.vector_store = FaissVectorStore.load(index_dir)
        self.document_store = DocumentStore(db_path)

    def search(self, query: str) -> dict:
        start = time.time()

        original_query = query
        processed_query = self.cleaner.clean(query)

        if not processed_query:
            return {
                "query": original_query,
                "processed_query": processed_query,
                "model": "BERT",
                "time_seconds": time.time() - start,
                "results": [],
            }

        query_vector = self.vectorizer.encode_query(processed_query)

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
            "model": "BERT",
            "time_seconds": time.time() - start,
            "results": results,
        }