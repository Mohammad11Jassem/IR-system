"""
Official BM25 retriever.

This class uses Terrier/PyTerrier over the built inverted index.
The old manual BM25 implementation is kept only for educational purposes
in bm25_engine.py, but it is not the official retrieval path.
"""

from src.retrieval.terrier_retriever import TerrierRetriever


class BM25Retriever:
    def __init__(
        self,
        index_path: str,
        db_path: str,
        top_k: int = 10,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        self.retriever = TerrierRetriever(
            index_path=index_path,
            db_path=db_path,
            model="bm25",
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    def search(self, query: str) -> dict:
        return self.retriever.search(query)