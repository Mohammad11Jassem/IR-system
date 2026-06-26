"""
Official TF-IDF retriever.

This class uses Terrier/PyTerrier over the built inverted index.
The old manual TF-IDF implementation is kept only for educational purposes
in tfidf_engine.py, but it is not the official retrieval path.
"""

from src.retrieval.terrier_retriever import TerrierRetriever


class TfidfRetriever:
    def __init__(
        self,
        index_path: str,
        db_path: str,
        top_k: int = 10,
    ):
        self.retriever = TerrierRetriever(
            index_path=index_path,
            db_path=db_path,
            model="tfidf",
            top_k=top_k,
        )

    def search(self, query: str) -> dict:
        return self.retriever.search(query)