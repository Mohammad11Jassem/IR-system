import time
from pathlib import Path
from typing import Literal

import pandas as pd
import pyterrier as pt

from src.indexing.terrier_index import init_pyterrier
from src.preprocessing import QueryProcessor
from src.storage.document_store import DocumentStore

from src.preprocessing import DocumentProcessor


ModelName = Literal["bm25", "tfidf", "tf-idf"]


def resolve_index_ref(index_path: str) -> str:
    """
    Terrier can load from data.properties.
    If the user gives the index directory, we resolve it.
    """
    path = Path(index_path)

    if path.is_dir():
        data_properties = path / "data.properties"
        if data_properties.exists():
            return str(data_properties)

    return str(path)


def normalize_model_name(model: str) -> str:
    model = model.lower().strip()

    if model == "bm25":
        return "BM25"

    if model in {"tfidf", "tf-idf", "tf_idf"}:
        return "TF_IDF"

    raise ValueError(f"Unsupported Terrier model: {model}")


class TerrierRetriever:
    """
    Official Terrier/PyTerrier retriever.

    It runs BM25 or TF-IDF over the Terrier inverted index,
    then fetches original document fields from SQLite DocumentStore.
    """

    def __init__(
        self,
        index_path: str,
        db_path: str,
        model: str = "bm25",
        top_k: int = 10,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        init_pyterrier()

        self.index_ref = resolve_index_ref(index_path)
        self.db_path = db_path
        self.model = normalize_model_name(model)
        self.top_k = top_k
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        # self.query_processor = QueryProcessor()
        self.document_processor = DocumentProcessor()
        self.store = DocumentStore(db_path)

        retriever_kwargs = {
            "wmodel": self.model,
            "num_results": top_k,
        }

        # BM25 parameters are scoring-time controls, not indexing-time settings.
        # Therefore they can be changed from the UI without rebuilding the Terrier index.
        if self.model == "BM25":
            retriever_kwargs["controls"] = {
                "bm25.k_1": str(self.bm25_k1),
                "bm25.b": str(self.bm25_b),
            }

        self.retriever = pt.terrier.Retriever(
            self.index_ref,
            **retriever_kwargs,
        )

    def _bm25_parameters(self) -> dict | None:
        if self.model != "BM25":
            return None
        return {
            "k1": self.bm25_k1,
            "b": self.bm25_b,
        }

    def search(self, query: str) -> dict:
        start = time.time()

        original_query = query
        # processed_query = self.query_processor.process(query)
        processed_query = self.document_processor.process_to_text(query)

        if not processed_query:
            return {
                "query": original_query,
                "processed_query": processed_query,
                "model": self.model,
                "bm25_parameters": self._bm25_parameters(),
                "time_seconds": time.time() - start,
                "results": [],
            }

        queries = pd.DataFrame(
            [
                {
                    "qid": "q1",
                    "query": processed_query,
                }
            ]
        )

        result_df = self.retriever.transform(queries)

        if result_df.empty:
            return {
                "query": original_query,
                "processed_query": processed_query,
                "model": self.model,
                "bm25_parameters": self._bm25_parameters(),
                "time_seconds": time.time() - start,
                "results": [],
            }

        result_df = result_df.sort_values("rank")

        doc_ids = [str(docno) for docno in result_df["docno"].tolist()]
        docs = self.store.get_by_ids(doc_ids)
        docs_by_id = {str(doc["doc_id"]): doc for doc in docs}

        results = []

        for _, row in result_df.iterrows():
            doc_id = str(row["docno"])
            original_doc = docs_by_id.get(doc_id)

            if original_doc is None:
                continue

            results.append(
                {
                    "rank": int(row["rank"]) + 1,
                    "doc_id": doc_id,
                    "score": float(row["score"]),
                    "title": original_doc.get("title"),
                    "abstract": original_doc.get("abstract"),
                }
            )

        return {
            "query": original_query,
            "processed_query": processed_query,
            "model": self.model,
            "bm25_parameters": self._bm25_parameters(),
            "time_seconds": time.time() - start,
            "results": results,
        }