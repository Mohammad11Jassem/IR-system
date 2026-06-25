# import time
# from pathlib import Path
# from typing import Literal

# import pandas as pd
# import pyterrier as pt


# from src.indexing.terrier_index import init_pyterrier
# from src.storage.document_store import DocumentStore
# from src.preprocessing import QueryProcessor


# ModelName = Literal["bm25", "tfidf", "tf-idf"]


# def resolve_index_ref(index_path: str) -> str:
#     """
#     Terrier can load from data.properties.
#     If the user gives the index directory, we resolve it.
#     """
#     path = Path(index_path)

#     if path.is_dir():
#         data_properties = path / "data.properties"
#         if data_properties.exists():
#             return str(data_properties)

#     return str(path)


# def normalize_model_name(model: str) -> str:
#     model = model.lower().strip()

#     if model == "bm25":
#         return "BM25"

#     if model in {"tfidf", "tf-idf", "tf_idf"}:
#         return "TF_IDF"

#     raise ValueError(f"Unsupported Terrier model: {model}")


# class TerrierRetriever:
#     """
#     Runs BM25 or TF-IDF using Terrier index,
#     then fetches original documents from SQLite by doc_id.
#     """

#     def __init__(
#         self,
#         index_path: str,
#         db_path: str,
#         model: str = "bm25",
#         top_k: int = 10,
#     ):
#         init_pyterrier()

#         self.index_ref = resolve_index_ref(index_path)
#         self.db_path = db_path
#         self.model = normalize_model_name(model)
#         self.top_k = top_k

#         self.store = DocumentStore(db_path)
#         self.query_processor = QueryProcessor()

#         self.retriever = pt.terrier.Retriever(
#             self.index_ref,
#             wmodel=self.model,
#             num_results=top_k,
#         )

#     # def search(self, query: str) -> dict:
#     #     start = time.time()

#     #     queries = pd.DataFrame([
#     #         {
#     #             "qid": "q1",
#     #             "query": query,
#     #         }
#     #     ])
    
#         # def search(self, query: str) -> dict:
#         # start = time.time()

#         # original_query = query
#         # processed_query = self.query_processor.process(query)

#         # if not processed_query:
#         #     return {
#         #         "query": original_query,
#         #         "processed_query": processed_query,
#         #         "model": self.model,
#         #         "time_seconds": time.time() - start,
#         #         "results": [],
#         #     }

#         # queries = pd.DataFrame([
#         #     {
#         #         "qid": "q1",
#         #         "query": processed_query,
#         #     }
#         # ])
        
#     def search(self, query: str) -> dict:
#         start = time.time()

#         original_query = query
#         processed_query = self.query_processor.process(query)

#         if not processed_query:
#             return {
#                 "query": original_query,
#                 "processed_query": processed_query,
#                 "model": self.model,
#                 "time_seconds": time.time() - start,
#                 "results": [],
#             }

#         queries = pd.DataFrame(
#             [
#                 {
#                     "qid": "q1",
#                     "query": processed_query,
#                 }
#             ]
#         )

#         retrieved = self.retriever.transform(queries)

#         if retrieved.empty:
#             return {
#                 "query": original_query,
#                 "processed_query": processed_query,
#                 "model": self.model,
#                 "time_seconds": time.time() - start,
#                 "results": [],
#             }

#         doc_ids = retrieved["docno"].astype(str).tolist()
#         documents = self.store.get_by_ids(doc_ids)

#         documents_by_id = {
#             str(doc["doc_id"]): doc
#             for doc in documents
#         }

#         results = []

#         for index, row in retrieved.iterrows():
#             doc_id = str(row["docno"])
#             document = documents_by_id.get(doc_id, {})

#             results.append(
#                 {
#                     "rank": int(row.get("rank", len(results))),
#                     "doc_id": doc_id,
#                     "score": float(row["score"]),
#                     "title": document.get("title"),
#                     "abstract": document.get("abstract"),
#                 }
#             )

#         return {
#             "query": original_query,
#             "processed_query": processed_query,
#             "model": self.model,
#             "time_seconds": time.time() - start,
#             "results": results,
#         }
        

#         result_df = self.retriever.transform(queries)

#         if result_df.empty:
#     return {
#         "query": original_query,
#         "processed_query": processed_query,
#         "model": self.model,
#         "time_seconds": time.time() - start,
#         "results": [],
#     }

#         result_df = result_df.sort_values("rank")

#         doc_ids = [str(docno) for docno in result_df["docno"].tolist()]
#         docs = self.store.get_by_ids(doc_ids)
#         docs_by_id = {doc["doc_id"]: doc for doc in docs}

#         results = []

#         for _, row in result_df.iterrows():
#             doc_id = str(row["docno"])
#             original_doc = docs_by_id.get(doc_id)

#             if original_doc is None:
#                 continue

#             results.append({
#                 # "rank": int(row["rank"]) + 1 if int(row["rank"]) == 0 else int(row["rank"]),
#                 "rank": int(row["rank"]) + 1,
#                 "doc_id": doc_id,
#                 "score": float(row["score"]),
#                 "title": original_doc["title"],
#                 "abstract": original_doc["abstract"],
#             })

#         # return {
#         #     "query": query,
#         #     "model": self.model,
#         #     "time_seconds": time.time() - start,
#         #     "results": results,
#         # }
#         return {
#             "query": original_query,
#             "processed_query": processed_query,
#             "model": self.model,
#             "time_seconds": time.time() - start,
#             "results": results,
#         }





import time
from pathlib import Path
from typing import Literal

import pandas as pd
import pyterrier as pt

from src.indexing.terrier_index import init_pyterrier
from src.preprocessing import QueryProcessor
from src.storage.document_store import DocumentStore


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
    ):
        init_pyterrier()

        self.index_ref = resolve_index_ref(index_path)
        self.db_path = db_path
        self.model = normalize_model_name(model)
        self.top_k = top_k

        self.query_processor = QueryProcessor()
        self.store = DocumentStore(db_path)

        self.retriever = pt.terrier.Retriever(
            self.index_ref,
            wmodel=self.model,
            num_results=top_k,
        )

    def search(self, query: str) -> dict:
        start = time.time()

        original_query = query
        processed_query = self.query_processor.process(query)

        if not processed_query:
            return {
                "query": original_query,
                "processed_query": processed_query,
                "model": self.model,
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
            "time_seconds": time.time() - start,
            "results": results,
        }