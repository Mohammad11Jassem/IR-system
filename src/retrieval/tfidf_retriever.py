# import math
# class TfidfRetriever:
#     def __init__(self, vectorizer):
#         self.vectorizer = vectorizer

#     def _cosine_similarity(self, q_vec, d_vec):
#         dot = 0.0
#         q_norm = 0.0
#         d_norm = 0.0

#         for term, q_val in q_vec.items():
#             q_norm += q_val ** 2
#             if term in d_vec:
#                 dot += q_val * d_vec[term]

#         for val in d_vec.values():
#             d_norm += val ** 2

#         if q_norm == 0 or d_norm == 0:
#             return 0.0

#         return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))

#     def search(self, query_tokens, top_k=10):
#         q_vec = self.vectorizer._vectorize(query_tokens)

#         scores = []

#         for i, d_vec in enumerate(self.vectorizer.doc_vectors):
#             score = self._cosine_similarity(q_vec, d_vec)
#             scores.append((self.vectorizer.doc_ids[i], score))

#         scores.sort(key=lambda x: x[1], reverse=True)

#         return scores[:top_k]

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