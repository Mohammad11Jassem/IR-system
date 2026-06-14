import math
from collections import Counter

from src.retrieval.bm25_engine import BM25Index
class BM25Retriever:
    def __init__(self, index: BM25Index):
        self.index = index

    def _idf(self, term):
        df = self.index.doc_freq.get(term, 0)
        N = self.index.N

        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens, doc_tokens, doc_id):
        score = 0.0
        tf = Counter(doc_tokens)

        doc_len = self.index.doc_lengths[doc_id]
        avg_len = self.index.avg_doc_length

        for term in query_tokens:
            if term not in tf:
                continue

            idf = self._idf(term)
            freq = tf[term]

            numerator = freq * (self.index.k1 + 1)
            denominator = freq + self.index.k1 * (
                1 - self.index.b + self.index.b * (doc_len / avg_len)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query_tokens, top_k=10):
        results = []

        for doc_tokens, doc_id in zip(self.index.docs, self.index.doc_ids):
            score = self._score(query_tokens, doc_tokens, doc_id)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]