"""
DEPRECATED / EDUCATIONAL ONLY.

This is a manual in-memory BM25 implementation.
It scans all documents and does not use the Terrier inverted index.

Do not use this file in the official system pipeline.
The official BM25 path is:
Terrier index -> TerrierRetriever -> BM25Retriever.
"""


import math
from collections import defaultdict, Counter


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.doc_freq = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0

        self.docs = []
        self.doc_ids = []

        self.N = 0  # number of documents

    def fit(self, tokenized_docs, doc_ids):
        self.docs = tokenized_docs
        self.doc_ids = doc_ids
        self.N = len(tokenized_docs)

        total_length = 0

        # compute DF + doc lengths
        for doc_id, doc in zip(doc_ids, tokenized_docs):
            self.doc_lengths[doc_id] = len(doc)
            total_length += len(doc)

            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freq[term] += 1

        self.avg_doc_length = total_length / self.N

        return self