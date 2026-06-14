import math
from collections import Counter, defaultdict


class TfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        self.doc_ids = []

    def fit(self, tokenized_docs, doc_ids):
        """
        Build vocabulary + IDF from documents
        """

        self.doc_ids = doc_ids

        df = defaultdict(int)
        total_docs = len(tokenized_docs)

        # compute DF
        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1

        # build vocab + IDF
        for term, freq in df.items():
            self.vocab[term] = len(self.vocab)
            self.idf[term] = math.log((total_docs + 1) / (freq + 1)) + 1

        # build document vectors
        self.doc_vectors = [
            self._vectorize(doc) for doc in tokenized_docs
        ]

        return self

    def _vectorize(self, tokens):
        tf = Counter(tokens)
        vector = {}

        for term, count in tf.items():
            if term in self.idf:
                vector[term] = count * self.idf[term]

        return vector