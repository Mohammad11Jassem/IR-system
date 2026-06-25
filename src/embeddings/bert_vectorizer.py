from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class BertVectorizer:
    """
    BERT/SentenceTransformer vectorizer.

    Converts documents and queries into dense embedding vectors.
    This class does not build or store the FAISS index.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.encode_texts(
            [query],
            batch_size=1,
            show_progress_bar=False,
        )

        return vector[0]