import pickle
from pathlib import Path
from typing import List

import numpy as np
from gensim.models import Word2Vec


class Word2VecVectorizer:
    """
    Word2Vec vectorizer.

    It trains a Word2Vec model on tokenized documents and represents
    each document/query by averaging its word vectors.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        sg: int = 1,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model: Word2Vec | None = None

    def train(
        self,
        tokenized_texts: List[List[str]],
        epochs: int = 5,
    ) -> None:
        if not tokenized_texts:
            raise ValueError("No tokenized texts provided for Word2Vec training.")

        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=epochs,
        )

    def encode_tokens(
        self,
        tokens: List[str],
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Word2Vec model is not trained or loaded.")

        vectors = [
            self.model.wv[token]
            for token in tokens
            if token in self.model.wv
        ]

        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def encode_batch(
        self,
        tokenized_texts: List[List[str]],
    ) -> np.ndarray:
        if not tokenized_texts:
            return np.empty((0, self.vector_size), dtype=np.float32)

        vectors = [
            self.encode_tokens(tokens)
            for tokens in tokenized_texts
        ]

        return np.vstack(vectors).astype(np.float32)

    def save(
        self,
        model_path: str,
    ) -> None:
        if self.model is None:
            raise ValueError("Cannot save Word2Vec model before training.")

        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(output_path))

    @classmethod
    def load(
        cls,
        model_path: str,
    ) -> "Word2VecVectorizer":
        model = Word2Vec.load(str(model_path))

        vectorizer = cls(
            vector_size=model.vector_size,
        )

        vectorizer.model = model
        return vectorizer