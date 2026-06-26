import time
from pathlib import Path
from typing import Literal

import numpy as np

from src.embeddings import BertVectorizer, Word2VecVectorizer
from src.preprocessing import EmbeddingTextCleaner
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.word2vec_retriever import Word2VecRetriever, tokenize_for_word2vec


FirstStageModel = Literal["bm25", "tfidf", "word2vec"]
SecondStageModel = Literal["bert", "word2vec"]


class SerialHybridRetriever:
    """
    Generic configurable Serial Hybrid Retriever.

    Pipeline:
    Query
      -> first_stage retrieves candidate_k documents
      -> second_stage reranks these candidate documents
      -> final top_k results

    Supported first_stage:
      - bm25
      - tfidf
      - word2vec

    Supported second_stage:
      - bert
      - word2vec
    """

    def __init__(
        self,
        first_stage: FirstStageModel,
        second_stage: SecondStageModel,
        index_path: str,
        db_path: str,
        word2vec_index_dir: str,
        bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        candidate_k: int = 100,
        top_k: int = 10,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        self.first_stage = first_stage.lower().strip()
        self.second_stage = second_stage.lower().strip()

        self.index_path = index_path
        self.db_path = db_path
        self.word2vec_index_dir = word2vec_index_dir
        self.bert_model_name = bert_model_name

        self.candidate_k = candidate_k
        self.top_k = top_k
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        if self.first_stage not in {"bm25", "tfidf", "word2vec"}:
            raise ValueError(
                "first_stage must be one of: bm25, tfidf, word2vec"
            )

        if self.second_stage not in {"bert", "word2vec"}:
            raise ValueError(
                "second_stage must be one of: bert, word2vec"
            )

        if self.first_stage == self.second_stage:
            raise ValueError(
                "first_stage and second_stage should be different in Serial Hybrid."
            )

        self.cleaner = EmbeddingTextCleaner(max_chars=4000)

        self.first_stage_retriever = self._build_first_stage_retriever()

        self.bert_vectorizer = None
        self.word2vec_vectorizer = None

        if self.second_stage == "bert":
            self.bert_vectorizer = BertVectorizer(
                model_name=self.bert_model_name,
                normalize_embeddings=True,
            )

        if self.second_stage == "word2vec":
            model_path = Path(self.word2vec_index_dir) / "word2vec.model"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Word2Vec model not found: {model_path}"
                )

            self.word2vec_vectorizer = Word2VecVectorizer.load(str(model_path))

    def _build_first_stage_retriever(self):
        if self.first_stage == "bm25":
            return BM25Retriever(
                index_path=self.index_path,
                db_path=self.db_path,
                top_k=self.candidate_k,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
            )

        if self.first_stage == "tfidf":
            return TfidfRetriever(
                index_path=self.index_path,
                db_path=self.db_path,
                top_k=self.candidate_k,
            )

        if self.first_stage == "word2vec":
            return Word2VecRetriever(
                index_dir=self.word2vec_index_dir,
                db_path=self.db_path,
                top_k=self.candidate_k,
            )

        raise ValueError(f"Unsupported first_stage: {self.first_stage}")

    def _build_candidate_text(self, candidate: dict) -> str:
        title = candidate.get("title") or ""
        abstract = candidate.get("abstract") or ""

        text = f"{title}. {abstract}".strip()
        return self.cleaner.clean(text)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32)

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector

        return vector / norm

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        return matrix / norms

    def _rerank_with_bert(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        if self.bert_vectorizer is None:
            raise RuntimeError("BERT vectorizer is not initialized.")

        processed_query = self.cleaner.clean(query)

        valid_candidates = []
        valid_texts = []

        for candidate in candidates:
            text = self._build_candidate_text(candidate)

            if text:
                valid_candidates.append(candidate)
                valid_texts.append(text)

        if not valid_candidates:
            return []

        query_vector = self.bert_vectorizer.encode_query(processed_query)

        document_vectors = self.bert_vectorizer.encode_texts(
            valid_texts,
            batch_size=32,
            show_progress_bar=False,
        )

        # BertVectorizer uses normalized embeddings, so dot product = cosine similarity.
        scores = document_vectors @ query_vector

        reranked = []

        for candidate, rerank_score in zip(valid_candidates, scores):
            reranked.append(
                self._make_reranked_item(
                    candidate=candidate,
                    rerank_score=float(rerank_score),
                )
            )

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked

    def _rerank_with_word2vec(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        if self.word2vec_vectorizer is None:
            raise RuntimeError("Word2Vec vectorizer is not initialized.")

        processed_query = self.cleaner.clean(query)
        query_tokens = tokenize_for_word2vec(processed_query)

        if not query_tokens:
            return []

        query_vector = self.word2vec_vectorizer.encode_tokens(query_tokens)

        if not query_vector.any():
            return []

        valid_candidates = []
        valid_tokenized_texts = []

        for candidate in candidates:
            text = self._build_candidate_text(candidate)
            tokens = tokenize_for_word2vec(text)

            if tokens:
                valid_candidates.append(candidate)
                valid_tokenized_texts.append(tokens)

        if not valid_candidates:
            return []

        document_vectors = self.word2vec_vectorizer.encode_batch(
            valid_tokenized_texts
        )

        query_vector = self._normalize_vector(query_vector)
        document_vectors = self._normalize_matrix(document_vectors)

        scores = document_vectors @ query_vector

        reranked = []

        for candidate, rerank_score in zip(valid_candidates, scores):
            reranked.append(
                self._make_reranked_item(
                    candidate=candidate,
                    rerank_score=float(rerank_score),
                )
            )

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked

    def _make_reranked_item(
        self,
        candidate: dict,
        rerank_score: float,
    ) -> dict:
        return {
            "doc_id": str(candidate.get("doc_id")),
            "title": candidate.get("title"),
            "abstract": candidate.get("abstract"),

            "first_stage": self.first_stage,
            "second_stage": self.second_stage,

            "first_stage_rank": candidate.get("rank"),
            "first_stage_score": float(candidate.get("score", 0.0)),

            "rerank_score": rerank_score,
            "score": rerank_score,
        }

    def search(self, query: str) -> dict:
        start = time.time()

        first_output = self.first_stage_retriever.search(query)
        candidates = first_output.get("results", [])

        if not candidates:
            return {
                "query": query,
                "processed_query": first_output.get("processed_query", query),
                "model": "SERIAL_HYBRID",
                "first_stage": self.first_stage,
                "second_stage": self.second_stage,
                "candidate_k": self.candidate_k,
                "top_k": self.top_k,
                "bm25_parameters": {
                    "k1": self.bm25_k1,
                    "b": self.bm25_b,
                } if self.first_stage == "bm25" else None,
                "time_seconds": time.time() - start,
                "results": [],
            }

        if self.second_stage == "bert":
            reranked = self._rerank_with_bert(
                query=query,
                candidates=candidates,
            )

        elif self.second_stage == "word2vec":
            reranked = self._rerank_with_word2vec(
                query=query,
                candidates=candidates,
            )

        else:
            raise ValueError(f"Unsupported second_stage: {self.second_stage}")

        final_results = []

        for rank, item in enumerate(reranked[: self.top_k], start=1):
            final_results.append(
                {
                    "rank": rank,
                    **item,
                }
            )

        return {
            "query": query,
            "processed_query": first_output.get("processed_query", query),
            "model": "SERIAL_HYBRID",
            "first_stage": self.first_stage,
            "second_stage": self.second_stage,
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "bm25_parameters": {
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            } if self.first_stage == "bm25" else None,
            "time_seconds": time.time() - start,
            "results": final_results,
        }