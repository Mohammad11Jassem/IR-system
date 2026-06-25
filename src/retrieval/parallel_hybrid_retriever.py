import time
from typing import Literal

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.word2vec_retriever import Word2VecRetriever
from src.retrieval.bert_retriever import BertRetriever


RetrievalModel = Literal["bm25", "tfidf", "word2vec", "bert"]
FusionMethod = Literal["rrf"]


class ParallelHybridRetriever:
    """
    Configurable Parallel Hybrid Retriever.

    Pipeline:
    Query
      -> each selected model retrieves top per_model_k independently
      -> results are fused using a fusion method
      -> final top_k documents are returned

    Supported models:
      - bm25
      - tfidf
      - word2vec
      - bert

    Supported fusion methods:
      - rrf: Reciprocal Rank Fusion
    """

    SUPPORTED_MODELS = {"bm25", "tfidf", "word2vec", "bert"}
    SUPPORTED_FUSION_METHODS = {"rrf"}

    def __init__(
        self,
        models: list[RetrievalModel],
        index_path: str,
        db_path: str,
        word2vec_index_dir: str,
        bert_index_dir: str,
        bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        fusion_method: FusionMethod = "rrf",
        per_model_k: int = 100,
        top_k: int = 10,
        rrf_k: int = 60,
    ):
        self.models = self._normalize_models(models)
        self.index_path = index_path
        self.db_path = db_path
        self.word2vec_index_dir = word2vec_index_dir
        self.bert_index_dir = bert_index_dir
        self.bert_model_name = bert_model_name

        self.fusion_method = fusion_method.lower().strip()
        self.per_model_k = per_model_k
        self.top_k = top_k
        self.rrf_k = rrf_k

        if len(self.models) < 2:
            raise ValueError("Parallel Hybrid requires at least two different models.")

        if self.fusion_method not in self.SUPPORTED_FUSION_METHODS:
            raise ValueError(
                f"Unsupported fusion_method: {fusion_method}. "
                f"Supported: {sorted(self.SUPPORTED_FUSION_METHODS)}"
            )

        self.retrievers = self._build_retrievers()

    def _normalize_models(
        self,
        models: list[str],
    ) -> list[str]:
        normalized = []

        for model in models:
            model = model.lower().strip()

            if model in {"tf-idf", "tf_idf"}:
                model = "tfidf"

            if model in {"w2v"}:
                model = "word2vec"

            if model not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model: {model}. "
                    f"Supported: {sorted(self.SUPPORTED_MODELS)}"
                )

            if model not in normalized:
                normalized.append(model)

        return normalized

    def _build_retrievers(self) -> dict:
        retrievers = {}

        for model in self.models:
            if model == "bm25":
                retrievers[model] = BM25Retriever(
                    index_path=self.index_path,
                    db_path=self.db_path,
                    top_k=self.per_model_k,
                )

            elif model == "tfidf":
                retrievers[model] = TfidfRetriever(
                    index_path=self.index_path,
                    db_path=self.db_path,
                    top_k=self.per_model_k,
                )

            elif model == "word2vec":
                retrievers[model] = Word2VecRetriever(
                    index_dir=self.word2vec_index_dir,
                    db_path=self.db_path,
                    top_k=self.per_model_k,
                )

            elif model == "bert":
                retrievers[model] = BertRetriever(
                    index_dir=self.bert_index_dir,
                    db_path=self.db_path,
                    model_name=self.bert_model_name,
                    top_k=self.per_model_k,
                )

            else:
                raise ValueError(f"Unsupported model: {model}")

        return retrievers

    def _rrf_score(
        self,
        rank: int,
    ) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _fuse_with_rrf(
        self,
        model_outputs: dict,
    ) -> list[dict]:
        fused_docs = {}

        for model_name, output in model_outputs.items():
            results = output.get("results", [])

            for item in results:
                doc_id = str(item.get("doc_id"))
                rank = int(item.get("rank"))
                original_score = float(item.get("score", 0.0))

                contribution = self._rrf_score(rank)

                if doc_id not in fused_docs:
                    fused_docs[doc_id] = {
                        "doc_id": doc_id,
                        "title": item.get("title"),
                        "abstract": item.get("abstract"),
                        "score": 0.0,
                        "model_contributions": {},
                    }

                # If title/abstract were missing from a previous model but available now
                if not fused_docs[doc_id].get("title") and item.get("title"):
                    fused_docs[doc_id]["title"] = item.get("title")

                if not fused_docs[doc_id].get("abstract") and item.get("abstract"):
                    fused_docs[doc_id]["abstract"] = item.get("abstract")

                fused_docs[doc_id]["score"] += contribution

                fused_docs[doc_id]["model_contributions"][model_name] = {
                    "rank": rank,
                    "score": original_score,
                    "rrf_contribution": contribution,
                }

        fused_results = list(fused_docs.values())
        fused_results.sort(key=lambda item: item["score"], reverse=True)

        return fused_results

    def search(
        self,
        query: str,
    ) -> dict:
        start = time.time()

        model_outputs = {}

        for model_name, retriever in self.retrievers.items():
            model_outputs[model_name] = retriever.search(query)

        if self.fusion_method == "rrf":
            fused_results = self._fuse_with_rrf(model_outputs)
        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

        final_results = []

        for rank, item in enumerate(fused_results[: self.top_k], start=1):
            final_results.append(
                {
                    "rank": rank,
                    "doc_id": item["doc_id"],
                    "score": float(item["score"]),
                    "fusion_method": self.fusion_method,
                    "title": item.get("title"),
                    "abstract": item.get("abstract"),
                    "model_contributions": item.get("model_contributions", {}),
                }
            )

        return {
            "query": query,
            "model": "PARALLEL_HYBRID",
            "models": self.models,
            "fusion_method": self.fusion_method,
            "per_model_k": self.per_model_k,
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
            "time_seconds": time.time() - start,
            "results": final_results,
            "model_outputs_summary": {
                model_name: {
                    "returned_results": len(output.get("results", [])),
                    "model": output.get("model"),
                    "time_seconds": output.get("time_seconds"),
                }
                for model_name, output in model_outputs.items()
            },
        }