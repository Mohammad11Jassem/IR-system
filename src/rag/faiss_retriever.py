from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.config import BERT_INDEX_DIR, EMBEDDING_MODEL_NAME
from src.rag.document_repository import DocumentRepository


class BertFaissRetriever:
    """
    BERT + FAISS retriever for RAG.

    Required files in index_dir:
    - index.faiss
    - doc_ids.pkl or doc_ids.jsonl
    - metadata.json optional
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        model_name: str | None = None,
    ):
        self.index_dir = Path(index_dir) if index_dir else BERT_INDEX_DIR
        self.faiss_index_path = self.index_dir / "index.faiss"
        self.doc_ids_pkl_path = self.index_dir / "doc_ids.pkl"
        self.doc_ids_jsonl_path = self.index_dir / "doc_ids.jsonl"
        self.metadata_path = self.index_dir / "metadata.json"

        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")

        self.model_name = model_name or self._load_model_name()
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(self.faiss_index_path))
        self.doc_ids = self._load_doc_ids()
        self.repo = DocumentRepository(db_path=db_path)

        if self.index.ntotal != len(self.doc_ids):
            raise RuntimeError(
                f"Mismatch between FAISS vectors and doc_ids: "
                f"FAISS has {self.index.ntotal}, doc_ids has {len(self.doc_ids)}"
            )

    def _load_model_name(self) -> str:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for key in ["model_name", "embedding_model", "sentence_transformer_model"]:
                if metadata.get(key):
                    return str(metadata[key])
        return EMBEDDING_MODEL_NAME

    def _load_doc_ids(self) -> List[str]:
        if self.doc_ids_pkl_path.exists():
            with open(self.doc_ids_pkl_path, "rb") as f:
                return [str(doc_id) for doc_id in pickle.load(f)]

        if self.doc_ids_jsonl_path.exists():
            with open(self.doc_ids_jsonl_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        raise FileNotFoundError(
            f"Neither doc_ids.pkl nor doc_ids.jsonl was found in {self.index_dir}"
        )

    def search(self, query: str, top_k: int = 10):
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vector, int(top_k))

        results = []
        result_doc_ids = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            doc_id = self.doc_ids[int(idx)]
            result_doc_ids.append(doc_id)
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": float(score),
                    "faiss_index": int(idx),
                }
            )

        docs = self.repo.get_documents_by_ids(result_doc_ids)
        for result in results:
            doc = docs.get(result["doc_id"], {})
            result["title"] = doc.get("title", "")
            result["abstract"] = doc.get("abstract", "")
            result["raw_text"] = doc.get("raw_text", "")

        return results
