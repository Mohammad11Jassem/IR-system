import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS vector index wrapper.

    It stores:
    - FAISS index
    - doc_ids mapping: faiss_position -> doc_id
    - metadata.json
    """

    def __init__(
        self,
        dimension: int,
        normalize_vectors: bool = True,
    ):
        self.dimension = dimension
        self.normalize_vectors = normalize_vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_ids: List[str] = []

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: List[str],
    ) -> None:
        if vectors is None or len(vectors) == 0:
            return

        if len(vectors) != len(doc_ids):
            raise ValueError(
                f"vectors count ({len(vectors)}) != doc_ids count ({len(doc_ids)})"
            )

        vectors = vectors.astype(np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector dimension {self.dimension}, got {vectors.shape[1]}"
            )

        if self.normalize_vectors:
            faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.doc_ids.extend([str(doc_id) for doc_id in doc_ids])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)

        if self.normalize_vectors:
            faiss.normalize_L2(query_vector)

        scores, positions = self.index.search(query_vector, top_k)

        results: List[Tuple[str, float]] = []

        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:
                continue

            doc_id = self.doc_ids[int(pos)]
            results.append((doc_id, float(score)))

        return results

    def save(
        self,
        index_dir: str,
        metadata: dict,
    ) -> None:
        output_dir = Path(index_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(output_dir / "index.faiss"),
        )

        with open(output_dir / "doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)

        metadata = dict(metadata)
        metadata["dimension"] = self.dimension
        metadata["num_documents"] = len(self.doc_ids)
        metadata["normalize_vectors"] = self.normalize_vectors

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "FaissVectorStore":
        index_dir_path = Path(index_dir)

        metadata_path = index_dir_path / "metadata.json"
        index_path = index_dir_path / "index.faiss"
        doc_ids_path = index_dir_path / "doc_ids.pkl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        if not index_path.exists():
            raise FileNotFoundError(f"index.faiss not found: {index_path}")

        if not doc_ids_path.exists():
            raise FileNotFoundError(f"doc_ids.pkl not found: {doc_ids_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(
            dimension=int(metadata["dimension"]),
            normalize_vectors=bool(metadata.get("normalize_vectors", True)),
        )

        store.index = faiss.read_index(str(index_path))

        with open(doc_ids_path, "rb") as f:
            store.doc_ids = pickle.load(f)

        return store