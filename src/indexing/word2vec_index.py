import html
import json
import pickle
import re
import shutil
import sqlite3
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
from gensim.models import Word2Vec

from src.preprocessing import EmbeddingTextCleaner


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")



def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = SPACE_RE.sub(" ", text).strip().lower()

    return text


# def tokenize_biomedical_text(text: str) -> list[str]:
#     text = clean_text(text)
#     return TOKEN_RE.findall(text)


def tokenize_biomedical_text(text: str) -> list[str]:
    if not text:
        return []

    return TOKEN_RE.findall(text.lower())


def build_document_text(row: sqlite3.Row) -> str:
    title = row["title"] or ""
    abstract = row["abstract"] or ""
    contents = row["contents"] or ""

    if title or abstract:
        return f"{title}. {abstract}"

    return contents


# def iter_document_tokens_from_sqlite(
#     db_path: str | Path,
#     limit: int | None = None,
# ) -> Iterator[tuple[str, list[str]]]:
#     db_path = Path(db_path)

#     conn = sqlite3.connect(str(db_path))
#     conn.row_factory = sqlite3.Row

#     sql = "SELECT doc_id, title, abstract, contents FROM documents"
#     params = []

#     if limit is not None:
#         sql += " LIMIT ?"
#         params.append(int(limit))

#     try:
#         cursor = conn.execute(sql, params)

#         for row in cursor:
#             doc_id = str(row["doc_id"])
#             text = build_document_text(row)
#             tokens = tokenize_biomedical_text(text)

#             yield doc_id, tokens

#     finally:
#         conn.close()


def iter_document_tokens_from_sqlite(
    db_path: str | Path,
    limit: int | None = None,
) -> Iterator[tuple[str, list[str]]]:
    db_path = Path(db_path)

    cleaner = EmbeddingTextCleaner()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    sql = "SELECT doc_id, title, abstract, contents FROM documents"
    params = []

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    try:
        cursor = conn.execute(sql, params)

        for row in cursor:
            doc_id = str(row["doc_id"])
            raw_text = build_document_text(row)
            cleaned_text = cleaner.clean(raw_text)
            tokens = tokenize_biomedical_text(cleaned_text)

            yield doc_id, tokens

    finally:
        conn.close()

class SQLiteTokenCorpus:
    """
    Re-iterable corpus for Gensim Word2Vec training.

    Gensim needs to iterate over the corpus more than once, so this class opens
    a fresh SQLite cursor each time __iter__ is called.
    """

    def __init__(self, db_path: str | Path, limit: int | None = None):
        self.db_path = Path(db_path)
        self.limit = limit

    def __iter__(self):
        for _, tokens in iter_document_tokens_from_sqlite(
            db_path=self.db_path,
            limit=self.limit,
        ):
            if tokens:
                yield tokens


def average_word_vectors(
    tokens: list[str],
    model: Word2Vec,
) -> np.ndarray | None:
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return None

    vector = np.mean(vectors, axis=0).astype("float32")

    norm = np.linalg.norm(vector)
    if norm == 0:
        return None

    vector = vector / norm

    return vector.astype("float32")


def save_doc_ids_jsonl(path: Path, doc_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(json.dumps({"doc_id": str(doc_id)}, ensure_ascii=False) + "\n")


def build_word2vec_faiss_index(
    db_path: str | Path,
    index_dir: str | Path,
    limit: int | None = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    sg: int = 0,
    epochs: int = 3,
    batch_size: int = 8192,
    overwrite: bool = False,
) -> dict:
    start = time.time()

    db_path = Path(db_path)
    index_dir = Path(index_dir)

    if index_dir.exists() and overwrite:
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    model_path = index_dir / "word2vec.model"
    keyed_vectors_path = index_dir / "word2vec.kv"
    doc_ids_pkl_path = index_dir / "doc_ids.pkl"
    doc_ids_jsonl_path = index_dir / "doc_ids.jsonl"
    metadata_path = index_dir / "metadata.json"

    if index_path.exists() and not overwrite:
        raise FileExistsError(
            f"Word2Vec index already exists at {index_path}. "
            f"Use overwrite=True or --overwrite to rebuild."
        )

    print("=" * 80)
    print("BUILD WORD2VEC FAISS INDEX")
    print("=" * 80)
    print(f"SQLite DB   : {db_path}")
    print(f"Index dir   : {index_dir}")
    print(f"Limit       : {limit if limit is not None else 'FULL DATASET'}")
    print(f"Vector size : {vector_size}")
    print(f"Window      : {window}")
    print(f"Min count   : {min_count}")
    print(f"Workers     : {workers}")
    print(f"SG          : {sg} | 0=CBOW, 1=Skip-gram")
    print(f"Epochs      : {epochs}")
    print(f"Batch size  : {batch_size}")
    print("=" * 80)

    corpus = SQLiteTokenCorpus(db_path=db_path, limit=limit)

    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    model.save(str(model_path))
    model.wv.save(str(keyed_vectors_path))

    print("Building FAISS index from averaged document vectors...")

    index = faiss.IndexFlatIP(vector_size)
    all_doc_ids: list[str] = []

    vector_batch = []
    id_batch = []

    total_seen = 0
    total_indexed = 0
    total_skipped_empty_tokens = 0
    total_skipped_empty_vectors = 0

    def flush_batch() -> None:
        nonlocal total_indexed

        if not vector_batch:
            return

        vectors = np.vstack(vector_batch).astype("float32")
        index.add(vectors)
        all_doc_ids.extend(id_batch)

        total_indexed += len(vector_batch)

        vector_batch.clear()
        id_batch.clear()

    for doc_id, tokens in iter_document_tokens_from_sqlite(
        db_path=db_path,
        limit=limit,
    ):
        total_seen += 1

        if not tokens:
            total_skipped_empty_tokens += 1
            continue

        vector = average_word_vectors(tokens, model)

        if vector is None:
            total_skipped_empty_vectors += 1
            continue

        vector_batch.append(vector)
        id_batch.append(doc_id)

        if len(vector_batch) >= batch_size:
            flush_batch()

        if total_seen % 10000 == 0:
            elapsed = time.time() - start
            print(
                f"Seen={total_seen:,} | Indexed={total_indexed:,} | "
                f"Skipped tokens={total_skipped_empty_tokens:,} | "
                f"Skipped vectors={total_skipped_empty_vectors:,} | "
                f"FAISS={index.ntotal:,} | Elapsed={elapsed / 60:.2f} min"
            )

    flush_batch()

    if index.ntotal != len(all_doc_ids):
        raise ValueError(
            f"FAISS/doc_ids mismatch: index.ntotal={index.ntotal}, "
            f"doc_ids={len(all_doc_ids)}"
        )

    faiss.write_index(index, str(index_path))

    with doc_ids_pkl_path.open("wb") as f:
        pickle.dump(all_doc_ids, f)

    save_doc_ids_jsonl(doc_ids_jsonl_path, all_doc_ids)

    metadata = {
        "model_type": "word2vec",
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "workers": workers,
        "sg": sg,
        "sg_description": "1=skip-gram, 0=CBOW",
        "epochs": epochs,
        "db_path": str(db_path),
        "limit": limit,
        "text_fields": "title + abstract, fallback to contents",
        "tokenizer": "biomedical regex tokenizer preserving hyphenated terms",
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine via L2-normalized averaged Word2Vec vectors + inner product",
        "normalized_vectors": True,
        "total_seen": int(total_seen),
        "total_indexed": int(index.ntotal),
        "total_doc_ids": len(all_doc_ids),
        "skipped_empty_tokens": int(total_skipped_empty_tokens),
        "skipped_empty_vectors": int(total_skipped_empty_vectors),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": {
            "index": "index.faiss",
            "word2vec_model": "word2vec.model",
            "word2vec_keyed_vectors": "word2vec.kv",
            "doc_ids_pkl": "doc_ids.pkl",
            "doc_ids_jsonl": "doc_ids.jsonl",
            "metadata": "metadata.json",
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - start

    print("=" * 80)
    print("DONE")
    print(f"Index vectors          : {index.ntotal:,}")
    print(f"Doc IDs                : {len(all_doc_ids):,}")
    print(f"Skipped empty tokens   : {total_skipped_empty_tokens:,}")
    print(f"Skipped empty vectors  : {total_skipped_empty_vectors:,}")
    print(f"Dimension              : {vector_size}")
    print(f"Saved to               : {index_dir}")
    print(f"Elapsed                : {elapsed / 60:.2f} min")
    print("=" * 80)

    return metadata