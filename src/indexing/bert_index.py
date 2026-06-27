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
from sentence_transformers import SentenceTransformer

from src.preprocessing import EmbeddingTextCleaner


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_embedding_text(text: str, max_chars: int | None = 4000) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = SPACE_RE.sub(" ", text).strip()

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]

    return text


# def build_document_text(row: sqlite3.Row, max_chars: int | None = 4000) -> str:
#     title = row["title"] or ""
#     abstract = row["abstract"] or ""
#     contents = row["contents"] or ""

#     if title or abstract:
#         text = f"{title}. {abstract}"
#     else:
#         text = contents

#     return clean_embedding_text(text, max_chars=max_chars)


def build_document_text(
    row: sqlite3.Row,
    cleaner: EmbeddingTextCleaner,
    max_chars: int | None = 4000,
) -> str:
    title = row["title"] or ""
    abstract = row["abstract"] or ""
    contents = row["contents"] or ""

    if title or abstract:
        raw_text = f"{title}. {abstract}"
    else:
        raw_text = contents

    cleaned_text = cleaner.clean(raw_text)

    if max_chars is not None and len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars]

    return cleaned_text

# def iter_documents_from_sqlite(
#     db_path: str | Path,
#     limit: int | None = None,
#     max_chars: int | None = 4000,
# ) -> Iterator[tuple[str, str]]:
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
#             text = build_document_text(row, max_chars=max_chars)

#             if not text.strip():
#                 continue

#             yield doc_id, text

#     finally:
#         conn.close()


def iter_documents_from_sqlite(
    db_path: str | Path,
    limit: int | None = None,
    max_chars: int | None = 4000,
) -> Iterator[tuple[str, str]]:
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
            text = build_document_text(
                row=row,
                cleaner=cleaner,
                max_chars=max_chars,
            )

            if not text.strip():
                continue

            yield doc_id, text

    finally:
        conn.close()

def save_doc_ids_jsonl(path: Path, doc_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(json.dumps({"doc_id": str(doc_id)}, ensure_ascii=False) + "\n")


def build_bert_faiss_index(
    db_path: str | Path,
    index_dir: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit: int | None = None,
    batch_size: int = 256,
    max_chars: int | None = 4000,
    overwrite: bool = False,
    write_compat_copy: bool = False,
) -> dict:
    start = time.time()

    db_path = Path(db_path)
    index_dir = Path(index_dir)

    if index_dir.exists() and overwrite:
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    doc_ids_pkl_path = index_dir / "doc_ids.pkl"
    doc_ids_jsonl_path = index_dir / "doc_ids.jsonl"
    metadata_path = index_dir / "metadata.json"

    if index_path.exists() and not overwrite:
        raise FileExistsError(
            f"BERT index already exists at {index_path}. "
            f"Use overwrite=True or --overwrite to rebuild."
        )

    print("=" * 80)
    print("BUILD BERT FAISS INDEX")
    print("=" * 80)
    print(f"SQLite DB    : {db_path}")
    print(f"Index dir    : {index_dir}")
    print(f"Model        : {model_name}")
    print(f"Limit        : {limit if limit is not None else 'FULL DATASET'}")
    print(f"Batch size   : {batch_size}")
    print(f"Max chars    : {max_chars}")
    print("=" * 80)

    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)

    all_doc_ids: list[str] = []
    batch_doc_ids: list[str] = []
    batch_texts: list[str] = []

    total_seen = 0
    total_indexed = 0
    total_skipped = 0

    def flush_batch() -> None:
        nonlocal total_indexed

        if not batch_texts:
            return

        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        index.add(embeddings)
        all_doc_ids.extend(batch_doc_ids)

        total_indexed += len(batch_texts)

        batch_doc_ids.clear()
        batch_texts.clear()

    for doc_id, text in iter_documents_from_sqlite(
        db_path=db_path,
        limit=limit,
        max_chars=max_chars,
    ):
        total_seen += 1

        if not text.strip():
            total_skipped += 1
            continue

        batch_doc_ids.append(doc_id)
        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            flush_batch()

        if total_seen % 10000 == 0:
            elapsed = time.time() - start
            print(
                f"Seen={total_seen:,} | Indexed={total_indexed:,} | "
                f"Skipped={total_skipped:,} | FAISS={index.ntotal:,} | "
                f"Elapsed={elapsed / 60:.2f} min"
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
        "model_type": "bert",
        "model_name": model_name,
        "dimension": int(dimension),
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine via normalized embeddings + inner product",
        "normalized_embeddings": True,
        "db_path": str(db_path),
        "limit": limit,
        "text_fields": "title + abstract, fallback to contents",
        "max_chars": max_chars,
        "batch_size": batch_size,
        "total_seen": int(total_seen),
        "total_indexed": int(index.ntotal),
        "total_doc_ids": len(all_doc_ids),
        "total_skipped": int(total_skipped),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": {
            "index": "index.faiss",
            "doc_ids_pkl": "doc_ids.pkl",
            "doc_ids_jsonl": "doc_ids.jsonl",
            "metadata": "metadata.json",
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if write_compat_copy:
        shutil.copy2(index_path, index_dir / "bert_faiss.index")

    elapsed = time.time() - start

    print("=" * 80)
    print("DONE")
    print(f"Index vectors : {index.ntotal:,}")
    print(f"Doc IDs       : {len(all_doc_ids):,}")
    print(f"Dimension     : {dimension}")
    print(f"Saved to      : {index_dir}")
    print(f"Elapsed       : {elapsed / 60:.2f} min")
    print("=" * 80)

    return metadata