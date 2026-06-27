import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import pyterrier as pt

from src.preprocessing import DocumentProcessor


_PYTERRIER_INITIALIZED = False


def init_pyterrier() -> None:
    """
    Initialize PyTerrier/Java only once per Python process.
    """
    global _PYTERRIER_INITIALIZED

    if _PYTERRIER_INITIALIZED:
        return

    try:
        if hasattr(pt, "java") and hasattr(pt.java, "started"):
            if pt.java.started():
                _PYTERRIER_INITIALIZED = True
                return

        pt.java.init()
        _PYTERRIER_INITIALIZED = True

    except ValueError as exc:
        message = str(exc)

        if "already been run" in message:
            _PYTERRIER_INITIALIZED = True
            return

        raise


def iter_documents_from_sqlite(
    db_path: str,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    """
    Stream documents from documents.sqlite to PyTerrier.

    IMPORTANT:
    The same DocumentProcessor used here is also used in TerrierRetriever
    for BM25/TF-IDF queries. This keeps document indexing and query
    processing consistent for traditional IR models.

    Terrier expects:
    - docno: external document id
    - text: processed document text to index
    """
    processor = DocumentProcessor()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if limit is None:
        cursor = conn.execute("""
            SELECT doc_id, contents
            FROM documents
        """)
    else:
        cursor = conn.execute(
            """
            SELECT doc_id, contents
            FROM documents
            LIMIT ?
            """,
            (limit,),
        )

    yielded = 0
    skipped_empty_raw = 0
    skipped_empty_after_processing = 0
    start = time.time()

    for row in cursor:
        doc_id = str(row["doc_id"])
        raw_text = row["contents"] or ""

        if not raw_text.strip():
            skipped_empty_raw += 1
            continue

        processed_text = processor.process_to_text(raw_text)

        if not processed_text.strip():
            skipped_empty_after_processing += 1
            continue

        yielded += 1

        if yielded % 100000 == 0:
            elapsed = time.time() - start
            print(
                f"Yielded {yielded:,} processed documents | "
                f"Skipped raw empty: {skipped_empty_raw:,} | "
                f"Skipped after preprocessing: {skipped_empty_after_processing:,} | "
                f"elapsed: {elapsed / 60:.2f} min"
            )

        yield {
            "docno": doc_id,
            "text": processed_text,
        }

    conn.close()

    print(
        "Finished streaming processed documents. "
        f"Yielded: {yielded:,}, "
        f"skipped raw empty: {skipped_empty_raw:,}, "
        f"skipped after preprocessing: {skipped_empty_after_processing:,}"
    )


def build_terrier_index(
    db_path: str,
    index_path: str,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> str:
    """
    Build a Terrier inverted index from documents.sqlite.

    Documents are processed with DocumentProcessor before being sent to
    Terrier. TerrierRetriever uses the same DocumentProcessor for queries.
    """
    init_pyterrier()

    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    index_dir = Path(index_path)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILD TERRIER INDEX")
    print("=" * 80)
    print(f"SQLite DB       : {db_file}")
    print(f"Index path      : {index_dir}")
    print(f"Overwrite       : {overwrite}")
    print(f"Limit           : {limit if limit is not None else 'FULL DATASET'}")
    print("Preprocessing   : DocumentProcessor -> process_to_text")
    print("=" * 80)

    indexer = pt.terrier.IterDictIndexer(
        str(index_dir),
        overwrite=overwrite,
        meta={
            "docno": 32,
        },
        meta_reverse=["docno"],
    )

    start = time.time()

    index_ref = indexer.index(
        iter_documents_from_sqlite(
            db_path=str(db_file),
            limit=limit,
        )
    )

    elapsed = time.time() - start

    index_ref_path = index_dir / "index_ref.txt"
    index_ref_path.write_text(str(index_ref), encoding="utf-8")

    print("=" * 80)
    print("DONE")
    print(f"IndexRef: {index_ref}")
    print(f"Saved IndexRef to: {index_ref_path}")
    print(f"Total time: {elapsed / 60:.2f} min")
    print("=" * 80)

    return str(index_ref)
