import argparse
import sqlite3
import time
from pathlib import Path

import ir_datasets


DEFAULT_DATASET_ID = "medline/2004/trec-genomics-2005"


def get_doc_field(doc, field_name: str) -> str:
    """
    Safely get a field from ir_datasets document object.
    Medline docs usually have: doc_id, title, abstract.
    """
    value = getattr(doc, field_name, "")
    if value is None:
        return ""
    return str(value)


def create_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            contents TEXT
        );
    """)

    # doc_id is already indexed because it is PRIMARY KEY.
    connection.commit()


def optimize_sqlite_for_bulk_insert(connection: sqlite3.Connection) -> None:
    """
    These PRAGMAs make bulk insertion faster.
    They are acceptable during DB building.
    """
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -200000;")  # around 200MB cache
    connection.commit()


def insert_batch(connection: sqlite3.Connection, rows: list[tuple[str, str, str, str]]) -> None:
    cursor = connection.cursor()

    cursor.executemany("""
        INSERT OR REPLACE INTO documents (doc_id, title, abstract, contents)
        VALUES (?, ?, ?, ?);
    """, rows)

    connection.commit()


def build_document_store(dataset_id: str, db_path: Path, batch_size: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILD DOCUMENT STORE")
    print("=" * 70)
    print(f"Dataset: {dataset_id}")
    print(f"SQLite DB: {db_path}")
    print(f"Batch size: {batch_size}")
    print("=" * 70)

    dataset = ir_datasets.load(dataset_id)

    connection = sqlite3.connect(str(db_path))
    optimize_sqlite_for_bulk_insert(connection)
    create_schema(connection)

    start_time = time.time()
    batch = []
    total = 0

    for doc in dataset.docs_iter():
        doc_id = get_doc_field(doc, "doc_id")
        title = get_doc_field(doc, "title")
        abstract = get_doc_field(doc, "abstract")

        contents = f"{title}\n{abstract}".strip()

        batch.append((doc_id, title, abstract, contents))
        total += 1

        if len(batch) >= batch_size:
            insert_batch(connection, batch)
            batch.clear()

            elapsed = time.time() - start_time
            print(f"Inserted {total:,} documents | elapsed: {elapsed / 60:.2f} min")

    if batch:
        insert_batch(connection, batch)

    elapsed = time.time() - start_time

    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents;")
    count = cursor.fetchone()[0]

    connection.close()

    print("=" * 70)
    print("DONE")
    print(f"Inserted/available documents: {count:,}")
    print(f"Total elapsed time: {elapsed / 60:.2f} min")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="ir_datasets dataset id"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to output SQLite database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of documents inserted per transaction"
    )

    args = parser.parse_args()

    build_document_store(
        dataset_id=args.dataset,
        db_path=Path(args.db_path),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()