import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pyterrier as pt


from src.storage.document_store import DocumentStore


DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
SMOKE_INDEX_PATH = r"E:\ir_project_artifacts\indexes\terrier_smoke_test"


def init_pyterrier():
    pt.java.init()


def iter_docs(limit: int = 5000):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """
        SELECT doc_id, contents
        FROM documents
        LIMIT ?
        """,
        (limit,),
    )

    for row in cursor:
        yield {
            "docno": str(row["doc_id"]),
            "text": row["contents"] or "",
        }

    conn.close()


def main():
    init_pyterrier()

    index_path = Path(SMOKE_INDEX_PATH)

    if index_path.exists():
        shutil.rmtree(index_path)

    indexer = pt.terrier.IterDictIndexer(
        str(index_path),
        overwrite=True,
        meta={"docno": 32},
        meta_reverse=["docno"],
    )

    print("Building smoke test Terrier index...")
    index_ref = indexer.index(iter_docs(limit=5000))
    print("Index built:", index_ref)

    query = "breast cancer gene"

    queries = pd.DataFrame([
        {
            "qid": "q1",
            "query": query,
        }
    ])

    for model in ["BM25", "TF_IDF"]:
        print("=" * 80)
        print(f"Testing model: {model}")

        retriever = pt.terrier.Retriever(
            index_ref,
            wmodel=model,
            num_results=10,
        )

        result_df = retriever.transform(queries)

        print(result_df[["qid", "docno", "rank", "score"]])

        doc_ids = result_df["docno"].astype(str).tolist()

        store = DocumentStore(DB_PATH)
        docs = store.get_by_ids(doc_ids)

        print("-" * 80)
        print("Original documents from SQLite:")
        for doc in docs[:3]:
            print(doc["doc_id"], "-", doc["title"])
        print("-" * 80)


if __name__ == "__main__":
    main()