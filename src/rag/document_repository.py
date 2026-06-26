import sqlite3
from typing import Dict, List

from src.rag.config import DB_PATH


class DocumentRepository:
    """
    يقرأ الوثائق الأصلية من SQLite.

    يدعم شكلين من قواعد البيانات:

    الشكل الأول:
    documents(doc_id, title, abstract, raw_text)

    الشكل الثاني الخاص بمشروع صديقك غالباً:
    documents(doc_id, title, abstract, contents)
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.columns = self._load_columns()

    def _load_columns(self) -> set[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("PRAGMA table_info(documents)").fetchall()

        return {row[1] for row in rows}

    def _text_column(self) -> str:
        """
        يحدد اسم العمود الذي يحتوي النص الأصلي.
        """
        if "raw_text" in self.columns:
            return "raw_text"

        if "contents" in self.columns:
            return "contents"

        raise RuntimeError(
            "Cannot find text column in documents table. "
            "Expected either raw_text or contents."
        )

    def count_documents(self) -> int:
        text_col = self._text_column()

        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM documents
                WHERE {text_col} IS NOT NULL AND {text_col} != ''
                """
            ).fetchone()[0]

        return count

    def get_documents_by_ids(self, doc_ids: List[str]) -> Dict[str, dict]:
        if not doc_ids:
            return {}

        text_col = self._text_column()
        placeholders = ",".join("?" for _ in doc_ids)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                f"""
                SELECT doc_id, title, abstract, {text_col} AS text_content
                FROM documents
                WHERE doc_id IN ({placeholders})
                """,
                doc_ids,
            ).fetchall()

        docs = {}

        for row in rows:
            doc_id = str(row["doc_id"])

            title = row["title"] or ""
            abstract = row["abstract"] or ""
            text_content = row["text_content"] or ""

            docs[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "raw_text": text_content,
            }

        return docs