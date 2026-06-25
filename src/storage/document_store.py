import sqlite3
from typing import Iterable


class DocumentStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_by_ids(self, doc_ids: Iterable[str]) -> list[dict]:
        doc_ids = list(doc_ids)



        if not doc_ids:
            return []

        placeholders = ",".join(["?"] * len(doc_ids))

        query = f"""
            SELECT doc_id, title, abstract, contents
            FROM documents
            WHERE doc_id IN ({placeholders})
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, doc_ids).fetchall()

        docs_by_id = {
            row["doc_id"]: {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "contents": row["contents"],
            }
            for row in rows
        }

        # مهم: نحافظ على نفس ترتيب النتائج الراجعة من نموذج البحث
        return [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]