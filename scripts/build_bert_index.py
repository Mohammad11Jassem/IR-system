# import argparse
# import sqlite3
# import sys
# import time
# from pathlib import Path
# from typing import Iterator, Optional

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.embeddings import BertVectorizer, FaissVectorStore
# from src.preprocessing import EmbeddingTextCleaner


# def build_embedding_text(row: sqlite3.Row) -> str:
#     """
#     Build the raw text that will be embedded.

#     We keep this function responsible only for selecting/combining fields.
#     Cleaning is done later by EmbeddingTextCleaner.
#     """
#     title = row["title"] or ""
#     abstract = row["abstract"] or ""
#     contents = row["contents"] or ""

#     text = f"{title}. {abstract}".strip()

#     if not text:
#         text = contents.strip()

#     return text


# def iter_sqlite_documents(
#     db_path: str,
#     cleaner: EmbeddingTextCleaner,
#     limit: Optional[int] = None,
# ) -> Iterator[tuple[str, str]]:
#     """
#     Stream documents from SQLite and return cleaned text for embedding.

#     SQLite keeps original documents.
#     This function creates a cleaned temporary version used only for BERT embeddings.
#     """
#     conn = sqlite3.connect(db_path)
#     conn.row_factory = sqlite3.Row

#     if limit is None:
#         cursor = conn.execute(
#             """
#             SELECT doc_id, title, abstract, contents
#             FROM documents
#             """
#         )
#     else:
#         cursor = conn.execute(
#             """
#             SELECT doc_id, title, abstract, contents
#             FROM documents
#             LIMIT ?
#             """,
#             (limit,),
#         )

#     try:
#         for row in cursor:
#             doc_id = str(row["doc_id"])

#             raw_text = build_embedding_text(row)
#             cleaned_text = cleaner.clean(raw_text)

#             if not cleaned_text:
#                 continue

#             yield doc_id, cleaned_text

#     finally:
#         conn.close()


# def build_bert_index(
#     db_path: str,
#     index_dir: str,
#     model_name: str,
#     limit: Optional[int],
#     batch_size: int,
# ) -> None:
#     print("=" * 80)
#     print("BUILD BERT FAISS INDEX")
#     print("=" * 80)
#     print(f"SQLite DB : {db_path}")
#     print(f"Index dir : {index_dir}")
#     print(f"Model     : {model_name}")
#     print(f"Limit     : {limit if limit is not None else 'FULL DATASET'}")
#     print(f"Batch size: {batch_size}")
#     print("=" * 80)

#     cleaner = EmbeddingTextCleaner(
#         normalize_unicode=True,
#         remove_html_tags=True,
#         collapse_whitespace=True,
#         max_chars=4000,
#     )

#     vectorizer = BertVectorizer(
#         model_name=model_name,
#         normalize_embeddings=True,
#     )

#     test_vector = vectorizer.encode_query("dimension test")
#     dimension = int(test_vector.shape[0])

#     store = FaissVectorStore(
#         dimension=dimension,
#         normalize_vectors=True,
#     )

#     batch_doc_ids: list[str] = []
#     batch_texts: list[str] = []

#     total_indexed = 0
#     total_seen = 0
#     start = time.time()

#     for doc_id, cleaned_text in iter_sqlite_documents(
#         db_path=db_path,
#         cleaner=cleaner,
#         limit=limit,
#     ):
#         total_seen += 1

#         batch_doc_ids.append(doc_id)
#         batch_texts.append(cleaned_text)

#         if len(batch_texts) >= batch_size:
#             vectors = vectorizer.encode_texts(
#                 batch_texts,
#                 batch_size=batch_size,
#                 show_progress_bar=False,
#             )

#             store.add(
#                 vectors=vectors,
#                 doc_ids=batch_doc_ids,
#             )

#             total_indexed += len(batch_texts)

#             elapsed = time.time() - start
#             print(
#                 f"Indexed {total_indexed:,} documents "
#                 f"| seen: {total_seen:,} "
#                 f"| elapsed: {elapsed / 60:.2f} min"
#             )

#             batch_doc_ids = []
#             batch_texts = []

#     if batch_texts:
#         vectors = vectorizer.encode_texts(
#             batch_texts,
#             batch_size=batch_size,
#             show_progress_bar=False,
#         )

#         store.add(
#             vectors=vectors,
#             doc_ids=batch_doc_ids,
#         )

#         total_indexed += len(batch_texts)

#     metadata = {
#         "model_type": "bert",
#         "model_name": model_name,
#         "text_fields": "title + abstract, fallback to contents",
#         "text_cleaning": {
#             "type": "EmbeddingTextCleaner",
#             "normalize_unicode": True,
#             "remove_html_tags": True,
#             "collapse_whitespace": True,
#             "max_chars": 4000,
#             "aggressive_preprocessing": False,
#             "notes": (
#                 "Light cleaning only. No stemming, no lemmatization, "
#                 "no stopword removal, and no aggressive punctuation removal."
#             ),
#         },
#         "faiss_index_type": "IndexFlatIP",
#         "similarity": "cosine via normalized vectors + inner product",
#         "limit": limit,
#     }

#     store.save(
#         index_dir=index_dir,
#         metadata=metadata,
#     )

#     elapsed = time.time() - start

#     print("=" * 80)
#     print("DONE")
#     print(f"Seen documents   : {total_seen:,}")
#     print(f"Indexed documents: {total_indexed:,}")
#     print(f"Skipped documents: {total_seen - total_indexed:,}")
#     print(f"Vector dimension : {dimension}")
#     print(f"Saved to         : {index_dir}")
#     print(f"Total time       : {elapsed / 60:.2f} min")
#     print("=" * 80)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Build BERT/SentenceTransformer FAISS index from SQLite documents."
#     )

#     parser.add_argument(
#         "--db-path",
#         required=True,
#         help="Path to documents.sqlite",
#     )

#     parser.add_argument(
#         "--index-dir",
#         required=True,
#         help="Output directory for BERT FAISS index",
#     )

#     parser.add_argument(
#         "--model-name",
#         default="sentence-transformers/all-MiniLM-L6-v2",
#         help="SentenceTransformer model name",
#     )

#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=1000,
#         help="Limit documents for testing. Use larger values later.",
#     )

#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=32,
#         help="Embedding batch size",
#     )

#     args = parser.parse_args()

#     build_bert_index(
#         db_path=args.db_path,
#         index_dir=args.index_dir,
#         model_name=args.model_name,
#         limit=args.limit,
#         batch_size=args.batch_size,
#     )


# if __name__ == "__main__":
#     main()




import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.bert_index import build_bert_faiss_index


DEFAULT_DB_PATH = r"E:\ir_project_artifacts\documents.sqlite"
DEFAULT_INDEX_DIR = r"E:\ir_project_artifacts\indexes\faiss_bert_full"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build BERT FAISS index from SQLite documents."
    )

    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-compat-copy", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    build_bert_faiss_index(
        db_path=args.db_path,
        index_dir=args.index_dir,
        model_name=args.model_name,
        limit=args.limit,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        overwrite=args.overwrite,
        write_compat_copy=args.write_compat_copy,
    )


if __name__ == "__main__":
    main()