import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gensim.models import Word2Vec

from src.embeddings import FaissVectorStore, Word2VecVectorizer
from src.preprocessing import EmbeddingTextCleaner


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")


def tokenize_for_word2vec(text: str) -> list[str]:
    """
    Biomedical-friendly tokenizer for Word2Vec.

    Keeps tokens such as:
    - brca1
    - brca-1
    - il-2
    - tnf-alpha
    - p53
    """
    if not text:
        return []

    text = text.lower()
    return TOKEN_PATTERN.findall(text)


def build_embedding_text(row: sqlite3.Row) -> str:
    title = row["title"] or ""
    abstract = row["abstract"] or ""
    contents = row["contents"] or ""

    text = f"{title}. {abstract}".strip()

    if not text:
        text = contents.strip()

    return text


def iter_sqlite_tokenized_documents(
    db_path: str,
    cleaner: EmbeddingTextCleaner,
    limit: Optional[int] = None,
) -> Iterator[tuple[str, list[str]]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if limit is None:
        cursor = conn.execute(
            """
            SELECT doc_id, title, abstract, contents
            FROM documents
            """
        )
    else:
        cursor = conn.execute(
            """
            SELECT doc_id, title, abstract, contents
            FROM documents
            LIMIT ?
            """,
            (limit,),
        )

    try:
        for row in cursor:
            doc_id = str(row["doc_id"])

            raw_text = build_embedding_text(row)
            cleaned_text = cleaner.clean(raw_text)
            tokens = tokenize_for_word2vec(cleaned_text)

            if not tokens:
                continue

            yield doc_id, tokens

    finally:
        conn.close()


class SQLiteTokenCorpus:
    """
    Iterable corpus for gensim Word2Vec.

    Important:
    This avoids loading all documents into memory.
    Gensim can iterate over this corpus multiple times.
    """

    def __init__(
        self,
        db_path: str,
        cleaner: EmbeddingTextCleaner,
        limit: Optional[int],
    ):
        self.db_path = db_path
        self.cleaner = cleaner
        self.limit = limit

    def __iter__(self) -> Iterator[list[str]]:
        for _, tokens in iter_sqlite_tokenized_documents(
            db_path=self.db_path,
            cleaner=self.cleaner,
            limit=self.limit,
        ):
            yield tokens


def train_word2vec_model(
    db_path: str,
    cleaner: EmbeddingTextCleaner,
    limit: Optional[int],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    epochs: int,
) -> Word2Vec:
    corpus = SQLiteTokenCorpus(
        db_path=db_path,
        cleaner=cleaner,
        limit=limit,
    )

    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )

    print("Building Word2Vec vocabulary...")
    model.build_vocab(corpus)

    print(f"Vocabulary size: {len(model.wv):,}")

    if len(model.wv) == 0:
        raise RuntimeError(
            "Word2Vec vocabulary is empty. Try lowering min_count or increasing limit."
        )

    print("Training Word2Vec model...")
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=epochs,
    )

    return model


def build_word2vec_index(
    db_path: str,
    index_dir: str,
    limit: Optional[int],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    epochs: int,
    batch_size: int,
) -> None:
    print("=" * 80)
    print("BUILD WORD2VEC FAISS INDEX")
    print("=" * 80)
    print(f"SQLite DB : {db_path}")
    print(f"Index dir : {index_dir}")
    print(f"Limit     : {limit if limit is not None else 'FULL DATASET'}")
    print(f"Vector size: {vector_size}")
    print(f"Window    : {window}")
    print(f"Min count : {min_count}")
    print(f"Workers   : {workers}")
    print(f"SG        : {sg}  (1=skip-gram, 0=CBOW)")
    print(f"Epochs    : {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    start = time.time()

    cleaner = EmbeddingTextCleaner(
        normalize_unicode=True,
        remove_html_tags=True,
        collapse_whitespace=True,
        max_chars=4000,
    )

    model = train_word2vec_model(
        db_path=db_path,
        cleaner=cleaner,
        limit=limit,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    vectorizer = Word2VecVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )
    vectorizer.model = model

    store = FaissVectorStore(
        dimension=vector_size,
        normalize_vectors=True,
    )

    batch_doc_ids: list[str] = []
    batch_tokens: list[list[str]] = []

    total_seen = 0
    total_indexed = 0
    skipped_empty_vectors = 0

    print("Encoding documents and building FAISS index...")

    for doc_id, tokens in iter_sqlite_tokenized_documents(
        db_path=db_path,
        cleaner=cleaner,
        limit=limit,
    ):
        total_seen += 1

        batch_doc_ids.append(doc_id)
        batch_tokens.append(tokens)

        if len(batch_tokens) >= batch_size:
            vectors = vectorizer.encode_batch(batch_tokens)

            valid_doc_ids = []
            valid_vectors = []

            for current_doc_id, vector in zip(batch_doc_ids, vectors):
                if vector.any():
                    valid_doc_ids.append(current_doc_id)
                    valid_vectors.append(vector)
                else:
                    skipped_empty_vectors += 1

            if valid_vectors:
                import numpy as np

                store.add(
                    vectors=np.vstack(valid_vectors).astype("float32"),
                    doc_ids=valid_doc_ids,
                )

                total_indexed += len(valid_doc_ids)

            elapsed = time.time() - start
            print(
                f"Indexed {total_indexed:,} documents "
                f"| seen: {total_seen:,} "
                f"| skipped empty vectors: {skipped_empty_vectors:,} "
                f"| elapsed: {elapsed / 60:.2f} min"
            )

            batch_doc_ids = []
            batch_tokens = []

    if batch_tokens:
        vectors = vectorizer.encode_batch(batch_tokens)

        valid_doc_ids = []
        valid_vectors = []

        for current_doc_id, vector in zip(batch_doc_ids, vectors):
            if vector.any():
                valid_doc_ids.append(current_doc_id)
                valid_vectors.append(vector)
            else:
                skipped_empty_vectors += 1

        if valid_vectors:
            import numpy as np

            store.add(
                vectors=np.vstack(valid_vectors).astype("float32"),
                doc_ids=valid_doc_ids,
            )

            total_indexed += len(valid_doc_ids)

    output_dir = Path(index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    word2vec_model_path = output_dir / "word2vec.model"
    vectorizer.save(str(word2vec_model_path))

    metadata = {
        "model_type": "word2vec",
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "workers": workers,
        "sg": sg,
        "sg_description": "1=skip-gram, 0=CBOW",
        "epochs": epochs,
        "text_fields": "title + abstract, fallback to contents",
        "tokenizer": "biomedical regex tokenizer preserving hyphenated terms",
        "text_cleaning": {
            "type": "EmbeddingTextCleaner",
            "normalize_unicode": True,
            "remove_html_tags": True,
            "collapse_whitespace": True,
            "max_chars": 4000,
            "aggressive_preprocessing": False,
        },
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine via normalized vectors + inner product",
        "word2vec_model_path": "word2vec.model",
        "limit": limit,
        "vocabulary_size": len(model.wv),
        "skipped_empty_vectors": skipped_empty_vectors,
    }

    store.save(
        index_dir=index_dir,
        metadata=metadata,
    )

    # store.save writes metadata.json. We rewrite it to include complete W2V metadata
    # after adding FAISS-generated fields such as dimension and num_documents.
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        saved_metadata = json.load(f)

    saved_metadata.update(metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(saved_metadata, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start

    print("=" * 80)
    print("DONE")
    print(f"Seen documents           : {total_seen:,}")
    print(f"Indexed documents        : {total_indexed:,}")
    print(f"Skipped empty vectors    : {skipped_empty_vectors:,}")
    print(f"Vocabulary size          : {len(model.wv):,}")
    print(f"Vector dimension         : {vector_size}")
    print(f"Saved to                 : {index_dir}")
    print(f"Word2Vec model           : {word2vec_model_path}")
    print(f"Total time               : {elapsed / 60:.2f} min")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec and build FAISS index from SQLite documents."
    )

    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to documents.sqlite",
    )

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Output directory for Word2Vec FAISS index",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit documents for testing. Use larger values later.",
    )

    parser.add_argument(
        "--vector-size",
        type=int,
        default=100,
        help="Word2Vec vector size.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Word2Vec context window size.",
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Ignore words with total frequency lower than this.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )

    parser.add_argument(
        "--sg",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=skip-gram, 0=CBOW.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for document encoding.",
    )

    args = parser.parse_args()

    build_word2vec_index(
        db_path=args.db_path,
        index_dir=args.index_dir,
        limit=args.limit,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()