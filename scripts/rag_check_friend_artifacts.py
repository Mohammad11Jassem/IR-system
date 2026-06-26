import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import (
    DB_PATH,
    FAISS_INDEX_PATH,
    DOC_IDS_PKL_PATH,
    DOC_IDS_JSONL_PATH,
    METADATA_PATH,
)
from src.rag.document_repository import DocumentRepository
from src.rag.faiss_retriever import BertFaissRetriever


def main():
    print("Checking RAG integration...")
    print("DB_PATH:", DB_PATH, "exists:", DB_PATH.exists())
    print("FAISS_INDEX_PATH:", FAISS_INDEX_PATH, "exists:", FAISS_INDEX_PATH.exists())
    print("DOC_IDS_PKL_PATH:", DOC_IDS_PKL_PATH, "exists:", DOC_IDS_PKL_PATH.exists())
    print("DOC_IDS_JSONL_PATH:", DOC_IDS_JSONL_PATH, "exists:", DOC_IDS_JSONL_PATH.exists())
    print("METADATA_PATH:", METADATA_PATH, "exists:", METADATA_PATH.exists())

    repo = DocumentRepository()
    print("Documents count:", repo.count_documents())

    retriever = BertFaissRetriever()
    print("FAISS vectors:", retriever.index.ntotal)
    print("Doc IDs:", len(retriever.doc_ids))
    print("Embedding model:", retriever.model_name)

    results = retriever.search("breast cancer gene expression", top_k=3)

    print("\nSample search results:")
    for item in results:
        print(item["doc_id"], item["score"], item["title"][:100])


if __name__ == "__main__":
    main()