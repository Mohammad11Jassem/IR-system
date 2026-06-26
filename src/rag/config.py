from __future__ import annotations

import os
from pathlib import Path

# Paths can be overridden without editing code:
# PowerShell examples:
#   $env:IR_DOCUMENT_DB_PATH="E:\ir_project_artifacts\documents.sqlite"
#   $env:IR_BERT_INDEX_DIR="E:\ir_project_artifacts\indexes\faiss_bert_full"

DB_PATH = Path(os.getenv("IR_DOCUMENT_DB_PATH", r"E:\ir_project_artifacts\documents.sqlite"))

BERT_INDEX_DIR = Path(os.getenv("IR_BERT_INDEX_DIR", r"E:\ir_project_artifacts\indexes\faiss_bert_full"))
FAISS_INDEX_PATH = BERT_INDEX_DIR / "index.faiss"
DOC_IDS_PKL_PATH = BERT_INDEX_DIR / "doc_ids.pkl"
DOC_IDS_JSONL_PATH = BERT_INDEX_DIR / "doc_ids.jsonl"
METADATA_PATH = BERT_INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = os.getenv(
    "IR_EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Gemini settings. Never commit real API keys.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
