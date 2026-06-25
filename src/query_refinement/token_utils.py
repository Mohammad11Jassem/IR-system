from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")
GENE_LIKE_PATTERN = re.compile(
    r"^(?:[a-z]+\d+[a-z0-9-]*|p\d+|il-\d+|tnf-alpha|brca\d+|cftr|prnp|apoe|gstm\d+|nm\d+)$",
    re.IGNORECASE,
)

DEFAULT_STOP_TERMS = {
    # General English stopwords
    "a", "an", "the",
    "and", "or", "but",
    "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did",
    "have", "has", "had",
    "can", "could", "should", "would", "may", "might", "must",
    "this", "that", "these", "those",
    "there", "their", "them", "they", "it", "its",
    "we", "our", "you", "your",
    "he", "she", "his", "her",
    "in", "on", "at", "by", "for", "from", "to", "of", "with", "without",
    "as", "into", "about", "between", "within", "during", "through",
    "than", "such", "also", "more", "most", "other",

    # Query boilerplate terms from TREC-style topics
    "provide", "provides", "provided", "providing",
    "information", "info",
    "describe", "describes", "described",
    "procedure", "procedures",
    "method", "methods",
    "role", "roles",
    "process", "processes",
    "effect", "effects",
    "impact", "impacts",
    "different",
    "exact",
    "take", "takes",
    "place",
}


def tokenize_terms(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(str(text).lower())


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = str(item).strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def is_gene_like(token: str) -> bool:
    return bool(GENE_LIKE_PATTERN.match(str(token).strip()))


# def is_meaningful_token(
#     token: str,
#     stop_terms: set[str],
#     min_token_length: int = 2,
#     allow_numeric_tokens: bool = False,
# ) -> bool:
#     token = str(token).strip().lower()
#     if not token:
#         return False

#     if token in stop_terms:
#         return False

#     if token.isdigit() and not allow_numeric_tokens:
#         return False

#     # Keep short biomedical symbols such as p53 and IL-2.
#     if len(token) < min_token_length and not is_gene_like(token):
#         return False

#     # Avoid punctuation-only / malformed expansion terms.
#     if not re.search(r"[a-zA-Z]", token) and not allow_numeric_tokens:
#         return False

#     return True


def is_meaningful_token(
    token: str,
    stop_terms: set[str] | None = None,
    min_token_length: int = 3,
    allow_numeric_tokens: bool = False,
) -> bool:
    token = str(token).lower().strip()

    if not token:
        return False

    combined_stop_terms = set(DEFAULT_STOP_TERMS)
    if stop_terms:
        combined_stop_terms.update(stop_terms)

    if token in combined_stop_terms:
        return False

    if len(token) < min_token_length:
        return False

    if not allow_numeric_tokens and token.isnumeric():
        return False

    return True


def document_to_text(document: dict[str, Any] | Any) -> str:
    """Return a robust text representation from retriever result dictionaries."""
    if document is None:
        return ""
    if not isinstance(document, dict):
        return str(document)

    parts = []
    for key in ("title", "abstract", "contents", "text", "snippet"):
        value = document.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def top_doc_ids(output: dict[str, Any], top_k: int) -> list[str]:
    results = output.get("results", []) if isinstance(output, dict) else []
    ids: list[str] = []
    for item in results[:top_k]:
        doc_id = item.get("doc_id") if isinstance(item, dict) else None
        if doc_id is not None:
            ids.append(str(doc_id))
    return ids
