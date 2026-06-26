from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

import ir_datasets
import pandas as pd


DATASET_IDS = {
    "main": "medline/2004/trec-genomics-2005",
}


def _record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record

    if hasattr(record, "_asdict"):
        return dict(record._asdict())

    if is_dataclass(record):
        return asdict(record)

    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def _records_to_dataframe(records: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame(_record_to_dict(record) for record in records)


def load_dataset(dataset_name: str = "main"):
    if dataset_name not in DATASET_IDS:
        allowed = ", ".join(sorted(DATASET_IDS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Allowed values: {allowed}"
        )

    return ir_datasets.load(DATASET_IDS[dataset_name])


def load_documents(dataset_name: str = "main", limit: int | None = None) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)

    records = dataset.docs_iter()
    if limit is not None:
        records = _take(records, limit)

    documents = _records_to_dataframe(records)

    required_columns = {"doc_id", "title", "abstract"}
    missing_columns = required_columns - set(documents.columns)

    if missing_columns:
        raise ValueError(f"Missing document columns: {sorted(missing_columns)}")

    documents["title"] = documents["title"].fillna("").astype(str)
    documents["abstract"] = documents["abstract"].fillna("").astype(str)
    documents["contents"] = (
        documents["title"].str.strip()
        + " "
        + documents["abstract"].str.strip()
    ).str.strip()

    return documents


def load_queries(dataset_name: str = "main") -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    queries = _records_to_dataframe(dataset.queries_iter())

    required_columns = {"query_id", "text"}
    missing_columns = required_columns - set(queries.columns)

    if missing_columns:
        raise ValueError(f"Missing query columns: {sorted(missing_columns)}")

    queries["text"] = queries["text"].fillna("").astype(str)

    return queries


def load_qrels(dataset_name: str = "main") -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    qrels = _records_to_dataframe(dataset.qrels_iter())

    required_columns = {"query_id", "doc_id", "relevance"}
    missing_columns = required_columns - set(qrels.columns)

    if missing_columns:
        raise ValueError(f"Missing qrels columns: {sorted(missing_columns)}")

    return qrels


def _take(records: Iterable[Any], limit: int):
    for index, record in enumerate(records):
        if index >= limit:
            break
        yield record