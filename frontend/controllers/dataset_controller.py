from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_ID = "medline/2004/trec-genomics-2005"


@dataclass(frozen=True)
class OfficialQuery:
    qid: str
    text: str

    @property
    def label(self) -> str:
        preview = " ".join(self.text.split())
        if len(preview) > 110:
            preview = preview[:110].rstrip() + "..."
        return f"qid={self.qid} | {preview}"


def _require_ir_datasets():
    try:
        import ir_datasets  # type: ignore
        return ir_datasets
    except Exception as exc:  # pragma: no cover - UI error path
        raise RuntimeError(
            "The package 'ir_datasets' is required to load official queries and qrels. "
            "Install it with: python -m pip install ir_datasets"
        ) from exc


@lru_cache(maxsize=4)
def _load_dataset(dataset_id: str):
    ir_datasets = _require_ir_datasets()
    return ir_datasets.load(dataset_id)


def _get_attr(obj: Any, names: list[str], default: str = "") -> str:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    return default


def _extract_query(raw_query: Any) -> OfficialQuery:
    qid = _get_attr(raw_query, ["query_id", "qid", "id"], default="")
    text = _get_attr(
        raw_query,
        ["text", "query", "title", "description", "narrative"],
        default="",
    )

    # Some IR dataset query objects expose useful fields but no single text field.
    if not text:
        parts = []
        for attr in ["title", "description", "narrative"]:
            if hasattr(raw_query, attr):
                value = getattr(raw_query, attr)
                if value:
                    parts.append(str(value).strip())
        text = " ".join(parts).strip()

    if not text:
        text = str(raw_query)

    return OfficialQuery(qid=str(qid), text=text)


@lru_cache(maxsize=4)
def load_official_queries(dataset_id: str = DEFAULT_DATASET_ID) -> tuple[OfficialQuery, ...]:
    dataset = _load_dataset(dataset_id)
    queries = []
    for raw_query in dataset.queries_iter():
        query = _extract_query(raw_query)
        if query.qid and query.text:
            queries.append(query)
    queries.sort(key=lambda q: q.qid)
    return tuple(queries)


def official_query_choices(dataset_id: str = DEFAULT_DATASET_ID) -> list[str]:
    return [q.label for q in load_official_queries(dataset_id)]


def parse_qid_from_label(label: str | None) -> str | None:
    if not label:
        return None
    label = str(label).strip()
    if label.startswith("qid="):
        return label.split("|", 1)[0].replace("qid=", "").strip()
    return None


def query_text_from_label(label: str | None, dataset_id: str = DEFAULT_DATASET_ID) -> tuple[str | None, str | None]:
    qid = parse_qid_from_label(label)
    if not qid:
        return None, None
    for query in load_official_queries(dataset_id):
        if query.qid == qid:
            return query.text, query.qid
    return None, qid


@lru_cache(maxsize=128)
def load_qrels_for_qid(dataset_id: str, qid: str) -> dict[str, int]:
    dataset = _load_dataset(dataset_id)
    qrels: dict[str, int] = {}
    for qrel in dataset.qrels_iter():
        qrel_qid = _get_attr(qrel, ["query_id", "qid"], default="")
        if str(qrel_qid) != str(qid):
            continue
        doc_id = _get_attr(qrel, ["doc_id", "docno", "document_id"], default="")
        rel_text = _get_attr(qrel, ["relevance", "rel", "grade", "iteration"], default="0")
        try:
            relevance = int(float(rel_text))
        except ValueError:
            relevance = 0
        if doc_id:
            qrels[str(doc_id)] = relevance
    return qrels


def judge_output_with_qrels(output: dict[str, Any], dataset_id: str, qid: str) -> dict[str, Any]:
    """Attach qrels judgments to retrieved results for a selected official query."""
    if not qid:
        return output

    qrels = load_qrels_for_qid(dataset_id, str(qid))
    results = output.get("results") or []

    relevant_count = 0
    non_relevant_count = 0
    unjudged_count = 0
    judged_count = 0

    for item in results:
        doc_id = str(item.get("doc_id", ""))
        rel = qrels.get(doc_id)

        if rel is None:
            item["qrels_status"] = "unjudged"
            item["qrels_relevance"] = None
            item["is_relevant"] = False
            unjudged_count += 1
        else:
            judged_count += 1
            item["qrels_relevance"] = rel
            if rel > 0:
                item["qrels_status"] = "relevant"
                item["is_relevant"] = True
                relevant_count += 1
            else:
                item["qrels_status"] = "non_relevant"
                item["is_relevant"] = False
                non_relevant_count += 1

    total_returned = len(results)
    output["official_query"] = {
        "enabled": True,
        "dataset_id": dataset_id,
        "qid": str(qid),
        "qrels_count_for_qid": len(qrels),
    }
    output["qrels_summary"] = {
        "evaluated_results": total_returned,
        "judged_count": judged_count,
        "relevant_count": relevant_count,
        "non_relevant_count": non_relevant_count,
        "unjudged_count": unjudged_count,
        "precision_at_returned": (relevant_count / total_returned) if total_returned else 0.0,
    }
    return output
