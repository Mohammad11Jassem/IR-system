from __future__ import annotations

import html
import json
from typing import Any


def _safe(value: Any) -> str:
    return html.escape(str(value or ""))


def _short(text: str | None, limit: int = 700) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def render_refinement(output: dict[str, Any]) -> str:
    refinement = output.get("refinement") or {}
    if not refinement and not output.get("refined_query"):
        return "<div class='empty-state'>Query refinement is disabled or did not change the query.</div>"

    rows = []
    def add(label: str, value: Any):
        rows.append(f"<tr><th>{_safe(label)}</th><td>{_safe(value)}</td></tr>")

    add("Original Query", output.get("query"))
    add("Processed Query", output.get("processed_query"))
    add("Refined Query", output.get("refined_query"))
    add("Method", refinement.get("method"))
    add("Changed", refinement.get("changed"))
    if "prf_terms" in refinement:
        add("PRF Terms", ", ".join(map(str, refinement.get("prf_terms") or [])))
    if "word2vec_terms" in refinement:
        add("Word2Vec Terms", ", ".join(map(str, refinement.get("word2vec_terms") or [])))
    if "history_terms" in refinement:
        add("History Terms", ", ".join(map(str, refinement.get("history_terms") or [])))
    if "safety_gate" in refinement:
        add("Safety Gate", refinement.get("safety_gate"))
    if "safety_reason" in refinement:
        add("Safety Reason", refinement.get("safety_reason"))
    if "overlap" in refinement:
        add("Overlap", refinement.get("overlap"))

    return "<table class='meta-table'>" + "".join(rows) + "</table>"



def render_qrels_summary(output: dict[str, Any]) -> str:
    official = output.get("official_query") or {}
    summary = output.get("qrels_summary") or {}
    if not official.get("enabled"):
        return ""

    return f"""
      <div class='qrels-summary'>
        <div><b>Official qid:</b> {_safe(official.get('qid'))}</div>
        <div><b>Qrels for qid:</b> {_safe(official.get('qrels_count_for_qid'))}</div>
        <div><b>Relevant in returned results:</b> {_safe(summary.get('relevant_count'))}</div>
        <div><b>Non-relevant:</b> {_safe(summary.get('non_relevant_count'))}</div>
        <div><b>Unjudged:</b> {_safe(summary.get('unjudged_count'))}</div>
        <div><b>Precision@returned:</b> {_safe(round(float(summary.get('precision_at_returned', 0.0)), 4))}</div>
      </div>
    """

def render_results(output: dict[str, Any]) -> str:
    if not output:
        return "<div class='empty-state'>No search has been executed yet.</div>"

    model = _safe(output.get("model"))
    query = _safe(output.get("query"))
    processed = _safe(output.get("processed_query"))
    refined = _safe(output.get("refined_query"))
    elapsed = output.get("time_seconds", output.get("ui_total_time_seconds", 0.0))
    results = output.get("results") or []

    header = f"""
    <div class='summary-card'>
      <div class='summary-title'>Search Summary</div>
      <div class='summary-grid'>
        <div><span>Model</span><strong>{model}</strong></div>
        <div><span>Results</span><strong>{len(results)}</strong></div>
        <div><span>Time</span><strong>{float(elapsed):.4f}s</strong></div>
      </div>
      {render_qrels_summary(output)}
      <div class='query-line'><b>Query:</b> {query}</div>
      {f"<div class='query-line'><b>Processed:</b> {processed}</div>" if processed else ""}
      {f"<div class='query-line'><b>Refined:</b> {refined}</div>" if refined else ""}
    </div>
    """

    if not results:
        return header + "<div class='empty-state'>No results found.</div>"

    cards = []
    for item in results:
        rank = _safe(item.get("rank"))
        doc_id = _safe(item.get("doc_id"))
        score = item.get("score")
        title = _safe(item.get("title"))
        abstract = _safe(_short(item.get("abstract"), 900))
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else _safe(score)

        qrels_badge = ""
        if "qrels_status" in item:
            status = str(item.get("qrels_status"))
            rel = item.get("qrels_relevance")
            if status == "relevant":
                qrels_badge = f"<div class='qrels-badge relevant'>Relevant | grade={_safe(rel)}</div>"
            elif status == "non_relevant":
                qrels_badge = f"<div class='qrels-badge non-relevant'>Non-relevant | grade={_safe(rel)}</div>"
            else:
                qrels_badge = "<div class='qrels-badge unjudged'>Unjudged in qrels</div>"

        extra_parts = []
        if "first_stage_rank" in item:
            extra_parts.append(
                f"<div class='badge'>First-stage rank: {_safe(item.get('first_stage_rank'))}</div>"
            )
        if "rerank_score" in item:
            extra_parts.append(
                f"<div class='badge'>Rerank score: {_safe(round(float(item.get('rerank_score')), 4))}</div>"
            )
        if item.get("model_contributions"):
            contrib_html = "".join(
                f"<li><b>{_safe(name)}</b>: rank={_safe(c.get('rank'))}, score={_safe(round(float(c.get('score', 0)), 4))}, rrf={_safe(round(float(c.get('rrf_contribution', 0)), 6))}</li>"
                for name, c in item["model_contributions"].items()
            )
            extra_parts.append(f"<details><summary>Model contributions</summary><ul>{contrib_html}</ul></details>")

        cards.append(f"""
        <div class='result-card'>
          <div class='result-top'>
            <div class='rank'>#{rank}</div>
            <div class='score'>score: {score_text}</div>
          </div>
          <div class='doc-id'>doc_id: {doc_id}</div>
          {qrels_badge}
          <div class='result-title'>{title}</div>
          <div class='abstract'>{abstract}</div>
          <div class='extra'>{''.join(extra_parts)}</div>
        </div>
        """)

    return header + "".join(cards)


def render_json(output: dict[str, Any]) -> str:
    return json.dumps(output or {}, ensure_ascii=False, indent=2, default=str)


def render_context_sources(results: list[dict[str, Any]], max_docs: int = 5) -> str:
    cards = []
    for item in results[:max_docs]:
        cards.append(f"""
        <div class='source-card'>
          <div><b>#{_safe(item.get('rank'))}</b> | doc_id: {_safe(item.get('doc_id'))} | score: {_safe(item.get('score'))}</div>
          <div class='result-title'>{_safe(item.get('title'))}</div>
          <div class='abstract'>{_safe(_short(item.get('abstract'), 500))}</div>
        </div>
        """)
    return "".join(cards) if cards else "<div class='empty-state'>No sources retrieved.</div>"
