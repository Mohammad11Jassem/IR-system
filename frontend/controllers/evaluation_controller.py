from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_LABEL_TO_VALUE = {
    "BM25": "bm25",
    "TF-IDF": "tfidf",
    "Word2Vec": "word2vec",
    "BERT": "bert",
    "Parallel Basic": "parallel_basic",
    "Parallel Full": "parallel_full",
    "Serial BM25 → Word2Vec": "serial_bm25_word2vec",
    "Serial BM25 → BERT": "serial_bm25_bert",
}


def build_evaluation_command(
    model_labels: Iterable[str],
    limit_queries: int | None,
    run_depth: int,
    bm25_k1: float,
    bm25_b: float,
    precision_k: int,
    recall_k: int,
    ndcg_k: int,
    map_depth: int,
    output_dir: str,
    index_path: str,
    db_path: str,
    bert_index_dir: str,
    bert_model_name: str,
    word2vec_index_dir: str,
    enable_refinement: bool,
    refinement_method: str,
    prf_feedback_docs: int,
    max_prf_terms: int,
    max_expansion_terms: int,
    word2vec_topn: int,
    min_word2vec_similarity: float,
    original_term_weight: int,
) -> list[str]:
    models = [MODEL_LABEL_TO_VALUE.get(m, m).lower().strip() for m in model_labels]
    if not models:
        raise ValueError("Select at least one model to evaluate.")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_retrievers.py"),
        "--models",
        *models,
        "--run-depth", str(int(run_depth)),
        "--bm25-k1", str(float(bm25_k1)),
        "--bm25-b", str(float(bm25_b)),
        "--precision-k", str(int(precision_k)),
        "--recall-k", str(int(recall_k)),
        "--ndcg-k", str(int(ndcg_k)),
        "--map-depth", str(int(map_depth)),
        "--output-dir", output_dir,
        "--index-path", index_path,
        "--db-path", db_path,
        "--bert-index-dir", bert_index_dir,
        "--bert-model-name", bert_model_name,
        "--word2vec-index-dir", word2vec_index_dir,
    ]

    if limit_queries and int(limit_queries) > 0:
        cmd += ["--limit-queries", str(int(limit_queries))]

    if enable_refinement:
        cmd += [
            "--enable-query-refinement",
            "--refinement-method", refinement_method,
            "--refinement-prf-feedback-docs", str(int(prf_feedback_docs)),
            "--refinement-max-prf-terms", str(int(max_prf_terms)),
            "--refinement-max-expansion-terms", str(int(max_expansion_terms)),
            "--refinement-word2vec-topn", str(int(word2vec_topn)),
            "--refinement-min-word2vec-similarity", str(float(min_word2vec_similarity)),
            "--refinement-original-term-weight", str(int(original_term_weight)),
        ]
    return cmd


def command_preview(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_evaluation(*args) -> tuple[str, str]:
    cmd = build_evaluation_command(*args)
    preview = command_preview(cmd)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        return preview, f"ERROR while launching evaluation:\n{exc}"

    output = []
    if completed.stdout:
        output.append(completed.stdout)
    if completed.stderr:
        output.append("\n--- STDERR ---\n" + completed.stderr)
    output.append(f"\nExit code: {completed.returncode}")
    return preview, "".join(output)
