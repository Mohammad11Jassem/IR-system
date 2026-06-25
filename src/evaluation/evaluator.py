import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import Qrels, Run, evaluate_run
from src.evaluation.report_writer import EvaluationReportWriter


@dataclass
class EvaluationConfig:
    run_depth: int = 100
    precision_k: int = 10
    recall_k: int | None = None
    ndcg_k: int = 10
    map_depth: int | None = None
    min_relevance: int = 1
    continue_on_error: bool = True
    save_charts: bool = True
    save_excel: bool = True

    def resolved_recall_k(self) -> int:
        return self.recall_k if self.recall_k is not None else self.run_depth

    def resolved_map_depth(self) -> int:
        return self.map_depth if self.map_depth is not None else self.run_depth


class EvaluationService:
    """
    Evaluation Service for the IR system.

    Responsibilities:
    - Accept already-built retrievers.
    - Run the same query set against each retriever.
    - Convert retriever outputs into TREC-like ranked runs.
    - Compute MAP, Precision@K, Recall@K, and nDCG@K.
    - Delegate persistence to EvaluationReportWriter.

    This is the real service. scripts/evaluate_retrievers.py should only be a
    CLI entry point that wires dependencies and calls this class.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/evaluation",
        config: EvaluationConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or EvaluationConfig()
        self.writer = EvaluationReportWriter(self.output_dir)

    @staticmethod
    def build_qrels_dict(qrels_df: pd.DataFrame) -> Qrels:
        required_columns = {"query_id", "doc_id", "relevance"}
        missing = required_columns - set(qrels_df.columns)
        if missing:
            raise ValueError(f"Missing qrels columns: {sorted(missing)}")

        qrels: Qrels = {}

        for _, row in qrels_df.iterrows():
            qid = str(row["query_id"])
            doc_id = str(row["doc_id"])
            relevance = int(row["relevance"])

            qrels.setdefault(qid, {})[doc_id] = relevance

        return qrels

    @staticmethod
    def _query_text_from_row(row: pd.Series) -> str:
        if "text" in row:
            return str(row["text"])
        if "query" in row:
            return str(row["query"])
        raise ValueError("Queries dataframe must contain either 'text' or 'query'.")

    @staticmethod
    def _query_id_from_row(row: pd.Series) -> str:
        if "query_id" not in row:
            raise ValueError("Queries dataframe must contain 'query_id'.")
        return str(row["query_id"])

    @staticmethod
    def _normalize_retriever_output(output: Any) -> list[dict]:
        """
        Supports the official project format:
          {"results": [{"doc_id": ..., "score": ...}, ...]}

        Also supports older prototype format:
          [(doc_id, score), ...]
        """
        if isinstance(output, dict):
            raw_results = output.get("results", [])
        elif isinstance(output, list):
            raw_results = output
        else:
            raise TypeError(f"Unsupported retriever output type: {type(output).__name__}")

        normalized = []

        for rank, item in enumerate(raw_results, start=1):
            if isinstance(item, dict):
                doc_id = item.get("doc_id")
                if doc_id is None:
                    continue

                normalized.append(
                    {
                        "rank": int(item.get("rank", rank)),
                        "doc_id": str(doc_id),
                        "score": float(item.get("score", 0.0) or 0.0),
                        "title": item.get("title"),
                        "abstract": item.get("abstract"),
                    }
                )

            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                normalized.append(
                    {
                        "rank": rank,
                        "doc_id": str(item[0]),
                        "score": float(item[1]),
                        "title": None,
                        "abstract": None,
                    }
                )

        normalized.sort(key=lambda x: x["rank"])
        return normalized

    def evaluate_models(
        self,
        retrievers: dict[str, Any],
        queries_df: pd.DataFrame,
        qrels_df: pd.DataFrame,
    ) -> dict:
        qrels = self.build_qrels_dict(qrels_df)

        summary_rows = []
        per_query_by_model: dict[str, list[dict]] = {}
        run_paths: dict[str, str] = {}

        for model_name, retriever in retrievers.items():
            result = self.evaluate_single_model(
                model_name=model_name,
                retriever=retriever,
                queries_df=queries_df,
                qrels=qrels,
            )

            summary_rows.append(result["summary"])
            per_query_by_model[model_name] = result["per_query"]
            run_paths[model_name] = str(result["run_path"])

        summary_csv = self.writer.save_summary_csv(summary_rows)
        summary_xlsx = None
        chart_paths = []

        if self.config.save_excel:
            summary_xlsx = self.writer.save_summary_xlsx(
                summary_rows=summary_rows,
                per_query_by_model=per_query_by_model,
            )

        if self.config.save_charts:
            chart_paths = self.writer.save_charts(summary_rows)

        return {
            "summary_rows": summary_rows,
            "per_query_by_model": per_query_by_model,
            "run_paths": run_paths,
            "summary_csv": str(summary_csv),
            "summary_xlsx": str(summary_xlsx) if summary_xlsx else None,
            "charts": [str(path) for path in chart_paths],
        }

    def evaluate_single_model(
        self,
        model_name: str,
        retriever: Any,
        queries_df: pd.DataFrame,
        qrels: Qrels,
    ) -> dict:
        print("=" * 90)
        print(f"Evaluating: {model_name}")
        print("=" * 90)

        run: Run = {}
        run_records = []
        per_query_rows = []
        total_query_time = 0.0
        failed_queries = 0

        wall_start = time.time()

        for index, row in queries_df.reset_index(drop=True).iterrows():
            qid = self._query_id_from_row(row)
            query_text = self._query_text_from_row(row)

            print(f"[{index + 1}/{len(queries_df)}] qid={qid} | {query_text[:80]}")

            query_start = time.time()
            error_message = None
            output = None
            results = []

            try:
                output = retriever.search(query_text)
                results = self._normalize_retriever_output(output)[: self.config.run_depth]
            except Exception as exc:
                failed_queries += 1
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"ERROR qid={qid}: {error_message}")

                if not self.config.continue_on_error:
                    raise

            query_time = 0.0
            if isinstance(output, dict) and output.get("time_seconds") is not None:
                query_time = float(output.get("time_seconds", 0.0) or 0.0)
            else:
                query_time = time.time() - query_start

            total_query_time += query_time

            retrieved_doc_ids = [item["doc_id"] for item in results]
            run[qid] = retrieved_doc_ids

            run_records.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "model": model_name,
                    "time_seconds": query_time,
                    "returned_results": len(results),
                    "error": error_message,
                    "retrieved": results,
                }
            )

        metrics = evaluate_run(
            qrels=qrels,
            run=run,
            precision_k=self.config.precision_k,
            recall_k=self.config.resolved_recall_k(),
            ndcg_k=self.config.ndcg_k,
            map_depth=self.config.resolved_map_depth(),
            min_relevance=self.config.min_relevance,
            include_per_query=True,
        )

        metric_per_query = {
            row["query_id"]: row
            for row in metrics.pop("per_query", [])
        }

        for record in run_records:
            qid = record["query_id"]
            per_query_rows.append(
                {
                    "model": model_name,
                    "query_id": qid,
                    "query": record["query"],
                    "time_seconds": record["time_seconds"],
                    "returned_results": record["returned_results"],
                    "error": record["error"],
                    **metric_per_query.get(qid, {}),
                }
            )

        run_path = self.writer.save_run_jsonl(model_name, run_records)
        per_query_path = self.writer.save_per_query_csv(model_name, per_query_rows)

        wall_time = time.time() - wall_start
        num_queries = len(queries_df)

        summary = {
            "model": model_name,
            "run_depth": self.config.run_depth,
            "precision_k": self.config.precision_k,
            "recall_k": self.config.resolved_recall_k(),
            "ndcg_k": self.config.ndcg_k,
            "map_depth": self.config.resolved_map_depth(),
            **metrics,
            "failed_queries": failed_queries,
            "avg_query_time_seconds": total_query_time / max(num_queries, 1),
            "total_query_time_seconds": total_query_time,
            "wall_time_seconds": wall_time,
            "run_file": str(run_path),
            "per_query_file": str(per_query_path) if per_query_path else "",
        }

        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return {
            "summary": summary,
            "per_query": per_query_rows,
            "run_path": run_path,
        }
