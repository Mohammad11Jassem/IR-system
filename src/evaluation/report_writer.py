import csv
import json
from pathlib import Path
from typing import Iterable


class EvaluationReportWriter:
    """
    Responsible only for persisting evaluation artifacts.

    This keeps EvaluationService focused on evaluation logic and makes the
    reporting part replaceable, which fits the SOA-style separation of concerns.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.per_query_dir = self.output_dir / "per_query"
        self.charts_dir = self.output_dir / "charts"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.per_query_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def save_run_jsonl(self, model_name: str, records: Iterable[dict]) -> Path:
        path = self.runs_dir / f"{model_name}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    def save_per_query_csv(self, model_name: str, rows: list[dict]) -> Path | None:
        if not rows:
            return None

        path = self.per_query_dir / f"{model_name}_per_query.csv"
        self._write_csv(path, rows)
        return path

    def save_summary_csv(self, rows: list[dict]) -> Path:
        path = self.output_dir / "evaluation_summary.csv"
        self._write_csv(path, rows)
        return path

    def save_summary_xlsx(self, summary_rows: list[dict], per_query_by_model: dict[str, list[dict]]) -> Path | None:
        """
        Optional Excel report. If openpyxl is not installed, the CSV files remain
        the primary reliable outputs.
        """
        if not summary_rows:
            return None

        try:
            import pandas as pd
        except ImportError:
            print("pandas is not installed. Skipping Excel report.")
            return None

        path = self.output_dir / "evaluation_summary.xlsx"

        try:
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)

                for model_name, rows in per_query_by_model.items():
                    if rows:
                        sheet_name = model_name[:31]
                        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

            return path
        except Exception as exc:
            print(f"Could not write Excel report: {exc}")
            return None

    def save_charts(self, summary_rows: list[dict]) -> list[Path]:
        if not summary_rows:
            return []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Skipping charts.")
            return []

        metric_keys = [
            key
            for key in summary_rows[0].keys()
            if key == "map"
            or key.startswith("precision_at_")
            or key.startswith("recall_at_")
            or key.startswith("ndcg_at_")
            or key == "avg_query_time_seconds"
        ]

        created = []
        models = [row["model"] for row in summary_rows]

        for metric in metric_keys:
            values = [float(row.get(metric, 0.0) or 0.0) for row in summary_rows]

            plt.figure(figsize=(10, 5))
            plt.bar(models, values)
            plt.title(metric)
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            path = self.charts_dir / f"{metric}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            created.append(path)

        return created

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = []
        seen = set()

        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
