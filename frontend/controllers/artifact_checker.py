from __future__ import annotations

from pathlib import Path


def _size_text(path: Path) -> str:
    if not path.exists() or path.is_dir():
        return "-"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def check_artifacts(index_path: str, db_path: str, bert_index_dir: str, word2vec_index_dir: str) -> str:
    checks = []

    def add(label: str, path_str: str, required_files: list[str] | None = None):
        path = Path(path_str)
        status = "✅" if path.exists() else "❌"
        details = []
        if path.exists():
            details.append("exists")
            if path.is_file():
                details.append(_size_text(path))
            if path.is_dir() and required_files:
                for f in required_files:
                    fp = path / f
                    details.append(f"{f}: {'✅' if fp.exists() else '❌'}")
        else:
            details.append("missing")
        checks.append((label, str(path), status, "; ".join(details)))

    add("SQLite document store", db_path)
    add("Terrier index", index_path)
    add("BERT FAISS index", bert_index_dir, ["index.faiss", "doc_ids.pkl", "metadata.json"])
    add("Word2Vec FAISS index", word2vec_index_dir, ["index.faiss", "doc_ids.pkl", "metadata.json"])

    rows = "".join(
        f"<tr><td>{status}</td><th>{label}</th><td><code>{path}</code></td><td>{details}</td></tr>"
        for label, path, status, details in checks
    )
    return f"""
    <div class='summary-card'>
      <div class='summary-title'>Artifact Status</div>
      <table class='meta-table'>
        <tr><th>Status</th><th>Artifact</th><th>Path</th><th>Details</th></tr>
        {rows}
      </table>
    </div>
    """
