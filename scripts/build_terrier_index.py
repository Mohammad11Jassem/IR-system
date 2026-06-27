import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.terrier_index import build_terrier_index


FULL_INDEX_NAMES = {
    "terrier_medline",
    "terrier_full",
    "medline_full",
}


def is_full_index_path(index_path: str) -> bool:
    path = Path(index_path)
    return path.name.lower() in FULL_INDEX_NAMES


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to documents.sqlite",
    )

    parser.add_argument(
        "--index-path",
        required=True,
        help="Output Terrier index directory",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Terrier index",
    )

    parser.add_argument(
        "--confirm-full-overwrite",
        action="store_true",
        help="Required when overwriting the full Terrier index",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional document limit for testing",
    )

    args = parser.parse_args()

    if (
        args.overwrite
        and args.limit is None
        and is_full_index_path(args.index_path)
        and not args.confirm_full_overwrite
    ):
        raise SystemExit(
            "\nRefusing to overwrite the FULL Terrier index without explicit confirmation.\n"
            "This protects the completed full index from accidental rebuilds.\n\n"
            "If you really want to rebuild it, run again with:\n"
            "  --confirm-full-overwrite\n"
        )

    build_terrier_index(
        db_path=args.db_path,
        index_path=args.index_path,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()