import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import RagPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--context-docs", type=int, default=5)
    args = parser.parse_args()

    rag = RagPipeline()

    start = time.time()

    result = rag.ask(
        question=args.question,
        retrieve_k=args.retrieve_k,
        context_docs=args.context_docs,
    )

    elapsed = time.time() - start

    print("=" * 100)
    print("QUESTION")
    print("=" * 100)
    print(result["question"])

    print("\n" + "=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(result["answer"])

    print("\n" + "=" * 100)
    print("SOURCES")
    print("=" * 100)

    for i, source in enumerate(result["sources"], start=1):
        print(f"\nSource {i}")
        print("Doc ID:", source["doc_id"])
        print("Score:", round(source["score"], 6))
        print("Title:", source.get("title", ""))

    print("\nTime:", round(elapsed, 3), "seconds")


if __name__ == "__main__":
    main()