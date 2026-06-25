import ir_datasets
import time
from pathlib import Path

DATASET_ID = "medline/2004/trec-genomics-2005"

def main():
    print(f"Loading dataset: {DATASET_ID}")
    dataset = ir_datasets.load(DATASET_ID)

    print("Reading queries...")
    q_count = 0
    for _ in dataset.queries_iter():
        q_count += 1
    print(f"Queries loaded: {q_count}")

    print("Reading qrels...")
    qr_count = 0
    for _ in dataset.qrels_iter():
        qr_count += 1
    print(f"Qrels loaded: {qr_count}")

    print("Reading all documents. This will trigger document downloads if needed...")
    doc_count = 0
    start = time.time()

    for doc in dataset.docs_iter():
        doc_count += 1

        if doc_count % 100000 == 0:
            elapsed = time.time() - start
            print(f"Read {doc_count:,} documents | elapsed: {elapsed/60:.1f} minutes")

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Finished reading documents: {doc_count:,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("Dataset is now cached locally.")
    print("=" * 60)

if __name__ == "__main__":
    main()