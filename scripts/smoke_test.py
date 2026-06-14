# Environment Check
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import sklearn
import numpy
import pandas
import ir_datasets

print("Python OK:", sys.version)
print("sklearn OK:", sklearn.__version__)
print("numpy OK:", numpy.__version__)
print("pandas OK:", pandas.__version__)
print("ir_datasets OK")

# اختبار تحميل الـDataset

from src.datasets.loader import load_dataset

dataset = load_dataset("main")

print("Dataset loaded ✔")
print("Docs:", dataset.docs_count())
print("Queries:", dataset.queries_count())
print("Qrels:", dataset.qrels_count())

# اختبار preprocessing

from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer

norm = Normalizer()
tok = Tokenizer()

text = "BRCA1 gene mutation in Breast Cancer!!!"

clean = norm.normalize(text)
tokens = tok.tokenize(clean)

print("Clean:", clean)
print("Tokens:", tokens)


# اختبار TF-IDF (mini test)

# from src.retrieval.tfidf_engine import TfidfVectorizer, TfidfRetriever

from src.retrieval.tfidf_engine import TfidfVectorizer
from src.retrieval.tfidf_retriever import TfidfRetriever

docs = [
    ["breast", "cancer", "gene"],
    ["heart", "disease"],
    ["gene", "mutation", "cancer"]
]

doc_ids = ["d1", "d2", "d3"]

tfidf = TfidfVectorizer().fit(docs, doc_ids)
retriever = TfidfRetriever(tfidf)

query = ["gene", "cancer"]

results = retriever.search(query, top_k=3)

print("TF-IDF Results:", results)

# اختبار BM25

from src.retrieval.bm25_engine import BM25Index
from src.retrieval.bm25_retriever import BM25Retriever

bm25_index = BM25Index().fit(docs, doc_ids)
bm25 = BM25Retriever(bm25_index)

results = bm25.search(query, top_k=3)

print("BM25 Results:", results)

