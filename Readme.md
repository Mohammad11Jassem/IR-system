# Information Retrieval Project

## 1. Project Overview

This project implements a modular Information Retrieval system over the dataset:

```text
medline/2004/trec-genomics-2005
```

The system supports multiple retrieval models:

* TF-IDF
* BM25
* Word2Vec + FAISS
* BERT + FAISS
* Serial Hybrid Retrieval
* Parallel Hybrid Retrieval
* Query Refinement
* Evaluation using standard IR metrics
* Optional RAG service
* Gradio-based user interface


---

## 2. Dataset and Artifacts

The project uses a large biomedical IR dataset.

Dataset statistics:

```text
Dataset ID : medline/2004/trec-genomics-2005
Documents  : 3,672,808
Queries    : 50
Qrels      : 39,958
```
---

## 3. High-Level Architecture

The project pipeline is:

```text
Dataset
  -> Data Loading
  -> SQLite Document Store
  -> Preprocessing
  -> Indexing
  -> Retrieval
  -> Query Refinement
  -> Ranking / Hybrid Fusion
  -> Evaluation
  -> UI / RAG
```

Each model has its own indexing and retrieval pipeline.

---

## 4. Project Structure

```text
IR/
│
├── main.py
├── requirements.txt
├── configs/
├── notebooks/
├── scripts/
├── src/
├── frontend/
└── reports/
```

---

## 5. Root Files

### `main.py`

Main command-line entry point for searching.

It supports:

```text
--model bm25
--model tfidf
--model word2vec
--model bert
--model serial
--model parallel
```

Example:

```powershell
python main.py --model bm25 --query "BRCA1 mutation breast cancer" --top-k 5
```

Parallel hybrid example:

```powershell
python main.py `
  --model parallel `
  --query "BRCA1 mutation breast cancer" `
  --parallel-models bm25 tfidf word2vec bert `
  --per-model-k 100 `
  --top-k 5
```

---

## 6. `src/datasets`

```text
src/datasets/
    loader.py
```

This module is responsible for loading dataset components using `ir_datasets`.

It loads:

```text
documents
queries
qrels
```
---

## 7. `src/storage`

```text
src/storage/
    document_store.py
```

This layer handles access to the local SQLite database:

The SQLite database stores the original document content, including:

```text
doc_id
title
abstract
contents
```

Retrieval models usually return only `doc_id`. The system then uses SQLite to fetch the document title and abstract for display.

---

## 8. `src/preprocessing`

```text
src/preprocessing/
    document_processor.py
    text_cleaner.py
    query_processor.py
    normalizer.py
    tokenizer.py
    __init__.py
```

This package contains text preprocessing utilities.

### 8.1 `DocumentProcessor`

Used for lexical retrieval models:

```text
BM25
TF-IDF
Terrier index
```

The pipeline is:

```text
raw text
  -> Unicode normalization
  -> lowercasing
  -> tokenization
  -> stopword removal
  -> whitespace normalization
  -> processed text
```
### 8.2 `EmbeddingTextCleaner`

Used for embedding-based models:

```text
BERT
Word2Vec
```

It does not apply aggressive stemming or lemmatization because BERT and Word2Vec benefit from preserving semantic structure.

---

## 9. `src/indexing`

```text
src/indexing/
    terrier_index.py
    bert_index.py
    word2vec_index.py
    __init__.py
```

This package contains the actual indexing logic.

The scripts in `scripts/` only call these indexing services.

### 9.1 `terrier_index.py`

Builds the Terrier inverted index used by:

```text
BM25
TF-IDF
```

### 9.2 `bert_index.py`

Builds the BERT FAISS vector index.


Main output files:

```text
index.faiss
doc_ids.pkl
doc_ids.jsonl
metadata.json
```

### 9.3 `word2vec_index.py`

Builds the Word2Vec FAISS vector index.

Main output files:

```text
index.faiss
doc_ids.pkl
doc_ids.jsonl
word2vec.model
word2vec.kv
metadata.json
```

---

## 10. `src/retrieval`

```text
src/retrieval/
    terrier_retriever.py
    bm25_retriever.py
    tfidf_retriever.py
    bert_retriever.py
    word2vec_retriever.py
    serial_hybrid_retriever.py
    parallel_hybrid_retriever.py
    __init__.py
```

This package contains all search/retrieval logic.

---

## 11. Hybrid Retrieval

### 11.1 Serial Hybrid Retrieval

```text
serial_hybrid_retriever.py
```

Serial hybrid works in two stages:

```text
Query
  -> first-stage retriever gets candidate documents
  -> second-stage retriever reranks candidates
  -> final top_k documents
```

Typical configuration:

```text
BM25 -> BERT
BM25 -> Word2Vec
TF-IDF -> BERT
```

---

### 11.2 Parallel Hybrid Retrieval

```text
parallel_hybrid_retriever.py
```

Parallel hybrid sends the same query to multiple retrievers independently and then fuses their ranked lists.

Fusion method:

```text
RRF = Reciprocal Rank Fusion
score(doc) = sum(1 / (rrf_k + rank))
```

---

## 12. `src/query_refinement`

```text
src/query_refinement/
    query_refinement_service.py
    prf_refiner.py
    word2vec_expander.py
    history_refiner.py
    token_utils.py
    models.py
    __init__.py
```

This package implements query refinement.

---

## 13. `src/evaluation`

```text
src/evaluation/
    metrics.py
    evaluator.py
    report_writer.py
    __init__.py
```

This package evaluates retrievers using official dataset queries and qrels.

Metrics:

```text
Precision@10
Recall@K
MAP@K
nDCG@10
Average query time
```
---

## 14. `src/rag`

```text
src/rag/
    config.py
    document_repository.py
    faiss_retriever.py
    generator.py
    rag_pipeline.py
    __init__.py
```

This package implements an optional Retrieval-Augmented Generation service.
---

## 15. `scripts`

```text
scripts/
    build_document_store.py
    build_terrier_index.py
    build_bert_index.py
    build_word2vec_index.py
    evaluate_retrievers.py
    search_terrier.py
    search_bert.py
    search_word2vec.py
    refine_query.py
    smoke_terrier_index.py
    rag_chat_cli.py
    rag_check_friend_artifacts.py
```

Scripts are command-line wrappers.

They should not contain the main business logic. The main logic should be inside `src`.

---

## 16. `frontend`

```text
frontend/
    app.py
    controllers/
    ui/
    assets/
```

The frontend is implemented using Gradio.

---

## 17. Summary

The project is organized into clear layers:

```text
datasets       -> load official dataset data
storage        -> SQLite document access
preprocessing  -> text processing for lexical and embedding models
indexing       -> build Terrier and FAISS indexes
retrieval      -> search using individual and hybrid models
query_refinement -> improve user queries using PRF and expansion
evaluation     -> compute IR metrics using qrels
rag            -> optional retrieval-augmented generation
frontend       -> Gradio user interface
```