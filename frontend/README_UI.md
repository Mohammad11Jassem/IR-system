# IR Search & RAG Assistant UI

This Gradio UI is a control panel for the IR project. It supports:

- BM25, TF-IDF, Word2Vec, and BERT retrieval.
- Serial Hybrid and Parallel Hybrid retrieval.
- Query Refinement using PRF, Context-Aware Word2Vec, and PRF + Word2Vec.
- Evaluation script execution from the UI.
- RAG-ready retrieval: retrieved context + optional LLM generation.

## Folder Structure

```text
frontend/
    app.py
    README_UI.md
    assets/
        theme.css
    controllers/
        artifact_checker.py
        evaluation_controller.py
        rag_controller.py
        search_controller.py
    ui/
        renderers.py
```

## Install Gradio

```powershell
python -m pip install gradio
```

If you want optional RAG generation using a HuggingFace model:

```powershell
python -m pip install transformers sentencepiece accelerate
```

## Run

From the project root:

```powershell
python frontend/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Recommended Refinement Settings

Use these settings for the final demo:

```text
Refinement Method: prf
PRF Feedback Docs: 10
Max PRF Terms: 3
Original Term Weight: 2
```

## Important Notes

- This UI does not duplicate retrieval logic. It imports the official project retrievers through `main.py` and `src/`.
- Large artifacts are not included in GitHub. Set the paths from the left sidebar.
- For BERT and Word2Vec, the FAISS index folders must include compatible `index.faiss` and `doc_ids.pkl` files.
- The RAG tab is prepared for integration. It can retrieve context immediately; LLM generation is optional.

## Official Query Mode and Qrels Judgment

The Search tab supports two modes:

1. **Manual Query**: type any query manually. This is useful for demos and qualitative inspection, but results cannot be judged against qrels unless the query corresponds to an official qid.
2. **Official Dataset Query**: load the official `ir_datasets` queries, select a `qid`, and run retrieval. The UI will attach qrels judgments to every returned result:
   - `Relevant | grade=1/2`
   - `Non-relevant | grade=0`
   - `Unjudged in qrels`

Default dataset:

```text
medline/2004/trec-genomics-2005
```

Workflow:

1. Select **Official Dataset Query**.
2. Open **Official Dataset Query / Qrels Judgment**.
3. Click **Load Official Queries**.
4. Select a qid from the dropdown.
5. Run Search.
6. Read the qrels summary and relevance badges on each result card.

This feature is for result-level verification. Full official evaluation is still performed through the Evaluation tab or `scripts/evaluate_retrievers.py`.
