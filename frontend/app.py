from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from main import (
    DEFAULT_BERT_INDEX_DIR,
    DEFAULT_BERT_MODEL_NAME,
    DEFAULT_DB_PATH,
    DEFAULT_TERRIER_INDEX_PATH,
    DEFAULT_WORD2VEC_INDEX_DIR,
)
from frontend.controllers.artifact_checker import check_artifacts
from frontend.controllers.evaluation_controller import build_evaluation_command, command_preview, run_evaluation
from frontend.controllers.rag_controller import run_rag
from frontend.controllers.search_controller import build_search_config, execute_search
from frontend.controllers.dataset_controller import (
    DEFAULT_DATASET_ID,
    official_query_choices,
    query_text_from_label,
)
from frontend.ui.renderers import render_json, render_refinement, render_results

CSS_PATH = Path(__file__).resolve().parent / "assets" / "theme.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def _error_html(title: str, exc: Exception) -> str:
    tb = traceback.format_exc()
    return f"""
    <div class='summary-card'>
      <div class='summary-title'>❌ {title}</div>
      <pre style='white-space:pre-wrap; color:#b91c1c'>{exc}\n\n{tb}</pre>
    </div>
    """



def on_load_official_queries(dataset_id: str):
    try:
        dataset_id = (dataset_id or DEFAULT_DATASET_ID).strip()
        choices = official_query_choices(dataset_id)
        if not choices:
            return gr.update(choices=[], value=None), "<div class='empty-state'>No official queries were found.</div>"
        return (
            gr.update(choices=choices, value=choices[0]),
            f"<div class='summary-card'><b>Loaded {len(choices)} official queries</b><br/>Dataset: <code>{dataset_id}</code></div>",
        )
    except Exception as exc:
        return gr.update(choices=[], value=None), _error_html("Failed to load official queries", exc)


def on_select_official_query(label: str, dataset_id: str):
    try:
        dataset_id = (dataset_id or DEFAULT_DATASET_ID).strip()
        text, qid = query_text_from_label(label, dataset_id)
        if not text or not qid:
            return "", "", "<div class='empty-state'>Select an official query first.</div>"
        return (
            text,
            qid,
            f"<div class='summary-card'><b>Selected official query</b><br/>qid: <code>{qid}</code><br/>Qrels judgment will be applied after search.</div>",
        )
    except Exception as exc:
        return "", "", _error_html("Failed to select official query", exc)

def on_search(*args):
    try:
        config = build_search_config(*args)
        output = execute_search(config)
        return render_results(output), render_refinement(output), render_json(output)
    except Exception as exc:
        err = _error_html("Search failed", exc)
        return err, err, str(exc)


def on_check_artifacts(index_path, db_path, bert_index_dir, word2vec_index_dir):
    try:
        return check_artifacts(index_path, db_path, bert_index_dir, word2vec_index_dir)
    except Exception as exc:
        return _error_html("Artifact check failed", exc)


def on_preview_eval(*args):
    try:
        cmd = build_evaluation_command(*args)
        return command_preview(cmd)
    except Exception as exc:
        return f"ERROR: {exc}"


def on_run_eval(*args):
    try:
        return run_evaluation(*args)
    except Exception as exc:
        return "", f"ERROR: {exc}\n{traceback.format_exc()}"


def on_rag(*args):
    try:
        return run_rag(*args)
    except Exception as exc:
        err_text = f"RAG failed: {exc}\n\n{traceback.format_exc()}"
        return err_text, "", _error_html("RAG failed", exc), err_text


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="IR Search & RAG Assistant") as demo:
        gr.HTML(
            """
            <div id='app-title'>
              <h1>IR Search & RAG Assistant</h1>
              <p>Control panel for BM25, TF-IDF, Word2Vec, BERT, Hybrid Retrieval, Query Refinement, Evaluation, and RAG-ready retrieval.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=380):
                gr.Markdown("### Global Paths")
                index_path = gr.Textbox(label="Terrier Index Path", value=DEFAULT_TERRIER_INDEX_PATH)
                db_path = gr.Textbox(label="SQLite Document Store", value=DEFAULT_DB_PATH)
                bert_index_dir = gr.Textbox(label="BERT FAISS Index Dir", value=DEFAULT_BERT_INDEX_DIR)
                bert_model_name = gr.Textbox(label="BERT / SentenceTransformer Model", value=DEFAULT_BERT_MODEL_NAME)
                word2vec_index_dir = gr.Textbox(label="Word2Vec Index Dir", value=DEFAULT_WORD2VEC_INDEX_DIR)
                check_btn = gr.Button("Check Artifacts", variant="secondary")
                artifact_status = gr.HTML()

                check_btn.click(
                    on_check_artifacts,
                    inputs=[index_path, db_path, bert_index_dir, word2vec_index_dir],
                    outputs=[artifact_status],
                )

                gr.Markdown(
                    """
                    ### Recommended Final Setup
                    - Query Refinement: `prf`
                    - PRF feedback docs: `10`
                    - Max PRF terms: `3`
                    - Original term weight: `2`
                    """
                )

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Search"):
                        query_mode = gr.Radio(
                            label="Query Mode",
                            choices=["Manual Query", "Official Dataset Query"],
                            value="Manual Query",
                            info="Use Official Dataset Query when you want qrels-based relevance judgment.",
                        )
                        official_qid_state = gr.State(value="")
                        with gr.Accordion("Official Dataset Query / Qrels Judgment", open=False):
                            official_dataset_id = gr.Textbox(
                                label="Dataset ID",
                                value=DEFAULT_DATASET_ID,
                                info="Official queries and qrels are loaded from this ir_datasets dataset.",
                            )
                            with gr.Row():
                                load_official_queries_btn = gr.Button("Load Official Queries", variant="secondary")
                                official_query_dropdown = gr.Dropdown(
                                    label="Official Query",
                                    choices=[],
                                    value=None,
                                    interactive=True,
                                )
                            official_query_status = gr.HTML(
                                "<div class='empty-state'>Click Load Official Queries, then select a qid.</div>"
                            )

                        with gr.Row():
                            query = gr.Textbox(
                                label="Query",
                                value="BRCA1 mutation breast cancer",
                                lines=3,
                                scale=4,
                            )
                            with gr.Column(scale=1):
                                model = gr.Dropdown(
                                    label="Model",
                                    choices=["BM25", "TF-IDF", "Word2Vec", "BERT", "Serial Hybrid", "Parallel Hybrid"],
                                    value="BM25",
                                )
                                top_k = gr.Slider(label="Top K", minimum=1, maximum=100, value=5, step=1)
                                search_btn = gr.Button("Search", variant="primary")

                        with gr.Accordion("BM25 Parameters", open=True):
                            gr.Markdown(
                                "These controls affect BM25 scoring at query time only. "
                                "Changing them does not require rebuilding the Terrier index."
                            )
                            with gr.Row():
                                bm25_k1 = gr.Slider(
                                    label="BM25 k1",
                                    minimum=0.1,
                                    maximum=3.0,
                                    value=1.2,
                                    step=0.1,
                                    info="Controls term-frequency saturation. Higher values give repeated terms more influence.",
                                )
                                bm25_b = gr.Slider(
                                    label="BM25 b",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.75,
                                    step=0.05,
                                    info="Controls document-length normalization. 0 disables it; 1 applies strong normalization.",
                                )

                        with gr.Accordion("Hybrid Settings", open=False):
                            with gr.Row():
                                first_stage = gr.Dropdown(
                                    label="Serial First Stage",
                                    choices=["bm25", "tfidf", "word2vec"],
                                    value="bm25",
                                )
                                second_stage = gr.Dropdown(
                                    label="Serial Second Stage",
                                    choices=["bert", "word2vec"],
                                    value="bert",
                                )
                                candidate_k = gr.Slider(label="Candidate K", minimum=10, maximum=2000, value=100, step=10)
                            parallel_models = gr.CheckboxGroup(
                                label="Parallel Models",
                                choices=["BM25", "TF-IDF", "Word2Vec", "BERT"],
                                value=["BM25", "TF-IDF", "Word2Vec"],
                            )
                            per_model_k = gr.Slider(label="Per Model K", minimum=10, maximum=2000, value=100, step=10)

                        with gr.Accordion("Query Refinement Settings", open=True):
                            with gr.Row():
                                enable_refinement = gr.Checkbox(label="Enable Query Refinement", value=False)
                                refinement_method = gr.Dropdown(
                                    label="Refinement Method",
                                    choices=["prf", "context_word2vec", "prf_word2vec", "word2vec", "history", "history_word2vec"],
                                    value="prf",
                                )
                                query_history_file = gr.Textbox(label="Query History File", value="")
                            with gr.Row():
                                prf_feedback_docs = gr.Slider(label="PRF Feedback Docs", minimum=1, maximum=50, value=10, step=1)
                                max_prf_terms = gr.Slider(label="Max PRF Terms", minimum=0, maximum=20, value=3, step=1)
                                original_term_weight = gr.Slider(label="Original Term Weight", minimum=1, maximum=5, value=2, step=1)
                            with gr.Row():
                                max_expansion_terms = gr.Slider(label="Max W2V Expansion Terms", minimum=0, maximum=20, value=2, step=1)
                                word2vec_topn = gr.Slider(label="W2V TopN", minimum=1, maximum=30, value=8, step=1)
                                min_word2vec_similarity = gr.Slider(label="Min W2V Similarity", minimum=0.0, maximum=1.0, value=0.60, step=0.01)

                        results_html = gr.HTML(label="Results")
                        refinement_html = gr.HTML(label="Refinement Details")
                        raw_json = gr.Code(label="Raw Output JSON", language="json")

                        search_inputs = [
                            query,
                            model,
                            top_k,
                            bm25_k1,
                            bm25_b,
                            index_path,
                            db_path,
                            bert_index_dir,
                            bert_model_name,
                            word2vec_index_dir,
                            first_stage,
                            second_stage,
                            candidate_k,
                            parallel_models,
                            per_model_k,
                            enable_refinement,
                            refinement_method,
                            prf_feedback_docs,
                            max_prf_terms,
                            max_expansion_terms,
                            word2vec_topn,
                            min_word2vec_similarity,
                            original_term_weight,
                            query_history_file,
                            query_mode,
                            official_dataset_id,
                            official_qid_state,
                        ]
                        load_official_queries_btn.click(
                            on_load_official_queries,
                            inputs=[official_dataset_id],
                            outputs=[official_query_dropdown, official_query_status],
                        )
                        official_query_dropdown.change(
                            on_select_official_query,
                            inputs=[official_query_dropdown, official_dataset_id],
                            outputs=[query, official_qid_state, official_query_status],
                        )
                        search_btn.click(on_search, inputs=search_inputs, outputs=[results_html, refinement_html, raw_json])

                    with gr.Tab("Evaluation"):
                        gr.Markdown("Run evaluation from the UI. For full BERT or serial BERT evaluations, expect long runtime.")
                        eval_models = gr.CheckboxGroup(
                            label="Models",
                            choices=[
                                "BM25",
                                "TF-IDF",
                                "Word2Vec",
                                "BERT",
                                "Parallel Basic",
                                "Parallel Full",
                                "Serial BM25 → Word2Vec",
                                "Serial BM25 → BERT",
                            ],
                            value=["BM25"],
                        )
                        with gr.Row():
                            limit_queries = gr.Number(label="Limit Queries (0 = all)", value=0, precision=0)
                            run_depth = gr.Number(label="Run Depth", value=1000, precision=0)
                            eval_bm25_k1 = gr.Number(label="BM25 k1", value=1.2)
                            eval_bm25_b = gr.Number(label="BM25 b", value=0.75)
                            precision_k = gr.Number(label="Precision@K", value=10, precision=0)
                            recall_k = gr.Number(label="Recall@K", value=1000, precision=0)
                            ndcg_k = gr.Number(label="nDCG@K", value=10, precision=0)
                            map_depth = gr.Number(label="MAP Depth", value=1000, precision=0)
                        eval_output_dir = gr.Textbox(label="Output Dir", value="reports/evaluation/ui_run")
                        with gr.Accordion("Evaluation Query Refinement", open=False):
                            eval_enable_refinement = gr.Checkbox(label="Enable Refinement", value=False)
                            eval_refinement_method = gr.Dropdown(
                                label="Method",
                                choices=["prf", "context_word2vec", "prf_word2vec"],
                                value="prf",
                            )
                            with gr.Row():
                                eval_prf_feedback_docs = gr.Number(label="PRF Feedback Docs", value=10, precision=0)
                                eval_max_prf_terms = gr.Number(label="Max PRF Terms", value=3, precision=0)
                                eval_original_term_weight = gr.Number(label="Original Term Weight", value=2, precision=0)
                            with gr.Row():
                                eval_max_expansion_terms = gr.Number(label="Max W2V Terms", value=2, precision=0)
                                eval_word2vec_topn = gr.Number(label="W2V TopN", value=8, precision=0)
                                eval_min_word2vec_similarity = gr.Number(label="Min W2V Similarity", value=0.60)

                        eval_inputs = [
                            eval_models,
                            limit_queries,
                            run_depth,
                            eval_bm25_k1,
                            eval_bm25_b,
                            precision_k,
                            recall_k,
                            ndcg_k,
                            map_depth,
                            eval_output_dir,
                            index_path,
                            db_path,
                            bert_index_dir,
                            bert_model_name,
                            word2vec_index_dir,
                            eval_enable_refinement,
                            eval_refinement_method,
                            eval_prf_feedback_docs,
                            eval_max_prf_terms,
                            eval_max_expansion_terms,
                            eval_word2vec_topn,
                            eval_min_word2vec_similarity,
                            eval_original_term_weight,
                        ]
                        with gr.Row():
                            preview_btn = gr.Button("Preview Command", variant="secondary")
                            run_eval_btn = gr.Button("Run Evaluation", variant="primary")
                        eval_command = gr.Code(label="Command", language="shell")
                        eval_log = gr.Textbox(label="Evaluation Log", lines=24)
                        preview_btn.click(on_preview_eval, inputs=eval_inputs, outputs=[eval_command])
                        run_eval_btn.click(on_run_eval, inputs=eval_inputs, outputs=[eval_command, eval_log])

                    with gr.Tab("RAG Ready"):
                        gr.Markdown(
                            """
                            This tab runs a complete RAG workflow: retrieve MEDLINE source documents, build context, build a grounded prompt, and optionally generate an answer using Gemini or a local HuggingFace model.
                            """
                        )
                        rag_question = gr.Textbox(
                            label="Question",
                            value="What is the role of BRCA1 mutations in breast cancer?",
                            lines=3,
                        )
                        with gr.Row():
                            rag_retriever = gr.Dropdown(
                                label="Retriever",
                                choices=["BERT", "BM25", "TF-IDF", "Parallel Hybrid"],
                                value="BERT",
                            )
                            retrieve_k = gr.Slider(label="Retrieve K", minimum=1, maximum=100, value=10, step=1)
                            context_docs = gr.Slider(label="Context Docs", minimum=1, maximum=10, value=5, step=1)
                            max_context_chars = gr.Slider(label="Max Context Chars", minimum=1000, maximum=30000, value=6000, step=500)

                        with gr.Accordion("RAG BM25 Parameters", open=False):
                            with gr.Row():
                                rag_bm25_k1 = gr.Slider(label="RAG BM25 k1", minimum=0.1, maximum=3.0, value=1.2, step=0.1)
                                rag_bm25_b = gr.Slider(label="RAG BM25 b", minimum=0.0, maximum=1.0, value=0.75, step=0.05)

                        with gr.Accordion("RAG Query Refinement", open=False):
                            with gr.Row():
                                rag_enable_refinement = gr.Checkbox(label="Enable Query Refinement", value=False)
                                rag_refinement_method = gr.Dropdown(
                                    label="Refinement Method",
                                    choices=["prf", "prf_word2vec"],
                                    value="prf",
                                )

                        with gr.Accordion("Answer Generation", open=True):
                            answer_mode = gr.Dropdown(
                                label="Answer Mode",
                                choices=["Build prompt only", "Gemini API", "Local HuggingFace"],
                                value="Build prompt only",
                                info="Start with Build prompt only to verify retrieval and context before calling an LLM.",
                            )
                            with gr.Row():
                                hf_model_name = gr.Textbox(label="Local HF Model", value="google/flan-t5-base")
                                gemini_model_name = gr.Textbox(label="Gemini Model", value="gemini-1.5-flash")
                            with gr.Row():
                                gemini_api_key = gr.Textbox(
                                    label="Gemini API Key",
                                    value="",
                                    type="password",
                                    placeholder="Optional: paste key here or set GEMINI_API_KEY",
                                )
                                gemini_temperature = gr.Slider(label="Gemini Temperature", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
                                max_new_tokens = gr.Slider(label="Max Output Tokens", minimum=32, maximum=2048, value=256, step=32)

                        rag_btn = gr.Button("Retrieve / Generate", variant="primary")
                        rag_answer = gr.Textbox(label="Answer or Prompt", lines=14)
                        rag_context = gr.Textbox(label="Retrieved Context", lines=14)
                        rag_sources = gr.HTML(label="Sources")
                        rag_raw_json = gr.Code(label="Raw RAG Output JSON", language="json")

                        rag_btn.click(
                            on_rag,
                            inputs=[
                                rag_question,
                                rag_retriever,
                                retrieve_k,
                                rag_bm25_k1,
                                rag_bm25_b,
                                context_docs,
                                max_context_chars,
                                answer_mode,
                                hf_model_name,
                                gemini_model_name,
                                gemini_api_key,
                                gemini_temperature,
                                max_new_tokens,
                                index_path,
                                db_path,
                                bert_index_dir,
                                bert_model_name,
                                word2vec_index_dir,
                                rag_enable_refinement,
                                rag_refinement_method,
                                prf_feedback_docs,
                                max_prf_terms,
                                max_expansion_terms,
                                word2vec_topn,
                                min_word2vec_similarity,
                                original_term_weight,
                            ],
                            outputs=[rag_answer, rag_context, rag_sources, rag_raw_json],
                        )

                    with gr.Tab("Help"):
                        gr.Markdown(
                            """
                            ## What this UI controls

                            - Standalone retrieval: BM25, TF-IDF, Word2Vec, BERT.
                            - Hybrid retrieval: Serial and Parallel RRF fusion.
                            - Query Refinement: PRF, context-aware Word2Vec, and combined PRF + Word2Vec.
                            - Official dataset queries: select a qid from `ir_datasets` and show qrels judgments per result.
                            - Evaluation: run the existing evaluation script from the UI.
                            - RAG-ready workflow: retrieve context documents and optionally call a local LLM.

                            ## Recommended Query Refinement

                            Based on the completed evaluation, use:

                            ```text
                            method = prf
                            feedback_docs = 10
                            max_prf_terms = 3
                            original_term_weight = 2
                            ```

                            ## Run

                            ```powershell
                            python frontend/app.py
                            ```
                            """
                        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
