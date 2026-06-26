from __future__ import annotations

from src.rag.faiss_retriever import BertFaissRetriever
from src.rag.generator import GeminiGenerator


class RagPipeline:
    def __init__(
        self,
        index_dir: str | None = None,
        db_path: str | None = None,
        embedding_model_name: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model_name: str | None = None,
        gemini_temperature: float | None = None,
        max_output_tokens: int | None = None,
    ):
        self.retriever = BertFaissRetriever(
            index_dir=index_dir,
            db_path=db_path,
            model_name=embedding_model_name,
        )
        self.generator = GeminiGenerator(
            api_key=gemini_api_key,
            model_name=gemini_model_name,
            temperature=gemini_temperature,
            max_output_tokens=max_output_tokens,
        )

    def build_context(self, docs, max_docs=5, max_chars_per_doc=900):
        context_parts = []
        for i, doc in enumerate(docs[:max_docs], start=1):
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            raw_text = doc.get("raw_text", "")
            text = raw_text or abstract
            snippet = text[:max_chars_per_doc].replace("\n", " ")
            context_parts.append(
                f"[Source {i}]\n"
                f"Doc ID: {doc['doc_id']}\n"
                f"Title: {title}\n"
                f"Text: {snippet}\n"
            )
        return "\n".join(context_parts)

    def build_prompt(self, question: str, context: str):
        return f"""
You are a biomedical question-answering assistant.

Answer the user's question using ONLY the retrieved MEDLINE context below.

Rules:
- Do not use outside knowledge.
- If the retrieved context is insufficient, say that the retrieved documents do not provide enough evidence.
- Mention biomedical entities only if they appear in the context.
- Keep the answer concise and factual.
- At the end, include the source numbers used.

Retrieved MEDLINE context:
{context}

User question:
{question}

Answer:
"""

    def ask(self, question: str, retrieve_k=10, context_docs=5, max_chars_per_doc=900):
        retrieved = self.retriever.search(question, top_k=retrieve_k)
        context = self.build_context(retrieved, max_docs=context_docs, max_chars_per_doc=max_chars_per_doc)
        prompt = self.build_prompt(question, context)
        answer = self.generator.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved[:context_docs],
            "context": context,
            "prompt": prompt,
        }
