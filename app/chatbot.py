from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from app.config import settings
from app.rag_engine import rag_engine


class Chatbot:
    """
    High-level chatbot over the RAG engine.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")
        if self._client is None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def build_context(
        self,
        query: str,
        top_k: int = 4,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Query the vector DB and build a context string from the retrieved chunks.
        """
        results = rag_engine.query(query_text=query, k=top_k)
        if not results:
            return "", []

        chunks_text = "\n\n---\n\n".join(r["document"] for r in results)
        return chunks_text, results

    def chat(
        self,
        query: str,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Run a full RAG-style chat: retrieve, then call the LLM.
        """
        context, docs = self.build_context(query=query, top_k=top_k)

        if not context:
            system_prompt = (
                "You are a helpful assistant. The user has asked about their "
                "stories, but no relevant context is available. Answer in a "
                "generic way and say that you don't see their stories yet."
            )
        else:
            system_prompt = (
                "You are a helpful assistant that answers questions about user "
                "stories. Use ONLY the context from the stories given to you. "
                "If the answer is not clearly in the context, say you are not "
                "sure and explain what information is missing."
            )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Here is the context from the stories:\n\n{context}",
                }
            )

        messages.append({"role": "user", "content": query})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = completion.choices[0].message.content or ""

        return {"answer": answer, "documents": docs}


chatbot = Chatbot()
