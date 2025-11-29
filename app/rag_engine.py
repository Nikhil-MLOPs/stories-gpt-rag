from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from docx import Document
from openai import OpenAI
from pypdf import PdfReader

from app.config import settings
from app.utils import get_file_extension


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


class RAGEngine:
    """
    Core RAG engine: text extraction, chunking, embeddings, and vector store.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "stories",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Chroma client (local persistent DB)
        self._chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self._chroma_client.get_or_create_collection(
            name=collection_name
        )

        # OpenAI client (lazy init)
        self._openai_client: OpenAI | None = None

    @property
    def openai_client(self) -> OpenAI:
        """
        Lazily create OpenAI client when embeddings are actually used.
        """
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def extract_text_from_file(self, path: Path) -> str:
        """
        Extract text from .txt, .pdf, .docx.
        For legacy .doc we currently ask user to convert to .docx.
        """
        ext = get_file_extension(path.name)

        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")

        if ext == ".pdf":
            reader = PdfReader(path)
            texts: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                texts.append(page_text)
            return "\n".join(texts)

        if ext == ".docx":
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        if ext == ".doc":
            # Honest limitation for now
            raise ValueError(
                "Legacy .doc is not supported for automatic text extraction yet. "
                "Please convert the file to .docx and re-upload."
            )

        raise ValueError(f"Unsupported file extension for extraction: {ext}")

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """
        Naive word-based chunking.
        chunk_size = number of words per chunk.
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap

        if size <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0 or overlap >= size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

        words = text.split()
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = start + size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start = end - overlap

        return chunks

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for a list of texts using OpenAI.
        NOTE: Not used in tests to avoid external calls.
        """
        if not texts:
            return []

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def index_chunks(self, doc_id: str, chunks: list[str]) -> None:
        """
        Store chunks + embeddings in Chroma vector DB.
        """
        if not chunks:
            return

        embeddings = self.embed_texts(chunks)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_text: str, k: int = 4) -> list[dict[str, Any]]:
        """
        Query the vector DB and return top-k similar chunks.
        """
        if not query_text.strip():
            return []

        query_embedding = self.embed_texts([query_text])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        docs: list[dict[str, Any]] = []
        # Chroma returns batched lists: ids[0], documents[0], metadatas[0], distances[0]
        for i in range(len(result["ids"][0])):
            docs.append(
                {
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result.get("distances", [[None]])[0][i],
                }
            )
        return docs


# Global engine instance for reuse by the app
rag_engine = RAGEngine()
