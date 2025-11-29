from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentIngestResponse(BaseModel):
    original_filename: str
    extension: str
    stored_path: str
    source: str
    doc_id: str
    num_chunks: Optional[int] = None
    status: str = "stored"
    message: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    top_k: int
    num_context_documents: int
    context_documents: List[Dict[str, Any]]
