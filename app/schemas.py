from __future__ import annotations

from pydantic import BaseModel


class DocumentIngestResponse(BaseModel):
    original_filename: str
    extension: str
    stored_path: str
    source: str
    status: str = "stored"
