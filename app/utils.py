from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import UploadFile


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx"}


def ensure_data_dirs() -> None:
    """Ensure that raw and processed data directories exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_data_dirs()


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_extension(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """
    Make filename safe for saving on disk (remove weird characters,
    keep extension).
    """
    name = Path(filename).name
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove anything that is not alphanumeric, dot, underscore or dash
    return re.sub(r"[^A-Za-z0-9._-]", "", name)


def build_storage_path(original_filename: str) -> Path:
    """
    Build a unique path for storing a file in RAW_DATA_DIR.
    """
    ext = get_file_extension(original_filename) or ".txt"
    clean_stem = sanitize_filename(Path(original_filename).stem) or "document"
    doc_id = uuid.uuid4().hex
    new_filename = f"{doc_id}_{clean_stem}{ext}"
    return RAW_DATA_DIR / new_filename


async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Save an uploaded file to disk and return the saved path.
    """
    destination = build_storage_path(upload_file.filename)
    with destination.open("wb") as f:
        while True:
            chunk = await upload_file.read(8192)
            if not chunk:
                break
            f.write(chunk)
    await upload_file.close()
    return destination


def save_text_content(title: str | None, text: str) -> Path:
    """
    Save pasted text as a .txt file and return the saved path.
    """
    base_name = (title or "pasted_text").strip() or "pasted_text"
    filename = f"{base_name}.txt"
    destination = build_storage_path(filename)
    destination.write_text(text, encoding="utf-8")
    return destination
