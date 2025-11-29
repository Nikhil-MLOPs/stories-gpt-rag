from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from app.schemas import DocumentIngestResponse
from app.utils import (
    get_file_extension,
    is_allowed_extension,
    save_text_content,
    save_upload_file,
)


app = FastAPI(
    title="Stories GPT RAG",
    version="0.1.0",
    description="RAG chatbot for story documents (.txt, .doc, .docx, .pdf, and pasted text).",
)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index():
    """
    Simple HTML UI for uploading files or pasting text.
    No JavaScript. Just plain HTML forms.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stories GPT RAG - Upload</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }
            h1 { margin-bottom: 0.5rem; }
            form { border: 1px solid #ddd; padding: 1rem; margin-bottom: 1.5rem; border-radius: 8px; }
            label { display: block; margin-bottom: 0.5rem; font-weight: bold; }
            input[type="file"],
            input[type="text"],
            textarea { width: 100%; margin-bottom: 0.75rem; }
            button { padding: 0.5rem 1rem; cursor: pointer; }
            small { color: #555; }
        </style>
    </head>
    <body>
        <h1>Stories GPT RAG - Document Intake</h1>
        <p>Upload a story file or paste your story text. Supported: .txt, .pdf, .doc, .docx.</p>

        <h2>Upload File</h2>
        <form action="/upload-file" method="post" enctype="multipart/form-data">
            <label for="file">Choose a file</label>
            <input type="file" id="file" name="file" required />
            <small>Allowed types: .txt, .pdf, .doc, .docx</small><br />
            <button type="submit">Upload File</button>
        </form>

        <h2>Paste Text</h2>
        <form action="/upload-text" method="post">
            <label for="title">Optional title</label>
            <input type="text" id="title" name="title" placeholder="My Story Title" />
            <label for="text">Story text</label>
            <textarea id="text" name="text" rows="8" required></textarea>
            <button type="submit">Submit Text</button>
        </form>
    </body>
    </html>
    """


@app.post("/upload-file", response_model=DocumentIngestResponse, tags=["ingestion"])
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a single file upload and store it under data/raw.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided.",
        )

    if not is_allowed_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: .txt, .pdf, .doc, .docx",
        )

    saved_path = await save_upload_file(file)
    ext = get_file_extension(file.filename)

    return DocumentIngestResponse(
        original_filename=file.filename,
        extension=ext,
        stored_path=str(saved_path),
        source="file",
    )


@app.post("/upload-text", response_model=DocumentIngestResponse, tags=["ingestion"])
async def upload_text(
    text: str = Form(...),
    title: str | None = Form(None),
):
    """
    Accept pasted text, save as a .txt file under data/raw.
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty.",
        )

    saved_path = save_text_content(title=title, text=text)
    filename = (title or "pasted_text").strip() or "pasted_text"
    filename = f"{filename}.txt"

    return DocumentIngestResponse(
        original_filename=filename,
        extension=".txt",
        stored_path=str(saved_path),
        source="pasted_text",
    )
