from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.chatbot import chatbot
from app.rag_engine import rag_engine
from app.schemas import ChatRequest, ChatResponse, DocumentIngestResponse
from app.utils import (
    get_file_extension,
    is_allowed_extension,
    save_text_content,
    save_upload_file,
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger("stories-gpt-rag")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Stories GPT RAG",
    version="0.3.0",
    description="RAG chatbot for story documents (.txt, .doc, .docx, .pdf, and pasted text).",
)

# ------------------------------------------------------------------------------
# Middleware for request logging & latency
# ------------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000.0

    logger.info(
        "request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "process_time_ms": round(process_time_ms, 2),
        },
    )
    return response


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


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

        <h2>Chat</h2>
        <p><a href="/chat-ui">Go to chat page →</a></p>
    </body>
    </html>
    """


@app.post("/upload-file", response_model=DocumentIngestResponse, tags=["ingestion"])
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a single file upload, store it under data/raw,
    extract text, chunk, and index into the vector store.
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
    # Our utils generate filenames as "<uuid>_<clean_stem>.<ext>"
    doc_id = Path(saved_path).stem.split("_")[0]

    num_chunks: int | None = None
    status_str = "stored"
    message: str | None = None

    try:
        # Try to extract, chunk and index
        text = rag_engine.extract_text_from_file(saved_path)
        chunks = rag_engine.chunk_text(text)
        num_chunks = len(chunks)

        if num_chunks > 0:
            rag_engine.index_chunks(doc_id=doc_id, chunks=chunks)
            status_str = "indexed"
        else:
            status_str = "stored_no_chunks"
            message = "File saved but produced no text chunks."
    except ValueError as exc:
        # E.g. unsupported .doc for extraction
        status_str = "stored_no_index"
        message = (
            "File saved, but could not be indexed automatically: "
            f"{exc}"
        )
    except Exception as exc:  # pragma: no cover (defensive)
        status_str = "stored_no_index"
        message = (
            "File saved, but indexing failed due to an internal error: "
            f"{exc}"
        )

    return DocumentIngestResponse(
        original_filename=file.filename,
        extension=ext,
        stored_path=str(saved_path),
        source="file",
        doc_id=doc_id,
        num_chunks=num_chunks,
        status=status_str,
        message=message,
    )


@app.post("/upload-text", response_model=DocumentIngestResponse, tags=["ingestion"])
async def upload_text(
    text: str = Form(...),
    title: str | None = Form(None),
):
    """
    Accept pasted text, save as a .txt file under data/raw,
    then chunk and index it.
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty.",
        )

    saved_path = save_text_content(title=title, text=text)
    filename = (title or "pasted_text").strip() or "pasted_text"
    filename = f"{filename}.txt"
    doc_id = Path(saved_path).stem.split("_")[0]

    num_chunks: int | None = None
    status_str = "stored"
    message: str | None = None

    try:
        chunks = rag_engine.chunk_text(text)
        num_chunks = len(chunks)
        if num_chunks > 0:
            rag_engine.index_chunks(doc_id=doc_id, chunks=chunks)
            status_str = "indexed"
        else:
            status_str = "stored_no_chunks"
            message = "Text saved but produced no text chunks."
    except Exception as exc:  # pragma: no cover (defensive)
        status_str = "stored_no_index"
        message = (
            "Text saved, but indexing failed due to an internal error: "
            f"{exc}"
        )

    return DocumentIngestResponse(
        original_filename=filename,
        extension=".txt",
        stored_path=str(saved_path),
        source="pasted_text",
        doc_id=doc_id,
        num_chunks=num_chunks,
        status=status_str,
        message=message,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Ask a question about any of the ingested stories (JSON API).
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    result = chatbot.chat(query=request.query, top_k=request.top_k)
    answer: str = result["answer"]
    docs = result["documents"]

    return ChatResponse(
        answer=answer,
        top_k=request.top_k,
        num_context_documents=len(docs),
        context_documents=docs,
    )


@app.get("/chat-ui", response_class=HTMLResponse, tags=["ui"])
def chat_ui():
    """
    Simple HTML-only chat page (no JavaScript).
    Lets user type a question and shows the answer on a new page.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stories GPT RAG - Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }
            h1 { margin-bottom: 0.5rem; }
            form { border: 1px solid #ddd; padding: 1rem; margin-top: 1rem; border-radius: 8px; }
            textarea { width: 100%; margin-bottom: 0.75rem; }
            button { padding: 0.5rem 1rem; cursor: pointer; }
            a { text-decoration: none; color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Stories GPT RAG - Chat</h1>
        <p>Ask a question about any ingested story document.</p>
        <form action="/chat-ui" method="post">
            <label for="query"><strong>Your question</strong></label><br/>
            <textarea id="query" name="query" rows="4" required></textarea><br/>
            <button type="submit">Ask</button>
        </form>
        <p><a href="/">← Back to Upload</a></p>
    </body>
    </html>
    """


@app.post("/chat-ui", response_class=HTMLResponse, tags=["ui"])
async def chat_ui_post(query: str = Form(...)):
    """
    Handle chat questions from the HTML form and render the answer.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return HTMLResponse(
            """
            <html>
            <body>
                <p>Query cannot be empty.</p>
                <p><a href="/chat-ui">Go back</a></p>
            </body>
            </html>
            """,
            status_code=400,
        )

    result = chatbot.chat(query=cleaned_query, top_k=3)
    answer: str = result["answer"]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stories GPT RAG - Answer</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }}
            h1, h2 {{ margin-bottom: 0.5rem; }}
            p {{ line-height: 1.5; }}
            pre {{ white-space: pre-wrap; }}
            a {{ text-decoration: none; color: #007bff; }}
        </style>
    </head>
    <body>
        <h1>Answer</h1>
        <h2>Your Question</h2>
        <p>{cleaned_query}</p>
        <h2>Response</h2>
        <p>{answer}</p>
        <p><a href="/chat-ui">Ask another question</a></p>
        <p><a href="/">← Back to Upload</a></p>
    </body>
    </html>
    """


# ------------------------------------------------------------------------------
# Prometheus metrics endpoint (/metrics)
# ------------------------------------------------------------------------------

Instrumentator().instrument(app).expose(app)
