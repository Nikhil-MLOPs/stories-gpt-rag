from fastapi import FastAPI

app = FastAPI(
    title="Stories GPT RAG",
    version="0.1.0",
    description="RAG chatbot for story documents (.txt, .doc, .docx, .pdf, and pasted text).",
)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}
