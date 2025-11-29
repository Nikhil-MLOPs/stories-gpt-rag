from pathlib import Path

from app.rag_engine import rag_engine


def test_chunk_text_basic():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = rag_engine.chunk_text(text, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_extract_text_from_txt(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    content = "Hello sample story."
    file_path.write_text(content, encoding="utf-8")

    extracted = rag_engine.extract_text_from_file(file_path)
    assert "Hello sample story." in extracted
