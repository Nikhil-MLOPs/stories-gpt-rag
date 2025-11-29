from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_upload_text():
    response = client.post(
        "/upload-text",
        data={"title": "test_story", "text": "Once upon a time..."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"].endswith(".txt")
    assert data["extension"] == ".txt"
    assert data["source"] == "pasted_text"
    assert "data" in data["stored_path"]  # basic sanity check


def test_upload_file_txt():
    file_content = b"Hello from a text file."
    files = {"file": ("story.txt", file_content, "text/plain")}
    response = client.post("/upload-file", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"] == "story.txt"
    assert data["extension"] == ".txt"
    assert data["source"] == "file"
    assert "data" in data["stored_path"]
