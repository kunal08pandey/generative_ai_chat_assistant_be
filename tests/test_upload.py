def test_upload_success(client, monkeypatch):
    captured = {}

    monkeypatch.setattr("src.api.routes.upload.extract_text_from_file", lambda file: "content")

    async def add_documents(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.api.routes.upload.add_documents", add_documents)
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["data"]["filename"] == "notes.txt"
    assert captured["ids"] == ["notes.txt"]


def test_upload_rejects_legacy_doc_files(client):
    response = client.post(
        "/upload",
        files={"file": ("notes.doc", b"content", "application/msword")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "HTTP_415"


def test_upload_preserves_extraction_error_status(client, monkeypatch):
    def fail(_file):
        raise ValueError("invalid document")

    monkeypatch.setattr("src.api.routes.upload.extract_text_from_file", fail)
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    assert response.status_code == 400
