import asyncio
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rag.extractor import extract_text_from_file
from src.rag import vector_store


def test_txt_extraction_uses_extension_fallback():
    upload = SimpleNamespace(
        filename="notes.txt",
        content_type="application/octet-stream",
        file=io.BytesIO(b"Plain text"),
    )
    assert extract_text_from_file(upload) == "Plain text"


def test_vector_store_upserts_duplicate_ids(monkeypatch):
    fake_collection = MagicMock()

    async def fake_embed(_texts):
        return [[0.1, 0.2]]

    monkeypatch.setattr(vector_store, "collection", fake_collection)
    monkeypatch.setattr(vector_store, "embed", fake_embed)
    asyncio.run(vector_store.add_documents(["text"], ["same-name.txt"]))
    fake_collection.upsert.assert_called_once()
