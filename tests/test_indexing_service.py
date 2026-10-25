import json

import pytest

from app.services.indexing_service import index_documents
from app.services.qdrant_service import QdrantService


@pytest.fixture
def sample_dataset(tmp_path):
    file_path = tmp_path / "documents.jsonl"

    documents = [
        {
            "id": f"doc-{index}",
            "title": f"Document {index}",
            "content": f"Machine learning document number {index}.",
            "source": "test",
            "metadata": {},
        }
        for index in range(5)
    ]

    with file_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")

    return file_path


def test_index_documents_limit(sample_dataset, monkeypatch, tmp_path):
    from app.services import indexing_service

    storage_path = tmp_path / "qdrant"

    monkeypatch.setattr(
        indexing_service,
        "QdrantService",
        lambda url: QdrantService(storage_path=str(storage_path)),
    )

    result = index_documents(
        file_path=sample_dataset,
        collection_name="test_indexing",
        batch_size=2,
        max_documents=3,
    )

    assert result["indexed_documents"] == 3
    assert result["indexed_chunks"] == 3

    client = QdrantService(storage_path=str(storage_path))

    try:
        assert client.client.get_collection("test_indexing").points_count == 3
    finally:
        client.close()


def test_index_documents_idempotency(sample_dataset, monkeypatch, tmp_path):
    from app.services import indexing_service

    storage_path = tmp_path / "qdrant"

    monkeypatch.setattr(
        indexing_service,
        "QdrantService",
        lambda url: QdrantService(storage_path=str(storage_path)),
    )

    for _ in range(2):
        result = index_documents(
            file_path=sample_dataset,
            collection_name="test_indexing",
            batch_size=2,
            max_documents=5,
        )

        assert result["indexed_documents"] == 5

    client = QdrantService(storage_path=str(storage_path))

    try:
        assert client.client.get_collection("test_indexing").points_count == 5
    finally:
        client.close()