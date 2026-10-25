import json

import pytest

from app.services.document_service import read_document_batches


def test_read_document_batches(tmp_path):
    file_path = tmp_path / "documents.jsonl"

    documents = [
        {
            "id": f"doc-{index}",
            "title": f"Document {index}",
            "content": f"Content {index}",
            "source": "test",
            "metadata": {},
        }
        for index in range(5)
    ]

    with file_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")

    batches = list(
        read_document_batches(
            file_path=file_path,
            batch_size=2,
        )
    )

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0].id == "doc-0"
    assert batches[-1][0].id == "doc-4"


def test_read_document_batches_invalid_size(tmp_path):
    with pytest.raises(ValueError):
        list(
            read_document_batches(
                file_path=tmp_path / "documents.jsonl",
                batch_size=0,
            )
        )