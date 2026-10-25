import pytest

from app.services.document_service import chunk_documents
from dataset.models import Document


def test_chunk_documents_fixed_size():
    document = Document(
        id="doc-1",
        title="Test Document",
        content="A" * 120,
        source="test",
    )

    chunks = chunk_documents(
        documents=[document],
        strategy="fixed_size",
        chunk_size=50,
        overlap=10,
    )

    assert len(chunks) == 3
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.metadata["chunking_strategy"] == "fixed_size" for chunk in chunks)


def test_chunk_documents_sentence_aware():
    document = Document(
        id="doc-2",
        title="Test Document",
        content="First sentence. Second sentence. Third sentence.",
        source="test",
    )

    chunks = chunk_documents(
        documents=[document],
        strategy="sentence_aware",
        chunk_size=35,
    )

    assert len(chunks) == 2
    assert chunks[0].content == "First sentence. Second sentence."
    assert chunks[1].content == "Third sentence."


def test_chunk_documents_invalid_strategy():
    document = Document(
        id="doc-3",
        title="Test Document",
        content="Some text.",
        source="test",
    )

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk_documents(
            documents=[document],
            strategy="unknown",
        )