import pytest

from dataset.chunking import (
    fixed_size_chunking,
    sentence_aware_chunking,
)
from dataset.models import Document


@pytest.fixture
def sample_document():
    return Document(
        id="doc-1",
        title="Test Document",
        content=(
            "Artificial intelligence is changing technology. "
            "Machine learning is a subset of artificial intelligence. "
            "Natural language processing helps computers understand text."
        ),
        source="test",
        metadata={"category": "technology"},
    )


def test_fixed_size_chunking():

    document = Document(
        id="doc-1",
        title="Test",
        content="A" * 1200,
        source="test",
    )

    chunks = fixed_size_chunking(
        document,
        chunk_size=500,
        overlap=50,
    )

    assert len(chunks) == 3

    assert len(chunks[0].content) == 500
    assert len(chunks[1].content) == 500
    assert len(chunks[2].content) == 300

    assert chunks[0].content[-50:] == chunks[1].content[:50]

    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata["chunking_strategy"] == "fixed_size"


def test_sentence_aware_chunking(sample_document):

    chunks = sentence_aware_chunking(
        sample_document,
        chunk_size=100,
    )

    assert len(chunks) >= 2

    assert all(len(chunk.content) <= 100 for chunk in chunks)

    assert chunks[0].content.endswith(".")

    assert chunks[0].metadata["chunking_strategy"] == "sentence_aware"

    assert chunks[0].metadata["category"] == "technology"


def test_long_sentence_is_split():

    document = Document(
        id="doc-2",
        title="Long Sentence",
        content="A" * 1200,
        source="test",
    )

    chunks = sentence_aware_chunking(
        document,
        chunk_size=500,
    )

    assert len(chunks) == 3

    assert all(len(chunk.content) <= 500 for chunk in chunks)


def test_invalid_chunking_parameters(sample_document):

    with pytest.raises(ValueError):
        fixed_size_chunking(
            sample_document,
            chunk_size=0,
        )

    with pytest.raises(ValueError):
        fixed_size_chunking(
            sample_document,
            chunk_size=500,
            overlap=500,
        )

    with pytest.raises(ValueError):
        sentence_aware_chunking(
            sample_document,
            chunk_size=0,
        )


def test_empty_document():

    document = Document(
        id="empty",
        title="Empty",
        content="",
        source="test",
    )

    assert fixed_size_chunking(document) == []
    assert sentence_aware_chunking(document) == []


def test_different_strategies_produce_different_ids(sample_document):

    fixed_chunks = fixed_size_chunking(
        sample_document,
        chunk_size=100,
        overlap=10,
    )

    sentence_chunks = sentence_aware_chunking(
        sample_document,
        chunk_size=100,
    )

    assert fixed_chunks[0].id != sentence_chunks[0].id