import pytest

from app.services.embedding_service import EmbeddingService
from dataset.chunking import Chunk


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_embedding_generation(embedding_service):
    chunks = [
        Chunk(
            id="doc-1:fixed_size:0",
            document_id="doc-1",
            content="Artificial intelligence is transforming technology.",
            chunk_index=0,
        ),
        Chunk(
            id="doc-1:fixed_size:1",
            document_id="doc-1",
            content="Machine learning is a branch of artificial intelligence.",
            chunk_index=1,
        ),
    ]

    vectors = embedding_service.embed_chunks(chunks)

    assert len(vectors) == 2
    assert embedding_service.vector_size == 384

    assert all(len(vector) == 384 for vector in vectors)
    assert all(
        isinstance(value, float)
        for vector in vectors
        for value in vector
    )


def test_empty_embeddings(embedding_service):
    assert embedding_service.embed_texts([]) == []


def test_embedding_is_deterministic(embedding_service):
    text = "Artificial intelligence and machine learning."

    first = embedding_service.embed_texts([text])[0]
    second = embedding_service.embed_texts([text])[0]

    assert first == pytest.approx(second, abs=1e-6)