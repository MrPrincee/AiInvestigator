import pytest

from app.services.hybrid_search_service import HybridSearchService


@pytest.fixture
def hybrid_service(tmp_path, monkeypatch):
    from app.services.qdrant_service import QdrantService
    from dataset.chunking import Chunk

    storage_path = str(tmp_path / "qdrant")

    class LocalQdrantService(QdrantService):
        def __init__(self, url=None):
            super().__init__(storage_path=storage_path)

    monkeypatch.setattr(
        "app.services.hybrid_search_service.QdrantService",
        LocalQdrantService,
    )

    qdrant = LocalQdrantService()

    chunks = [
        Chunk(
            id="doc-1:fixed_size:0",
            document_id="doc-1",
            content="Python is a programming language.",
            chunk_index=0,
            metadata={"title": "Python", "source": "test"},
        ),
        Chunk(
            id="doc-2:fixed_size:0",
            document_id="doc-2",
            content="Qdrant is a vector database.",
            chunk_index=0,
            metadata={"title": "Qdrant", "source": "test"},
        ),
        Chunk(
            id="doc-3:fixed_size:0",
            document_id="doc-3",
            content="BM25 is a keyword ranking algorithm.",
            chunk_index=0,
            metadata={"title": "BM25", "source": "test"},
        ),
    ]

    from app.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService()
    vectors = embedding_service.embed_chunks(chunks)

    qdrant.upsert_chunks(
        collection_name="test_hybrid",
        chunks=chunks,
        vectors=vectors,
    )
    qdrant.close()

    service = HybridSearchService(
        collection_name="test_hybrid",
    )

    yield service

    service.close()


def test_semantic_search(hybrid_service):
    results = hybrid_service.search(
        "programming language",
        mode="semantic",
        top_k=2,
    )

    assert len(results) == 2
    assert all("chunk_id" in result for result in results)


def test_keyword_search(hybrid_service):
    results = hybrid_service.search(
        "BM25 keyword",
        mode="keyword",
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0]["title"] == "BM25"


def test_hybrid_search(hybrid_service):
    results = hybrid_service.search(
        "vector database",
        mode="hybrid",
        top_k=2,
    )

    assert len(results) == 2
    assert len({result["chunk_id"] for result in results}) == 2
    assert all(result["score"] > 0 for result in results)


def test_invalid_search_parameters(hybrid_service):
    with pytest.raises(ValueError):
        hybrid_service.search("")

    with pytest.raises(ValueError):
        hybrid_service.search("Python", top_k=0)

    with pytest.raises(ValueError):
        hybrid_service.search("Python", mode="invalid")