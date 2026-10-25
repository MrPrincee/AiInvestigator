from app.services.qdrant_service import QdrantService
import pytest

from dataset.chunking import Chunk



def test_collection_creation(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        service.ensure_collection("test_documents")

        assert service.client.collection_exists(
            "test_documents"
        )

        collection = service.client.get_collection(
            "test_documents"
        )

        assert collection.config.params.vectors.size == 384

        service.ensure_collection("test_documents")

        assert service.client.collection_exists(
            "test_documents"
        )

    finally:
        service.close()


def test_collection_persists_after_restart(tmp_path):
    storage_path = str(tmp_path / "qdrant")

    first_service = QdrantService(storage_path)
    first_service.ensure_collection("persistent_documents")
    first_service.close()

    second_service = QdrantService(storage_path)

    try:
        assert second_service.client.collection_exists(
            "persistent_documents"
        )
    finally:
        second_service.close()

def test_upsert_chunks(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        chunks = [
            Chunk(
                id="doc-1:fixed_size:0",
                document_id="doc-1",
                content="Artificial intelligence is transforming technology.",
                chunk_index=0,
                metadata={
                    "title": "AI Document",
                    "source": "test.pdf",
                    "chunking_strategy": "fixed_size",
                },
            ),
            Chunk(
                id="doc-1:fixed_size:1",
                document_id="doc-1",
                content="Machine learning is a branch of AI.",
                chunk_index=1,
                metadata={
                    "title": "AI Document",
                    "source": "test.pdf",
                    "chunking_strategy": "fixed_size",
                },
            ),
        ]

        vectors = [
            [0.1] * 384,
            [0.2] * 384,
        ]

        inserted = service.upsert_chunks(
            collection_name="test_documents",
            chunks=chunks,
            vectors=vectors,
            batch_size=1,
        )

        assert inserted == 2

        count = service.client.count(
            collection_name="test_documents",
            exact=True,
        ).count

        assert count == 2

        point_id = service.create_point_id(chunks[0].id)

        points = service.client.retrieve(
            collection_name="test_documents",
            ids=[point_id],
            with_payload=True,
        )

        assert len(points) == 1
        assert points[0].payload["document_id"] == "doc-1"
        assert points[0].payload["content"] == chunks[0].content
        assert points[0].payload["title"] == "AI Document"


        service.upsert_chunks(
            collection_name="test_documents",
            chunks=chunks,
            vectors=vectors,
        )

        assert service.client.count(
            collection_name="test_documents",
            exact=True,
        ).count == 2

    finally:
        service.close()


def test_upsert_rejects_invalid_vectors(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        chunk = Chunk(
            id="doc-1:fixed_size:0",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
        )

        with pytest.raises(ValueError):
            service.upsert_chunks(
                collection_name="test_documents",
                chunks=[chunk],
                vectors=[],
            )

        with pytest.raises(ValueError):
            service.upsert_chunks(
                collection_name="test_documents",
                chunks=[chunk],
                vectors=[[0.1] * 10],
            )

        with pytest.raises(ValueError):
            service.upsert_chunks(
                collection_name="test_documents",
                chunks=[chunk],
                vectors=[[0.1] * 384],
                batch_size=0,
            )

    finally:
        service.close()



def test_semantic_search(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        chunks = [
            Chunk(
                id="doc-1:fixed_size:0",
                document_id="doc-1",
                content="Artificial intelligence and machine learning.",
                chunk_index=0,
                metadata={
                    "title": "AI",
                    "source": "ai.pdf",
                    "chunking_strategy": "fixed_size",
                },
            ),
            Chunk(
                id="doc-2:fixed_size:0",
                document_id="doc-2",
                content="Football and basketball are popular sports.",
                chunk_index=0,
                metadata={
                    "title": "Sports",
                    "source": "sports.pdf",
                    "chunking_strategy": "fixed_size",
                },
            ),
        ]

        vectors = [
            [1.0] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,
        ]

        service.upsert_chunks(
            collection_name="test_search",
            chunks=chunks,
            vectors=vectors,
        )

        query_vector = [1.0] + [0.0] * 383

        results = service.search(
            collection_name="test_search",
            query_vector=query_vector,
            top_k=2,
        )

        assert len(results) == 2
        assert results[0]["document_id"] == "doc-1"
        assert results[0]["content"] == chunks[0].content
        assert results[0]["title"] == "AI"
        assert results[0]["score"] > results[1]["score"]

    finally:
        service.close()


def test_search_invalid_parameters(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        service.ensure_collection("test_search")

        with pytest.raises(ValueError):
            service.search(
                collection_name="test_search",
                query_vector=[0.1] * 384,
                top_k=0,
            )

        with pytest.raises(ValueError):
            service.search(
                collection_name="test_search",
                query_vector=[0.1] * 10,
            )

        with pytest.raises(ValueError):
            service.search(
                collection_name="missing_collection",
                query_vector=[0.1] * 384,
            )

    finally:
        service.close()


def test_payload_protects_system_fields(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        chunk = Chunk(
            id="doc-1:fixed_size:0",
            document_id="doc-1",
            content="Correct content",
            chunk_index=0,
            metadata={
                "document_id": "incorrect-id",
                "content": "Incorrect content",
                "title": "Test Document",
            },
        )

        service.upsert_chunks(
            collection_name="test_payload",
            chunks=[chunk],
            vectors=[[0.1] * 384],
        )

        point_id = service.create_point_id(chunk.id)

        points = service.client.retrieve(
            collection_name="test_payload",
            ids=[point_id],
            with_payload=True,
        )

        assert len(points) == 1
        assert points[0].payload["document_id"] == "doc-1"
        assert points[0].payload["content"] == "Correct content"
        assert points[0].payload["title"] == "Test Document"

    finally:
        service.close()


def test_delete_document(tmp_path):
    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        chunks = [
            Chunk(
                id="doc-1:fixed_size:0",
                document_id="doc-1",
                content="First chunk",
                chunk_index=0,
            ),
            Chunk(
                id="doc-1:fixed_size:1",
                document_id="doc-1",
                content="Second chunk",
                chunk_index=1,
            ),
            Chunk(
                id="doc-2:fixed_size:0",
                document_id="doc-2",
                content="Another document",
                chunk_index=0,
            ),
        ]

        service.upsert_chunks(
            collection_name="test_delete",
            chunks=chunks,
            vectors=[[0.1] * 384 for _ in chunks],
        )

        assert service.client.count(
            collection_name="test_delete",
            exact=True,
        ).count == 3

        service.delete_document(
            collection_name="test_delete",
            document_id="doc-1",
        )

        assert service.client.count(
            collection_name="test_delete",
            exact=True,
        ).count == 1

        remaining = service.client.scroll(
            collection_name="test_delete",
            limit=10,
            with_payload=True,
        )[0]

        assert len(remaining) == 1
        assert remaining[0].payload["document_id"] == "doc-2"

    finally:
        service.close()



def test_get_all_chunks(tmp_path):
    from app.services.qdrant_service import QdrantService
    from dataset.chunking import Chunk

    service = QdrantService(storage_path=str(tmp_path / "qdrant"))

    chunks = [
        Chunk(
            id=f"doc-1:fixed_size:{index}",
            document_id="doc-1",
            content=f"Content {index}",
            chunk_index=index,
            metadata={"title": "Test Document", "source": "test"},
        )
        for index in range(5)
    ]

    vectors = [[0.1] * 384 for _ in chunks]

    try:
        service.upsert_chunks(
            collection_name="test_scroll",
            chunks=chunks,
            vectors=vectors,
        )

        results = service.get_all_chunks(
            collection_name="test_scroll",
            batch_size=2,
        )

        assert len(results) == 5
        assert {item["chunk_id"] for item in results} == {
            chunk.id for chunk in chunks
        }
        assert all("content" in item for item in results)

    finally:
        service.close()