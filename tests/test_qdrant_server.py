from uuid import uuid4

from app.services.qdrant_service import QdrantService
from dataset.chunking import Chunk


def test_qdrant_server_connection():
    service = QdrantService(
        url="http://localhost:6333"
    )

    collection_name = f"integration_test_{uuid4().hex}"

    try:
        assert service.client.get_collections() is not None

        chunk = Chunk(
            id="server-test:fixed_size:0",
            document_id="server-test",
            content="Qdrant server integration test",
            chunk_index=0,
            metadata={
                "title": "Integration Test",
                "source": "test",
            },
        )

        service.upsert_chunks(
            collection_name=collection_name,
            chunks=[chunk],
            vectors=[[0.1] * 384],
        )

        results = service.search(
            collection_name=collection_name,
            query_vector=[0.1] * 384,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0]["document_id"] == "server-test"
        assert results[0]["title"] == "Integration Test"

    finally:
        try:
            if service.client.collection_exists(collection_name):
                service.client.delete_collection(collection_name)
        finally:
            service.close()