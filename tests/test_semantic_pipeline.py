import pytest

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from dataset.chunking import fixed_size_chunking
from dataset.models import Document


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_end_to_end_semantic_search(tmp_path, embedding_service):
    documents = [
        Document(
            id="doc-ai",
            title="Machine Learning",
            content=(
                "Machine learning enables computers to learn patterns "
                "from data without being explicitly programmed."
            ),
            source="test",
        ),
        Document(
            id="doc-sports",
            title="Football",
            content=(
                "Football is a team sport where players score goals "
                "by moving a ball into the opposing team's net."
            ),
            source="test",
        ),
        Document(
            id="doc-space",
            title="Astronomy",
            content=(
                "The Moon is Earth's natural satellite and orbits "
                "around our planet."
            ),
            source="test",
        ),
    ]

    chunks = []

    for document in documents:
        chunks.extend(
            fixed_size_chunking(
                document,
                chunk_size=500,
                overlap=50,
            )
        )

    vectors = embedding_service.embed_chunks(chunks)

    service = QdrantService(
        storage_path=str(tmp_path / "qdrant")
    )

    try:
        service.upsert_chunks(
            collection_name="semantic_test",
            chunks=chunks,
            vectors=vectors,
        )

        question = "How can computers learn from examples?"

        query_vector = embedding_service.embed_texts(
            [question]
        )[0]

        results = service.search(
            collection_name="semantic_test",
            query_vector=query_vector,
            top_k=3,
        )

        assert len(results) == 3
        assert results[0]["document_id"] == "doc-ai"
        assert results[0]["title"] == "Machine Learning"
        assert results[0]["score"] > 0

    finally:
        service.close()