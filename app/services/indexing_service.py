from pathlib import Path

from app.services.document_service import chunk_documents, read_document_batches
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


def index_documents(
    file_path: str | Path,
    collection_name: str = "hotpotqa_fixed_size",
    batch_size: int = 16,
    max_documents: int | None = 100,
    qdrant_url: str = "http://localhost:6333",
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be greater than zero")

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService(url=qdrant_url)

    indexed_documents = 0
    indexed_chunks = 0

    try:
        for documents in read_document_batches(file_path, batch_size=batch_size):
            if max_documents is not None:
                remaining = max_documents - indexed_documents

                if remaining <= 0:
                    break

                documents = documents[:remaining]

            chunks = chunk_documents(
                documents=documents,
                strategy="fixed_size",
                chunk_size=500,
                overlap=50,
            )

            if chunks:
                vectors = embedding_service.embed_chunks(chunks)

                qdrant_service.upsert_chunks(
                    collection_name=collection_name,
                    chunks=chunks,
                    vectors=vectors,
                    batch_size=32,
                )

            indexed_documents += len(documents)
            indexed_chunks += len(chunks)

            print(
                f"Indexed documents: {indexed_documents}, "
                f"chunks: {indexed_chunks}"
            )

        return {
            "indexed_documents": indexed_documents,
            "indexed_chunks": indexed_chunks,
            "collection_name": collection_name,
        }

    finally:
        qdrant_service.close()