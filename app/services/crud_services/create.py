from uuid import uuid4

from dataset.models import Document
from app.services.document_service import chunk_documents
from app.services.crud_services.common import (
    content_hash,
    delete_points,
    duplicate_exists,
    validate_document,
)


def create_document(
    qdrant_service,
    embedding_service,
    collection_name: str,
    title: str,
    content: str,
    source: str = "user",
    metadata: dict | None = None,
    chunking_strategy: str = "fixed_size",
) -> dict:
    title, content, metadata = validate_document(
        title,
        content,
        metadata,
        chunking_strategy,
    )

    digest = content_hash(content)

    if duplicate_exists(
        qdrant_service,
        collection_name,
        digest,
    ):
        raise ValueError(
            "A document with the same content already exists"
        )

    document_id = str(uuid4())

    document = Document(
        id=document_id,
        title=title,
        content=content,
        source=source,
        metadata=metadata,
    )

    chunks = chunk_documents(
        documents=[document],
        strategy=chunking_strategy,
        chunk_size=500,
        overlap=50,
    )

    if not chunks:
        raise ValueError("Document produced no chunks")

    for chunk in chunks:
        chunk.metadata["content_hash"] = digest

        if chunk.chunk_index == 0:
            chunk.metadata["document_content"] = content

    vectors = embedding_service.embed_chunks(chunks)

    point_ids = [
        qdrant_service.create_point_id(chunk.id)
        for chunk in chunks
    ]

    try:
        count = qdrant_service.upsert_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            batch_size=32,
        )
    except Exception:
        if qdrant_service.client.collection_exists(collection_name):
            try:
                delete_points(
                    qdrant_service,
                    collection_name,
                    point_ids,
                )
            except Exception as cleanup_error:
                raise RuntimeError(
                    "Document creation failed and cleanup was unsuccessful"
                ) from cleanup_error

        raise

    return {
        "status": "created",
        "document_id": document_id,
        "chunks_affected": count,
    }