from uuid import uuid4

from dataset.models import Document
from app.services.document_service import chunk_documents
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.crud_services.common import (
    content_hash,
    delete_points,
    document_points,
    duplicate_exists,
    validate_document,
)
from app.services.crud_services.read import get_document


def update_document(
    qdrant_service: QdrantService,
    embedding_service: EmbeddingService,
    collection_name: str,
    document_id: str,
    title: str,
    content: str,
    metadata: dict | None = None,
    chunking_strategy: str = "fixed_size",
) -> dict:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    title, content, metadata = validate_document(
        title,
        content,
        metadata,
        chunking_strategy,
    )

    existing_document = get_document(
        qdrant_service,
        collection_name,
        document_id,
    )

    if existing_document is None:
        raise LookupError("Document not found")

    digest = content_hash(content)

    if duplicate_exists(
        qdrant_service,
        collection_name,
        digest,
        exclude_document_id=document_id,
    ):
        raise ValueError(
            "A different document with the same content already exists"
        )

    old_points = document_points(
        qdrant_service,
        collection_name,
        document_id,
        with_payload=False,
    )

    if not old_points:
        raise LookupError("Document not found")

    document = Document(
        id=document_id,
        title=title,
        content=content,
        source=existing_document["source"],
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

    version = str(uuid4())

    for chunk in chunks:
        chunk.id = f"{chunk.id}:{version}"
        chunk.metadata["content_hash"] = digest

        if chunk.chunk_index == 0:
            chunk.metadata["document_content"] = content

    vectors = embedding_service.embed_chunks(chunks)

    new_point_ids = [
        qdrant_service.create_point_id(chunk.id)
        for chunk in chunks
    ]

    old_point_ids = [
        point.id
        for point in old_points
    ]

    try:
        qdrant_service.upsert_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            batch_size=32,
        )
    except Exception:
        try:
            delete_points(
                qdrant_service,
                collection_name,
                new_point_ids,
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "Document update failed and cleanup was unsuccessful"
            ) from cleanup_error

        raise

    delete_points(
        qdrant_service,
        collection_name,
        old_point_ids,
    )

    return {
        "status": "updated",
        "document_id": document_id,
        "chunks_affected": len(chunks),
    }