import hashlib
from uuid import uuid4

from qdrant_client import models

from dataset.models import Document
from app.services.document_service import chunk_documents


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
    title = title.strip()
    content = content.strip()

    if not title:
        raise ValueError("Document title cannot be empty")

    if not content:
        raise ValueError("Document content cannot be empty")

    if chunking_strategy not in {"fixed_size", "sentence_aware"}:
        raise ValueError("Unsupported chunking strategy")

    metadata = dict(metadata or {})

    reserved_fields = {
        "id",
        "document_id",
        "chunk_id",
        "chunk_index",
        "content",
        "title",
        "source",
        "chunking_strategy",
        "document_content",
        "content_hash",
    }

    if reserved_fields.intersection(metadata):
        raise ValueError("Metadata contains reserved field names")

    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    if qdrant_service.client.collection_exists(collection_name):
        existing_points, _ = qdrant_service.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if existing_points:
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
        chunk.metadata["content_hash"] = content_hash

        if chunk.chunk_index == 0:
            chunk.metadata["document_content"] = content

    vectors = embedding_service.embed_chunks(chunks)

    try:
        chunk_count = qdrant_service.upsert_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            batch_size=32,
        )
    except Exception:
        if qdrant_service.client.collection_exists(collection_name):
            qdrant_service.delete_document(
                collection_name=collection_name,
                document_id=document_id,
            )
        raise

    return {
        "status": "created",
        "document_id": document_id,
        "chunks_affected": chunk_count,
    }