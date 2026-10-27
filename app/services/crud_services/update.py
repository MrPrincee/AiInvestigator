import hashlib
from uuid import uuid4

from qdrant_client import models

from dataset.models import Document
from app.services.document_service import chunk_documents
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
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

    title = title.strip()
    content = content.strip()

    if not title:
        raise ValueError("Document title cannot be empty")

    if not content:
        raise ValueError("Document content cannot be empty")

    if chunking_strategy not in {"fixed_size", "sentence_aware"}:
        raise ValueError("Unsupported chunking strategy")

    existing_document = get_document(
        qdrant_service=qdrant_service,
        collection_name=collection_name,
        document_id=document_id,
    )

    if existing_document is None:
        raise LookupError("Document not found")

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

    duplicate_points, _ = qdrant_service.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="content_hash",
                    match=models.MatchValue(value=content_hash),
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ],
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if duplicate_points:
        raise ValueError(
            "A different document with the same content already exists"
        )

    old_points = []
    offset = None

    while True:
        points, offset = qdrant_service.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        old_points.extend(points)

        if offset is None:
            break

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
        chunk.metadata["content_hash"] = content_hash

        if chunk.chunk_index == 0:
            chunk.metadata["document_content"] = content

    vectors = embedding_service.embed_chunks(chunks)

    new_point_ids = [
        qdrant_service.create_point_id(chunk.id)
        for chunk in chunks
    ]

    try:
        qdrant_service.upsert_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            batch_size=32,
        )

        old_point_ids = [point.id for point in old_points]

        if old_point_ids:
            qdrant_service.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=old_point_ids,
                ),
                wait=True,
            )

    except Exception:
        qdrant_service.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=new_point_ids,
            ),
            wait=True,
        )
        raise

    return {
        "status": "updated",
        "document_id": document_id,
        "chunks_affected": len(chunks),
    }