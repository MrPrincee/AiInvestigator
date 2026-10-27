from app.services.qdrant_service import QdrantService
from app.services.crud_services.common import (
    document_points,
    user_document_filter,
)


def get_document(
    qdrant_service: QdrantService,
    collection_name: str,
    document_id: str,
) -> dict | None:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    points = document_points(
        qdrant_service,
        collection_name,
        document_id,
    )

    if not points:
        return None

    first_chunk = next(
        (
            point.payload or {}
            for point in points
            if (point.payload or {}).get("chunk_index") == 0
        ),
        None,
    )

    if first_chunk is None:
        raise ValueError("Document's first chunk is missing")

    content = first_chunk.get("document_content")

    if content is None:
        raise ValueError("Original document content is missing")

    system_fields = {
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
        "document_type",
        "managed_by",
    }

    metadata = {
        key: value
        for key, value in first_chunk.items()
        if key not in system_fields
    }

    return {
        "document_id": document_id,
        "title": first_chunk.get("title"),
        "content": content,
        "source": first_chunk.get("source"),
        "metadata": metadata,
    }


def list_documents(
    qdrant_service: QdrantService,
    collection_name: str,
) -> dict:
    client = qdrant_service.client

    if not client.collection_exists(collection_name):
        return {
            "total": 0,
            "documents": [],
        }

    documents = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=user_document_filter(),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    "chunk_count": 0,
                }

            documents[document_id]["chunk_count"] += 1

        if offset is None:
            break

    result = sorted(
        documents.values(),
        key=lambda item: item["document_id"],
    )

    return {
        "total": len(result),
        "documents": result,
    }