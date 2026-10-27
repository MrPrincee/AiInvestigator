from app.services.qdrant_service import QdrantService
from app.services.crud_services.common import (
    delete_points,
    document_points,
)


def delete_document(
    qdrant_service: QdrantService,
    collection_name: str,
    document_id: str,
) -> dict:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    points = document_points(
        qdrant_service,
        collection_name,
        document_id,
        with_payload=False,
    )

    if not points:
        raise LookupError("Document not found")

    point_ids = [
        point.id
        for point in points
    ]

    delete_points(
        qdrant_service,
        collection_name,
        point_ids,
    )

    remaining = document_points(
        qdrant_service,
        collection_name,
        document_id,
        with_payload=False,
    )

    if remaining:
        raise RuntimeError(
            "Document deletion could not be verified"
        )

    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_affected": len(point_ids),
    }