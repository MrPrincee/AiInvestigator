from app.services.qdrant_service import QdrantService


def delete_document(
    qdrant_service: QdrantService,
    collection_name: str,
    document_id: str,
) -> dict:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    chunks = qdrant_service.get_document_chunks(
        collection_name=collection_name,
        document_id=document_id,
    )

    if not chunks:
        raise LookupError("Document not found")

    qdrant_service.delete_document(
        collection_name=collection_name,
        document_id=document_id,
    )

    remaining_chunks = qdrant_service.get_document_chunks(
        collection_name=collection_name,
        document_id=document_id,
    )

    if remaining_chunks:
        raise RuntimeError(
            "Document deletion could not be verified"
        )

    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_affected": len(chunks),
    }