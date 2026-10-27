from app.services.qdrant_service import QdrantService


def get_document(
    qdrant_service: QdrantService,
    collection_name: str,
    document_id: str,
) -> dict | None:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    chunks = qdrant_service.get_document_chunks(
        collection_name=collection_name,
        document_id=document_id,
    )

    if not chunks:
        return None

    first_chunk = chunks[0]
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
    documents = qdrant_service.list_documents(
        collection_name=collection_name,
    )

    documents.sort(
        key=lambda document: document["document_id"]
    )

    return {
        "total": len(documents),
        "documents": documents,
    }