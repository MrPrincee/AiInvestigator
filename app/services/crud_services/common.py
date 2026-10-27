import hashlib

from qdrant_client import models


RESERVED_FIELDS = {
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


def user_document_filter(document_id: str | None = None) -> models.Filter:
    conditions = [
        models.FieldCondition(
            key="managed_by",
            match=models.MatchValue(value="crud"),
        ),
        models.FieldCondition(
            key="document_type",
            match=models.MatchValue(value="user"),
        ),
    ]

    if document_id is not None:
        conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )

    return models.Filter(must=conditions)


def validate_document(
    title: str,
    content: str,
    metadata: dict | None,
    chunking_strategy: str,
) -> tuple[str, str, dict]:
    title = title.strip()
    content = content.strip()

    if not title:
        raise ValueError("Document title cannot be empty")

    if not content:
        raise ValueError("Document content cannot be empty")

    if chunking_strategy not in {"fixed_size", "sentence_aware"}:
        raise ValueError("Unsupported chunking strategy")

    metadata = dict(metadata or {})

    if RESERVED_FIELDS.intersection(metadata):
        raise ValueError("Metadata contains reserved field names")

    metadata["document_type"] = "user"
    metadata["managed_by"] = "crud"

    return title, content, metadata


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def document_points(
    qdrant_service,
    collection_name: str,
    document_id: str,
    with_payload: bool = True,
) -> list:
    client = qdrant_service.client

    if not client.collection_exists(collection_name):
        return []

    result = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=user_document_filter(document_id),
            limit=100,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )

        result.extend(points)

        if offset is None:
            break

    return result


def duplicate_exists(
    qdrant_service,
    collection_name: str,
    digest: str,
    exclude_document_id: str | None = None,
) -> bool:
    client = qdrant_service.client

    if not client.collection_exists(collection_name):
        return False

    conditions = list(user_document_filter().must)

    conditions.append(
        models.FieldCondition(
            key="content_hash",
            match=models.MatchValue(value=digest),
        )
    )

    excluded = []

    if exclude_document_id is not None:
        excluded.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=exclude_document_id),
            )
        )

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=conditions,
            must_not=excluded or None,
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    return bool(points)


def delete_points(
    qdrant_service,
    collection_name: str,
    point_ids: list,
) -> None:
    if not point_ids:
        return

    qdrant_service.client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=point_ids),
        wait=True,
    )