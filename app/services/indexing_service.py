from pathlib import Path

from app.services.document_service import chunk_documents, read_document_batches
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


def index_documents(
    file_path: str | Path,
    collection_name: str = "hotpotqa_50k_hybrid",
    batch_size: int = 16,
    max_documents: int | None = 100,
    qdrant_url: str = "http://localhost:6333",
    resume: bool = True,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be greater than zero")

    qdrant_service = QdrantService(url=qdrant_url)

    processed_documents = 0
    total_chunks = 0
    reused_chunks = 0
    new_chunks = 0

    try:
        collection_exists = qdrant_service.client.collection_exists(
            collection_name
        )

        if collection_exists and not resume:
            raise ValueError(
                "Collection already exists. "
                "Enable resume or choose a new collection."
            )

        if not collection_exists:
            qdrant_service.ensure_collection(collection_name)

        embedding_service = EmbeddingService()

        for documents in read_document_batches(
            file_path,
            batch_size=batch_size,
        ):
            if max_documents is not None:
                remaining = max_documents - processed_documents

                if remaining <= 0:
                    break

                documents = documents[:remaining]

            chunks = chunk_documents(
                documents=documents,
                strategy="fixed_size",
                chunk_size=500,
                overlap=50,
            )

            missing_chunks = []

            if resume and chunks:
                point_ids = [
                    qdrant_service.create_point_id(chunk.id)
                    for chunk in chunks
                ]

                existing_points = {}

                for start in range(0, len(point_ids), 128):
                    points = qdrant_service.client.retrieve(
                        collection_name=collection_name,
                        ids=point_ids[start:start + 128],
                        with_payload=True,
                        with_vectors=False,
                    )

                    for point in points:
                        existing_points[str(point.id)] = point

                for chunk, point_id in zip(
                    chunks,
                    point_ids,
                    strict=True,
                ):
                    existing_point = existing_points.get(point_id)

                    if existing_point is None:
                        missing_chunks.append(chunk)
                        continue

                    payload = existing_point.payload or {}

                    if (
                        payload.get("chunk_id") != chunk.id
                        or payload.get("document_id") != chunk.document_id
                        or payload.get("content") != chunk.content
                    ):
                        raise ValueError(
                            f"Existing chunk data mismatch: {chunk.id}"
                        )

                reused_chunks += len(chunks) - len(missing_chunks)
            else:
                missing_chunks = chunks

            if missing_chunks:
                vectors = embedding_service.embed_chunks(missing_chunks)

                qdrant_service.upsert_chunks(
                    collection_name=collection_name,
                    chunks=missing_chunks,
                    vectors=vectors,
                    batch_size=32,
                )

                new_chunks += len(missing_chunks)

            processed_documents += len(documents)
            total_chunks += len(chunks)

            print(
                f"Processed documents: {processed_documents}, "
                f"total chunks: {total_chunks}, "
                f"reused chunks: {reused_chunks}, "
                f"new chunks: {new_chunks}",
                flush=True,
            )

        return {
            "processed_documents": processed_documents,
            "total_chunks": total_chunks,
            "reused_chunks": reused_chunks,
            "new_chunks": new_chunks,
            "collection_name": collection_name,
        }

    finally:
        qdrant_service.close()