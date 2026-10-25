from pathlib import Path

from qdrant_client import QdrantClient, models

from uuid import NAMESPACE_URL, uuid5

from dataset.chunking import Chunk

class QdrantService:
    VECTOR_SIZE = 384

    def __init__(
            self,
            storage_path: str = "data/qdrant",
            url: str | None = None,
    ):
        if url is not None:
            self.client = QdrantClient(url=url)
        else:
            Path(storage_path).mkdir(
                parents=True,
                exist_ok=True,
            )
            self.client = QdrantClient(path=storage_path)

    def ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def create_point_id(chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> int:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Chunks and vectors must have the same length"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        for vector in vectors:
            if len(vector) != self.VECTOR_SIZE:
                raise ValueError(
                    f"Expected {self.VECTOR_SIZE}-dimensional vector"
                )

        self.ensure_collection(collection_name)

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]

            points = []

            for chunk, vector in zip(
                batch_chunks,
                batch_vectors,
                strict=True,
            ):
                point = models.PointStruct(
                    id=self.create_point_id(chunk.id),
                    vector=vector,
                    payload={
                        **chunk.metadata,
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                    },
                )

                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

        return len(chunks)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if len(query_vector) != self.VECTOR_SIZE:
            raise ValueError(
                f"Expected {self.VECTOR_SIZE}-dimensional query vector"
            )

        if not self.client.collection_exists(collection_name):
            raise ValueError(
                f"Collection does not exist: {collection_name}"
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "document_id": point.payload.get("document_id"),
                "chunk_id": point.payload.get("chunk_id"),
                "chunk_index": point.payload.get("chunk_index"),
                "content": point.payload.get("content"),
                "title": point.payload.get("title"),
                "source": point.payload.get("source"),
                "chunking_strategy": point.payload.get(
                    "chunking_strategy"
                ),
                "page_number": point.payload.get("page_number"),
            }
            for point in response.points
        ]


    def delete_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:

        if not document_id:
            raise ValueError("document_id cannot be empty")

        if not self.client.collection_exists(collection_name):
            raise ValueError(
                f"Collection does not exist: {collection_name}"
            )

        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def get_all_chunks(
            self,
            collection_name: str,
            batch_size: int = 100,
    ) -> list[dict]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection does not exist: {collection_name}")

        chunks = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                chunks.append({
                    **point.payload,
                    "id": str(point.id),
                })

            if offset is None:
                break

        return chunks


    def close(self) -> None:
        self.client.close()