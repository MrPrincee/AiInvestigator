from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from dataset.chunking import Chunk
from app.settings import settings


class QdrantService:
    VECTOR_SIZE = 384
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(
            self,
            storage_path: str = "data/qdrant",
            url: str | None = None,
    ):
        qdrant_url = url or settings.QDRANT_URL
        qdrant_api_key = settings.QDRANT_API_KEY

        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=120,
            )
        else:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)

        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    def ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
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

            texts = [chunk.content for chunk in batch_chunks]
            sparse_vectors = list(self.sparse_model.embed(texts))

            points = []

            for chunk, vector, sparse_vector in zip(
                batch_chunks,
                batch_vectors,
                sparse_vectors,
                strict=True,
            ):
                points.append(
                    models.PointStruct(
                        id=self.create_point_id(chunk.id),
                        vector={
                            self.DENSE_VECTOR_NAME: vector,
                            self.SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist(),
                            ),
                        },
                        payload={
                            **chunk.metadata,
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                        },
                    )
                )

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

        return len(chunks)

    @staticmethod
    def _format_results(response) -> list[dict]:
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "document_id": (point.payload or {}).get("document_id"),
                "chunk_id": (point.payload or {}).get("chunk_id"),
                "chunk_index": (point.payload or {}).get("chunk_index"),
                "content": (point.payload or {}).get("content"),
                "title": (point.payload or {}).get("title"),
                "source": (point.payload or {}).get("source"),
                "chunking_strategy": (point.payload or {}).get(
                    "chunking_strategy"
                ),
                "page_number": (point.payload or {}).get("page_number"),
            }
            for point in response.points
        ]

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
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )

        return self._format_results(response)

    def keyword_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self.client.collection_exists(collection_name):
            raise ValueError(
                f"Collection does not exist: {collection_name}"
            )

        sparse_embedding = next(
            iter(self.sparse_model.query_embed(query))
        )

        sparse_vector = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )

        response = self.client.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )

        return self._format_results(response)

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
            raise ValueError(
                f"Collection does not exist: {collection_name}"
            )

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
                    **(point.payload or {}),
                    "id": str(point.id),
                })

            if offset is None:
                break

        return chunks

    def get_document_chunks(
        self,
        collection_name: str,
        document_id: str,
    ) -> list[dict]:
        if not document_id:
            raise ValueError("document_id cannot be empty")

        if not self.client.collection_exists(collection_name):
            return []

        chunks = []
        offset = None

        while True:
            points, offset = self.client.scroll(
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

            for point in points:
                chunks.append({
                    **(point.payload or {}),
                    "id": str(point.id),
                })

            if offset is None:
                break

        return sorted(
            chunks,
            key=lambda chunk: chunk.get("chunk_index", 0),
        )

    def list_documents(
        self,
        collection_name: str,
    ) -> list[dict]:
        if not self.client.collection_exists(collection_name):
            return []

        documents = {}
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
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

        return list(documents.values())

    def close(self) -> None:
        self.client.close()
