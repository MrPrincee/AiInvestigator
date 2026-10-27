import time

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


SOURCE_COLLECTION = "hotpotqa_50k_fixed_size"
TARGET_COLLECTION = "hotpotqa_50k_hybrid"
BATCH_SIZE = 100


def migrate() -> None:
    client = QdrantClient(url="http://localhost:6333", timeout=120)
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    try:
        source_count = client.count(
            collection_name=SOURCE_COLLECTION,
            exact=True,
        ).count

        print(f"Source chunks: {source_count}", flush=True)

        offset = None
        processed = 0
        started_at = time.perf_counter()

        while True:
            points, offset = client.scroll(
                collection_name=SOURCE_COLLECTION,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            texts = [
                point.payload["content"]
                for point in points
            ]

            sparse_vectors = list(sparse_model.embed(texts))

            migrated_points = []

            for point, sparse_vector in zip(
                points,
                sparse_vectors,
                strict=True,
            ):
                migrated_points.append(
                    models.PointStruct(
                        id=point.id,
                        vector={
                            "dense": point.vector,
                            "bm25": models.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist(),
                            ),
                        },
                        payload=point.payload,
                    )
                )

            client.upsert(
                collection_name=TARGET_COLLECTION,
                points=migrated_points,
                wait=True,
            )

            processed += len(points)

            elapsed = time.perf_counter() - started_at

            print(
                f"Migrated {processed}/{source_count} chunks "
                f"in {elapsed:.1f}s",
                flush=True,
            )

            if offset is None:
                break

        target_count = client.count(
            collection_name=TARGET_COLLECTION,
            exact=True,
        ).count

        print(
            f"Migration complete. "
            f"Source: {source_count}, Target: {target_count}",
            flush=True,
        )

        if target_count != source_count:
            raise RuntimeError(
                "Migration verification failed: "
                "source and target counts differ."
            )

    finally:
        client.close()


if __name__ == "__main__":
    migrate()