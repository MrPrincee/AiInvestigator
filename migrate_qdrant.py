import os
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

COLLECTION_NAME = "hotpotqa_50k_hybrid"
BATCH_SIZE = 100
TARGET_POINTS = 78181


def migrate():
    local_client = QdrantClient(
        url="http://127.0.0.1:6333",
        timeout=120,
    )

    cloud_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120,
    )

    transferred = 0
    offset = None
    started_at = time.perf_counter()

    try:
        if not local_client.collection_exists(COLLECTION_NAME):
            raise RuntimeError("Local collection does not exist")

        if not cloud_client.collection_exists(COLLECTION_NAME):
            raise RuntimeError("Cloud collection does not exist")

        while transferred < TARGET_POINTS:
            remaining = TARGET_POINTS - transferred
            batch_limit = min(BATCH_SIZE, remaining)

            points, offset = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            cloud_points = [
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ]

            cloud_client.upsert(
                collection_name=COLLECTION_NAME,
                points=cloud_points,
                wait=True,
            )

            transferred += len(cloud_points)

            if transferred % 500 == 0 or transferred == TARGET_POINTS:
                elapsed = time.perf_counter() - started_at

                print(
                    f"Transferred {transferred}/{TARGET_POINTS} "
                    f"points in {elapsed:.1f}s",
                    flush=True,
                )

            if offset is None:
                break

        cloud_count = cloud_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        elapsed = time.perf_counter() - started_at

        print()
        print("Migration completed")
        print(f"Transferred this run: {transferred}")
        print(f"Cloud total points: {cloud_count}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

    finally:
        local_client.close()
        cloud_client.close()


if __name__ == "__main__":
    migrate()