from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.rrf_service import reciprocal_rank_fusion
from app.settings import settings


class HybridSearchService:
    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_service: QdrantService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        if qdrant_url is not None and qdrant_service is not None:
            raise ValueError(
                "Provide either qdrant_url or qdrant_service, not both"
            )

        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._owns_qdrant_service = qdrant_service is None

        self.qdrant_service = (
            qdrant_service
            if qdrant_service is not None
            else QdrantService(url=qdrant_url)
        )

        try:
            self.embedding_service = (
                embedding_service
                if embedding_service is not None
                else EmbeddingService()
            )
        except Exception:
            if self._owns_qdrant_service:
                self.qdrant_service.close()
            raise

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if mode not in {"semantic", "keyword", "hybrid"}:
            raise ValueError(
                "Mode must be 'semantic', 'keyword', or 'hybrid'"
            )

        if mode == "keyword":
            return self.qdrant_service.keyword_search(
                collection_name=self.collection_name,
                query=query,
                top_k=top_k,
            )

        query_vector = self.embedding_service.embed_texts([query])[0]

        if mode == "semantic":
            return self.qdrant_service.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )

        candidate_count = max(top_k * 4, 20)

        semantic_results = self.qdrant_service.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=candidate_count,
        )

        keyword_results = self.qdrant_service.keyword_search(
            collection_name=self.collection_name,
            query=query,
            top_k=candidate_count,
        )

        return reciprocal_rank_fusion(
            result_lists=[semantic_results, keyword_results],
            top_k=top_k,
        )

    def close(self) -> None:
        if self._owns_qdrant_service:
            self.qdrant_service.close()