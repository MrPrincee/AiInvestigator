from app.services.crud_services.create import create_document
from app.services.crud_services.read import get_document, list_documents
from app.services.crud_services.update import update_document
from app.services.crud_services.delete import delete_document

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


class DocumentCRUDService:
    COLLECTION_NAME = "hotpotqa_50k_hybrid"

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
    ):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService(url=qdrant_url)

    def create_document(
        self,
        title: str,
        content: str,
        source: str = "user",
        metadata: dict | None = None,
        chunking_strategy: str = "fixed_size",
    ) -> dict:
        return create_document(
            qdrant_service=self.qdrant_service,
            embedding_service=self.embedding_service,
            collection_name=self.COLLECTION_NAME,
            title=title,
            content=content,
            source=source,
            metadata=metadata,
            chunking_strategy=chunking_strategy,
        )

    def get_document(self, document_id: str) -> dict | None:
        return get_document(
            qdrant_service=self.qdrant_service,
            collection_name=self.COLLECTION_NAME,
            document_id=document_id,
        )

    def list_documents(self) -> dict:
        return list_documents(
            qdrant_service=self.qdrant_service,
            collection_name=self.COLLECTION_NAME,
        )

    def update_document(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: dict | None = None,
        chunking_strategy: str = "fixed_size",
    ) -> dict:
        return update_document(
            qdrant_service=self.qdrant_service,
            embedding_service=self.embedding_service,
            collection_name=self.COLLECTION_NAME,
            document_id=document_id,
            title=title,
            content=content,
            metadata=metadata,
            chunking_strategy=chunking_strategy,
        )

    def delete_document(self, document_id: str) -> dict:
        return delete_document(
            qdrant_service=self.qdrant_service,
            collection_name=self.COLLECTION_NAME,
            document_id=document_id,
        )

    def close(self) -> None:
        self.qdrant_service.close()