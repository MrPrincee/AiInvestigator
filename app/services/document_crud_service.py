from app.services.crud_services.create import create_document
from app.services.crud_services.read import get_document, list_documents
from app.services.crud_services.update import update_document
from app.services.crud_services.delete import delete_document

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.settings import settings


class DocumentCRUDService:
    COLLECTION_NAME = settings.QDRANT_COLLECTION

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_service: QdrantService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        if qdrant_url is not None and qdrant_service is not None:
            raise ValueError("Provide either qdrant_url or qdrant_service, not both")

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
        if self._owns_qdrant_service:
            self.qdrant_service.close()
